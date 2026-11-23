"""Fast recursive, Oracle-pure grouping in full CatBoost SHAP space."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.tree import DecisionTreeClassifier

from legacy.experiments.shap_family_propagation import (
    REVIEW_BATCH_SIZE,
    REVIEW_ROUNDS,
    SEED,
    TRUSTED_FRACTION,
    feature_matrix,
    fit_split_model,
    load_initial,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "legacy" / "experiments" / "results"


def reconstruct_oracle_clean_model() -> tuple[pd.DataFrame, object, int]:
    """Recreate notebook's 10x50 split model and residual Oracle corrections."""
    frame = load_initial()
    frame["posting_id"] = frame["posting_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    frame["pred_label"] = frame["pred_label"].astype(int)
    ids = frame["posting_id"]
    uncertainty = 1.0 - 2.0 * np.abs(frame["pred_score"].to_numpy(dtype=float) - 0.5)
    tie_break = np.random.RandomState(SEED).random_sample(len(frame))
    trusted_order = np.lexsort((tie_break, uncertainty))
    trusted_ids = set(ids.iloc[trusted_order[: round(TRUSTED_FRACTION * len(frame))]])
    pseudo = dict(zip(ids, frame["pred_label"].to_numpy(dtype=int)))
    trusted_targets = {pid: pseudo[pid] for pid in trusted_ids}
    trusted_mask = ids.isin(trusted_ids).to_numpy()

    review_labels: dict[str, int] = {}
    reviewed_ids: set[str] = set()
    model = None
    for round_i in range(REVIEW_ROUNDS):
        if round_i == 0:
            query_u = uncertainty
        else:
            score = model.predict_proba(Pool(feature_matrix(frame)))[:, 1]
            query_u = 1.0 - 2.0 * np.abs(score - 0.5)
        eligible = ~(trusted_mask | ids.isin(reviewed_ids).to_numpy())
        rows = np.flatnonzero(eligible)
        order = rows[np.lexsort((tie_break[rows], -query_u[rows]))]
        batch_ids = ids.iloc[order[:REVIEW_BATCH_SIZE]].tolist()
        reviewed_ids.update(batch_ids)
        review_labels.update({
            pid: int(frame.loc[ids == pid, "label"].iloc[0]) for pid in batch_ids
        })
        model = fit_split_model(frame, trusted_ids, trusted_targets, review_labels)
        print(f"Split-model replay {round_i + 1}/{REVIEW_ROUNDS}")

    oracle_reviews = dict(review_labels)
    corrections = 0
    for cleanup_i in range(5):
        pred = np.asarray(model.predict(Pool(feature_matrix(frame)))).reshape(-1).astype(int)
        errors = np.flatnonzero(pred != frame["label"].to_numpy(dtype=int))
        new_ids = [ids.iloc[i] for i in errors if ids.iloc[i] not in oracle_reviews]
        print(f"Oracle correction pass {cleanup_i + 1}: {len(new_ids)} new labels")
        if not new_ids:
            break
        oracle_reviews.update({pid: int(frame.loc[ids == pid, "label"].iloc[0]) for pid in new_ids})
        corrections += len(new_ids)
        model = fit_split_model(frame, trusted_ids, trusted_targets, oracle_reviews)
    return frame, model, corrections


def extract_leaf_paths(tree: DecisionTreeClassifier, feature_names: list[str]) -> dict[int, list[dict]]:
    structure = tree.tree_
    paths: dict[int, list[dict]] = {}

    def visit(node: int, path: list[dict]) -> None:
        left = int(structure.children_left[node])
        right = int(structure.children_right[node])
        if left == right:
            paths[node] = path
            return
        feature = int(structure.feature[node])
        name = feature_names[feature]
        threshold = float(structure.threshold[node])
        visit(left, path + [{"feature": name, "op": "<=", "threshold": threshold}])
        visit(right, path + [{"feature": name, "op": ">", "threshold": threshold}])

    visit(0, [])
    return paths


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frame, model, n_corrections = reconstruct_oracle_clean_model()
    X = feature_matrix(frame)
    feature_names = list(X.columns)
    labels = frame["label"].to_numpy(dtype=int)

    print(f"Computing SHAP_ALL: {len(frame):,} rows × {len(feature_names)} dimensions")
    shap_values = model.get_feature_importance(Pool(X), type="ShapValues")[:, :-1]
    if shap_values.shape != (len(frame), len(feature_names)):
        raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")
    if not np.isfinite(shap_values).all():
        raise ValueError("SHAP_ALL contains non-finite values.")

    #split on one feature and threshold until each separable leaf is pure
    #do not cap family size or prune
    tree = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=SEED,
    )
    tree_started = time.perf_counter()
    tree.fit(shap_values, labels)
    leaf_nodes = tree.apply(shap_values).astype(int)
    tree_seconds = time.perf_counter() - tree_started
    paths = extract_leaf_paths(tree, feature_names)

    per_leaf = pd.DataFrame({"node_id": leaf_nodes, "label": labels}).groupby("node_id")["label"]
    stats = per_leaf.agg(
        support="size",
        anomalies="sum",
    ).reset_index()
    stats["normals"] = stats["support"] - stats["anomalies"]
    stats["purity"] = stats[["normals", "anomalies"]].max(axis=1) / stats["support"]
    stats["pure"] = (stats["normals"] == 0) | (stats["anomalies"] == 0)
    stats["group_label"] = (stats["anomalies"] > stats["normals"]).astype(int)
    stats["group_id"] = stats["node_id"].map(lambda node: f"L{int(node):06d}")
    stats["path"] = stats["node_id"].map(lambda node: json.dumps(paths[int(node)], ensure_ascii=False))
    stats["features"] = stats["node_id"].map(lambda node: json.dumps(
        list(dict.fromkeys(rule["feature"] for rule in paths[int(node)])), ensure_ascii=False
    ))
    stats["n_rule_features"] = stats["node_id"].map(
        lambda node: len({rule["feature"] for rule in paths[int(node)]})
    )
    stats["depth"] = stats["node_id"].map(lambda node: len(paths[int(node)]))
    stats = stats.sort_values(["pure", "support"], ascending=[False, False]).reset_index(drop=True)

    group_id_by_node = {int(node): f"L{int(node):06d}" for node in stats["node_id"]}
    membership = pd.DataFrame({
        "posting_id": frame["posting_id"].to_numpy(),
        "group_id": [group_id_by_node[int(node)] for node in leaf_nodes],
        "label": labels,
    })
    group_labels = stats.set_index("group_id")["group_label"]
    membership["group_label"] = membership["group_id"].map(group_labels).astype(int)
    membership["correct"] = membership["label"] == membership["group_label"]

    groups_path = RESULTS / "shap_all_groups.csv"
    membership_path = RESULTS / "shap_all_group_membership.csv"
    meta_path = RESULTS / "shap_all_groups_meta.json"
    stats.to_csv(groups_path, index=False)
    membership.to_csv(membership_path, index=False)
    pure = stats.loc[stats["pure"]]
    mixed = stats.loc[~stats["pure"]]
    meta = {
        "run_seed": SEED,
        "shap_space": "shap_all",
        "n_cases": int(len(frame)),
        "n_dimensions": int(len(feature_names)),
        "feature_names": feature_names,
        "oracle_reviewed_direct": 10 * REVIEW_BATCH_SIZE,
        "oracle_residual_corrections": n_corrections,
        "tree_algorithm": "greedy CART; one SHAP feature threshold per split; recurse to pure/no-split leaves",
        "tree_parameters": {
            "criterion": "gini", "max_depth": None,
            "min_samples_split": 2, "min_samples_leaf": 1,
            "random_state": SEED,
        },
        "family_size_cap": None,
        "nodes": int(tree.tree_.node_count),
        "leaves": int(len(stats)),
        "max_depth": int(tree.tree_.max_depth),
        "pure_groups": int(len(pure)),
        "pure_cases": int(pure["support"].sum()),
        "pure_case_coverage": float(pure["support"].sum() / len(frame)),
        "mixed_groups": int(len(mixed)),
        "mixed_cases": int(mixed["support"].sum()),
        "largest_pure_group": int(pure["support"].max()) if len(pure) else 0,
        "median_pure_group": float(pure["support"].median()) if len(pure) else None,
        "mean_rule_features_pure_groups": float(pure["n_rule_features"].mean()) if len(pure) else None,
        "tree_fit_seconds": round(tree_seconds, 2),
        "total_seconds": round(time.perf_counter() - started, 2),
        "uses_oracle_labels_for_group_discovery": True,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Saved groups: {groups_path}")
    print(f"Saved membership: {membership_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()

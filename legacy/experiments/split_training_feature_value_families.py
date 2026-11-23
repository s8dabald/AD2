"""Isolated feature-value propagation modes; baseline pipeline stays unchanged."""

from __future__ import annotations

import json
import argparse
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legacy" / "experiments"))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

from dataprep import full_dataprep
from isolation_forest import run_if
from models.kittyboost import train_catboost
from legacy.experiments.shap_family_propagation import (
    CORRECTED_WEIGHT, ROOT_CACHE, SEED as DEFAULT_SEED, TRUSTED_FRACTION,
    feature_matrix, load_initial,
)
from legacy.experiments.split_training_shap_box_timing import select_batch


RESULTS = Path(__file__).resolve().parent / "results"
SCHEDULES = ((10, 50), (20, 25))
REVIEW_BUDGET = 500
PROPAGATION_CAP = 4
RUN_SEEDS = (42, 7, 123, 2024)
STEM = RESULTS / "split_training_feature_value_families"


def _initial_cache(seed: int):
    """Load existing canonical state or build same IF→CatBoost initial state."""
    if seed == DEFAULT_SEED:
        frame = load_initial()
        frame["posting_id"] = frame["posting_id"].astype(str)
        frame["label"] = frame["label"].astype(int)
        frame["pred_label"] = frame["pred_label"].astype(int)
        model = CatBoostClassifier()
        model.load_model(str(ROOT_CACHE / f"initial_seed_{seed}.cbm"))
        return frame, model

    ROOT_CACHE.mkdir(parents=True, exist_ok=True)
    stem = f"initial_seed_{seed}"
    paths = {
        "frame": ROOT_CACHE / f"{stem}_predictions.pkl",
        "model": ROOT_CACHE / f"{stem}.cbm",
        "meta": ROOT_CACHE / f"{stem}.json",
    }
    if all(path.exists() for path in paths.values()):
        signature = json.loads(paths["meta"].read_text(encoding="utf-8"))
        if signature.get("feature_value_runner") is True and signature.get("seed") == seed:
            with paths["frame"].open("rb") as handle:
                frame = pickle.load(handle)
            model = CatBoostClassifier()
            model.load_model(str(paths["model"]))
            return frame, model

    raw, labels = full_dataprep()
    initial = run_if(raw, labels, verbose=False, seed=seed)
    fitted = train_catboost(initial, verbose=False, compute_shap=False, seed=seed)
    frame, model = fitted[0].reset_index(drop=True), fitted[4]
    frame["posting_id"] = frame["posting_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    frame["pred_label"] = frame["pred_label"].astype(int)
    with paths["frame"].open("wb") as handle:
        pickle.dump(frame, handle, protocol=pickle.HIGHEST_PROTOCOL)
    model.save_model(str(paths["model"]))
    paths["meta"].write_text(json.dumps({
        "feature_value_runner": True, "seed": seed,
        "initialization": "full_dataprep -> seeded IsolationForest -> seeded CatBoost; compute_shap=False",
        "rows": len(frame), "features": feature_matrix(frame).shape[1],
    }, indent=2), encoding="utf-8")
    return frame, model


def _quantile_bins(values, bins=20):
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) <= 2:
        return np.zeros(len(values), dtype=int)
    return np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, len(edges) - 2)


def build_value_view(X: pd.DataFrame):
    """Decode existing model columns into stable numeric and categorical values."""
    columns = list(X.columns)
    n = len(X)
    values: dict[str, np.ndarray] = {}
    kinds: dict[str, str] = {}
    scales: dict[str, float] = {}

    numeric_names = [name for name in ("amount_1", "amount_2", "amount_3", "date_difference")
                     if name in X.columns]
    for name in numeric_names:
        col = X[name].to_numpy(dtype=float)
        q25, q75 = np.quantile(col, [0.25, 0.75])
        scale = float(q75 - q25)
        if scale <= 1e-12:
            scale = float(np.median(np.abs(col - np.median(col))) * 1.4826)
        if scale <= 1e-12:
            continue
        values[name] = col
        kinds[name] = "numeric"
        scales[name] = scale

    #group one-hot columns by their original categorical field
    categorical_prefixes = (
        "gl_account_name_1", "gl_account_name_2", "gl_account_name_3",
        "cd_flag_1", "cd_flag_2", "cd_flag_3", "tax_rate", "user",
        "weekend_1", "weekend_2", "weekend_3", "nwh_1", "nwh_2", "nwh_3",
        "top_n_1", "top_n_2", "top_n_3", "high_cash_1", "high_cash_2", "high_cash_3",
    )
    consumed = set(numeric_names)
    for prefix in categorical_prefixes:
        dummy_cols = [name for name in columns if name.startswith(prefix + "_")]
        if not dummy_cols:
            continue
        matrix = X[dummy_cols].to_numpy(dtype=float)
        #constant fields do not help match cases
        if matrix.shape[1] == 1 and np.unique(matrix[:, 0]).size == 1:
            consumed.update(dummy_cols)
            continue
        category_names = [name[len(prefix) + 1:] for name in dummy_cols]
        winners = matrix.argmax(axis=1)
        active = matrix.max(axis=1) > 0.5
        categories = np.asarray(category_names, dtype=object)[winners].astype(str)
        categories[~active] = "<missing>"
        _levels, codes = np.unique(categories, return_inverse=True)
        values[prefix] = codes
        kinds[prefix] = "categorical"
        consumed.update(dummy_cols)

    #keep encoded flags categorical instead of treating them as ordinal
    for name in columns:
        if name in consumed or name in values:
            continue
        if name.startswith(("promptly_",)):
            col = X[name].to_numpy()
            _levels, codes = np.unique(col.astype(str), return_inverse=True)
            values[name] = codes.astype(np.int32)
            kinds[name] = "categorical"
            consumed.add(name)

    #group related features using existing model inputs only
    group_map = {
        "all": list(values),
        "numeric": [name for name, kind in kinds.items() if kind == "numeric"],
        "financial": [name for name in numeric_names if name in values],
        "accounting": [name for name in values if name.startswith(("amount_", "gl_account_name_", "cd_flag_", "tax_rate"))],
        "categories": [name for name, kind in kinds.items() if kind == "categorical"],
        "business_context": [name for name in values if name.startswith(("gl_account_name_", "cd_flag_", "tax_rate", "user"))],
        "behavior_flags": [name for name in values if name.startswith(("promptly_", "weekend_", "nwh_", "top_n_", "high_cash_"))],
    }
    quantile_bins = {
        bins: {name: _quantile_bins(values[name], bins=bins)
               for name, kind in kinds.items() if kind == "numeric"}
        for bins in (5, 10, 20, 40)
    }
    return {
        "values": values, "kinds": kinds, "scales": scales,
        "groups": {key: [name for name in names if name in values] for key, names in group_map.items()},
        "quantile_bins": quantile_bins,
        "n_rows": n,
    }


def value_distances(view, seed_i, rows, feature_names):
    if not feature_names:
        return np.zeros(len(rows), dtype=float)
    distances = np.zeros(len(rows), dtype=float)
    for name in feature_names:
        col = view["values"][name]
        if view["kinds"][name] == "numeric":
            component = np.minimum(np.abs(col[rows].astype(float) - float(col[seed_i])) /
                                   view["scales"][name], 1.0)
        else:
            component = (col[rows] != col[seed_i]).astype(float)
        distances += component
    return distances / len(feature_names)


def family_mask(policy, view, seed_i, kmeans_labels=None,
                reviewed_tree_leaf_ids=None, reviewed_tree_pure_leaves=None,
                seed_label=None, reviewed_kmeans_labels=None):
    n = view["n_rows"]
    kind = policy["kind"]
    if kind == "gower_nearest":
        return np.ones(n, dtype=bool)
    if kind == "direct_only":
        return np.zeros(n, dtype=bool)
    if kind == "value_box":
        tau = float(policy["tolerance"])
        names = policy["features"]
        mask = np.ones(n, dtype=bool)
        for name in names:
            col = view["values"][name]
            if view["kinds"][name] == "numeric":
                mask &= np.abs(col.astype(float) - float(col[seed_i])) <= tau * view["scales"][name]
            else:
                mask &= col == col[seed_i]
        return mask
    if kind == "value_signature":
        mask = np.ones(n, dtype=bool)
        for name in policy["categorical_features"]:
            col = view["values"][name]
            mask &= col == col[seed_i]
        for name in policy["numeric_features"]:
            bin_values = view["quantile_bins"][int(policy.get("bin_count", 20))][name]
            mask &= bin_values == bin_values[seed_i]
        return mask
    if kind == "kmeans_family":
        family_id = int(kmeans_labels[seed_i])
        if (policy.get("reviewed_family_purity") and
                reviewed_kmeans_labels.get(family_id) != {int(seed_label)}):
            return np.zeros(n, dtype=bool)
        return np.asarray(kmeans_labels) == family_id
    if kind == "kmeans_value_box":
        family_id = int(kmeans_labels[seed_i])
        if (policy.get("reviewed_family_purity") and
                reviewed_kmeans_labels.get(family_id) != {int(seed_label)}):
            return np.zeros(n, dtype=bool)
        mask = np.asarray(kmeans_labels) == family_id
        tau = float(policy["tolerance"])
        for name in policy["features"]:
            col = view["values"][name]
            if view["kinds"][name] == "numeric":
                mask &= np.abs(col.astype(float) - float(col[seed_i])) <= tau * view["scales"][name]
            else:
                mask &= col == col[seed_i]
        return mask
    if kind == "reviewed_value_tree":
        leaf = int(reviewed_tree_leaf_ids[seed_i])
        if reviewed_tree_pure_leaves.get(leaf) != int(seed_label):
            return np.zeros(n, dtype=bool)
        return np.asarray(reviewed_tree_leaf_ids) == leaf
    raise ValueError(f"Unsupported value policy: {kind}")


def fit_reviewed_value_tree(policy, X, ids, direct_labels, seed):
    """Fit transparent value-family rules on directly reviewed examples only."""
    id_to_row = {pid: row for row, pid in enumerate(ids)}
    reviewed_ids = list(direct_labels)
    train_rows = np.asarray([id_to_row[pid] for pid in reviewed_ids], dtype=int)
    labels = np.asarray([direct_labels[pid] for pid in reviewed_ids], dtype=int)
    tree = DecisionTreeClassifier(
        max_depth=int(policy["max_depth"]),
        min_samples_leaf=int(policy["min_samples_leaf"]),
        class_weight="balanced" if policy.get("balanced_classes", False) else None,
        random_state=seed,
    )
    tree.fit(X.iloc[train_rows], labels)
    leaves = tree.apply(X)
    observed: dict[int, set[int]] = {}
    for row, label in zip(train_rows, labels):
        observed.setdefault(int(leaves[row]), set()).add(int(label))
    pure = {leaf: next(iter(classes)) for leaf, classes in observed.items() if len(classes) == 1}
    return tree, leaves, pure


def tree_leaf_rule(tree, X, row_i):
    node = 0
    row = X.iloc[row_i].to_numpy(dtype=float)
    clauses = []
    while tree.tree_.children_left[node] != tree.tree_.children_right[node]:
        feature_i = int(tree.tree_.feature[node])
        threshold = float(tree.tree_.threshold[node])
        name = str(X.columns[feature_i])
        if row[feature_i] <= threshold:
            clauses.append(f"{name} <= {threshold:.6g}")
            node = int(tree.tree_.children_left[node])
        else:
            clauses.append(f"{name} > {threshold:.6g}")
            node = int(tree.tree_.children_right[node])
    return clauses


def metric_full(model, X, frame):
    pred = np.asarray(model.predict(Pool(X))).reshape(-1).astype(int)
    y = frame["label"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "full_n": int(len(y)),
        "full_precision": float(precision_score(y, pred, zero_division=0)),
        "full_recall": float(recall_score(y, pred, zero_division=0)),
        "full_tp": int(tp), "full_fp": int(fp), "full_fn": int(fn), "full_tn": int(tn),
    }


def fit_value_model(frame, trusted_ids, trusted_targets, direct_labels,
                    propagated_targets, seed):
    """Same split-training fit recipe, local to experiment; no baseline edits."""
    targets = dict(trusted_targets)
    targets.update(propagated_targets)
    targets.update(direct_labels)
    fit_ids = set(targets)
    fit = frame.loc[frame["posting_id"].isin(fit_ids)].copy()
    y_fit = fit["posting_id"].map(targets).to_numpy(dtype=int)
    n_positive = int(y_fit.sum())
    if n_positive == 0 or n_positive == len(y_fit):
        raise ValueError("Fit targets must contain both classes")
    weights = np.ones(len(fit), dtype=float)
    weights[fit["posting_id"].isin(direct_labels).to_numpy()] = CORRECTED_WEIGHT
    model = CatBoostClassifier(
        iterations=500, depth=8, learning_rate=0.05, loss_function="Logloss",
        random_seed=seed, verbose=False, class_weights={0: 1.0, 1: max(5.0, len(fit) / n_positive)},
        allow_writing_files=False,
    )
    model.fit(Pool(feature_matrix(fit), label=y_fit, weight=weights))
    return model


def run_trajectory(seed, schedule, arm_name, policy, frame, X, initial_model,
                   trusted_ids, trusted_targets, tie_break, value_view,
                   *, propagation_cap=PROPAGATION_CAP, kmeans_labels=None):
    rounds, batch_size = schedule
    ids = frame["posting_id"].astype(str).to_numpy()
    true_by_id = dict(zip(ids, frame["label"].to_numpy(dtype=int)))
    id_to_row = {pid: row for row, pid in enumerate(ids)}
    direct_labels: dict[str, int] = {}
    reviewed_ids: set[str] = set()
    propagated: dict[str, int] = {}
    model = initial_model
    query_score = frame["pred_score"].to_numpy(dtype=float)
    query_pred = frame["pred_label"].to_numpy(dtype=int)
    rows, seed_rows, direct_rows, events = [], [], [], []
    started = time.perf_counter()

    for round_i in range(1, rounds + 1):
        uncertainty = 1.0 - 2.0 * np.abs(query_score - 0.5)
        batch_ids = select_batch(frame, trusted_ids, reviewed_ids, uncertainty, tie_break, batch_size)
        for pid in batch_ids:
            propagated.pop(pid, None)  #direct review overrides earlier targets
            direct_labels[pid] = true_by_id[pid]
            reviewed_ids.add(pid)
            direct_rows.append({
                "seed": seed, "schedule": f"{rounds}x{batch_size}", "arm": arm_name,
                "round": round_i, "review_budget": (round_i - 1) * batch_size + len(direct_rows) % batch_size + 1,
                "posting_id": pid, "reviewed_label": direct_labels[pid],
            })

        fit_started = time.perf_counter()
        model = fit_value_model(frame, trusted_ids, trusted_targets, direct_labels, propagated, seed)
        query_score = model.predict_proba(Pool(X))[:, 1]
        query_pred = np.asarray(model.predict(Pool(X))).reshape(-1).astype(int)
        metrics = metric_full(model, X, frame)
        fit_ids = trusted_ids | set(direct_labels) | set(propagated)
        fit_mask = np.zeros(len(ids), dtype=bool)
        fit_rows = np.fromiter((id_to_row[pid] for pid in fit_ids), dtype=int, count=len(fit_ids))
        fit_mask[fit_rows] = True
        uncertainty = 1.0 - 2.0 * np.abs(query_score - 0.5)
        new_ids = set()
        candidate_sum = 0
        if policy["kind"] == "reviewed_value_tree":
            tree_model, reviewed_tree_leaf_ids, reviewed_tree_pure_leaves = fit_reviewed_value_tree(
                policy, X, ids, direct_labels, seed
            )
        else:
            tree_model, reviewed_tree_leaf_ids, reviewed_tree_pure_leaves = None, None, None
        reviewed_kmeans_labels: dict[int, set[int]] = {}
        if policy.get("reviewed_family_purity") and kmeans_labels is not None:
            for reviewed_id, reviewed_label in direct_labels.items():
                family_id = int(kmeans_labels[id_to_row[reviewed_id]])
                reviewed_kmeans_labels.setdefault(family_id, set()).add(int(reviewed_label))
        for rank, seed_id in enumerate(batch_ids, start=1):
            seed_i = id_to_row[seed_id]
            feature_names = policy.get("features", [])
            mask = family_mask(
                policy, value_view, seed_i, kmeans_labels=kmeans_labels,
                reviewed_tree_leaf_ids=reviewed_tree_leaf_ids,
                reviewed_tree_pure_leaves=reviewed_tree_pure_leaves,
                seed_label=direct_labels[seed_id],
                reviewed_kmeans_labels=reviewed_kmeans_labels,
            )
            if (policy.get("seed_label_filter") is not None and
                    int(direct_labels[seed_id]) != int(policy["seed_label_filter"])):
                mask[:] = False
            matched = np.flatnonzero(mask)
            candidate_sum += int(mask.sum())
            eligible = matched[~fit_mask[matched]]
            eligible = eligible[eligible != seed_i]
            if len(eligible):
                distance = value_distances(value_view, seed_i, eligible, feature_names)
                order = np.lexsort((eligible, distance))
                cap_by_label = policy.get("recipient_cap_by_label", {})
                cap = cap_by_label.get(
                    str(int(direct_labels[seed_id])),
                    policy.get("propagation_cap", propagation_cap),
                )
                selected = eligible[order[:cap]] if cap is not None else eligible[order]
                selected_distances = distance[order[:len(selected)]]
            else:
                selected = np.asarray([], dtype=int)
                selected_distances = np.asarray([], dtype=float)
            assigned = []
            for row_i, dist in zip(selected, selected_distances):
                pid = ids[row_i]
                if pid in propagated or pid in direct_labels or pid in trusted_ids:
                    continue
                propagated[pid] = int(direct_labels[seed_id])
                fit_mask[row_i] = True
                new_ids.add(pid)
                assigned.append(pid)
                events.append({
                    "seed": seed, "schedule": f"{rounds}x{batch_size}", "arm": arm_name,
                    "round": round_i, "review_budget": (round_i - 1) * batch_size + rank,
                    "seed_posting_id": seed_id, "posting_id": pid,
                    "propagated_label": int(direct_labels[seed_id]),
                    "true_label_offline": int(true_by_id[pid]),
                    "value_distance": float(dist),
                })
            seed_rows.append({
                "seed": seed, "schedule": f"{rounds}x{batch_size}", "arm": arm_name,
                "round": round_i, "review_budget": (round_i - 1) * batch_size + rank,
                "seed_posting_id": seed_id, "seed_label": int(direct_labels[seed_id]),
                "policy": policy["name"], "matched_population": int(mask.sum()),
                "fit_excluded_matches": int(np.sum(fit_mask[matched])),
                "eligible_matches": int(len(eligible)), "new_propagated": len(assigned),
                "features": json.dumps(feature_names),
                "family_rule": json.dumps({key: value for key, value in policy.items() if key not in {"features"}}, default=str),
                "reviewed_value_tree_path": json.dumps(
                    tree_leaf_rule(tree_model, X, seed_i) if tree_model is not None else []
                ),
            })

        row = {
            "seed": seed, "schedule": f"{rounds}x{batch_size}", "arm": arm_name,
            "round": round_i, "review_budget": len(direct_labels),
            "new_propagated": len(new_ids), "active_propagated": len(propagated),
            "fit_rows": len(fit_ids | set(propagated)),
            "fit_seconds": round(time.perf_counter() - fit_started, 2),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            **metrics,
        }
        rows.append(row)
        print(json.dumps({
            "seed": seed, "schedule": row["schedule"], "arm": arm_name,
            "checkpoint": f"{round_i}/{rounds}", "reviews": len(direct_labels),
            "P_full": metrics["full_precision"], "R_full": metrics["full_recall"],
            "new_prop": len(new_ids), "active_prop": len(propagated),
        }, sort_keys=True), flush=True)

    #fit labels propagated by the final batch
    final_started = time.perf_counter()
    model = fit_value_model(frame, trusted_ids, trusted_targets, direct_labels, propagated, seed)
    final_metrics = metric_full(model, X, frame)
    rows.append({
        "seed": seed, "schedule": f"{rounds}x{batch_size}", "arm": arm_name,
        "round": rounds, "review_budget": len(direct_labels), "final_flush": True,
        "new_propagated": 0, "active_propagated": len(propagated),
        "fit_rows": len(trusted_ids | set(direct_labels) | set(propagated)),
        "fit_seconds": round(time.perf_counter() - final_started, 2),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        **final_metrics,
    })
    return {
        "checkpoints": rows, "seed_policies": seed_rows, "direct_reviews": direct_rows,
        "propagation_events": events, "active_propagated": propagated,
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }


def policies_for_view(view):
    groups = view["groups"]
    specs = [
        {"name": "gower_all_top4", "kind": "gower_nearest", "features": groups["all"]},
        {"name": "numeric_box_tau010", "kind": "value_box", "features": groups["numeric"], "tolerance": 0.10},
        {"name": "numeric_box_tau025", "kind": "value_box", "features": groups["numeric"], "tolerance": 0.25},
        {"name": "numeric_box_tau050", "kind": "value_box", "features": groups["numeric"], "tolerance": 0.50},
        {"name": "all_value_box_tau025", "kind": "value_box", "features": groups["all"], "tolerance": 0.25},
        {"name": "all_value_box_tau010", "kind": "value_box", "features": groups["all"], "tolerance": 0.10},
        {"name": "all_value_box_tau050", "kind": "value_box", "features": groups["all"], "tolerance": 0.50},
        {"name": "core_value_box_tau025", "kind": "value_box",
         "features": [name for name in groups["all"] if not name.startswith(("promptly_", "weekend_", "nwh_", "top_n_", "high_cash_"))],
         "tolerance": 0.25},
        {"name": "all_value_box_tau025_uncapped", "kind": "value_box", "features": groups["all"],
         "tolerance": 0.25, "propagation_cap": None},
        {"name": "accounting_value_box_tau025", "kind": "value_box", "features": groups["accounting"], "tolerance": 0.25},
        {"name": "behavior_flags_value_box_tau025", "kind": "value_box", "features": groups["behavior_flags"], "tolerance": 0.25},
        {"name": "accounting_signature_20bins", "kind": "value_signature",
         "categorical_features": [name for name in groups["accounting"] if view["kinds"][name] == "categorical"],
         "numeric_features": [name for name in groups["financial"] if view["kinds"].get(name) == "numeric"],
         "bin_count": 20,
         "features": [name for name in groups["accounting"] if view["kinds"][name] == "categorical"] +
                     [name for name in groups["financial"] if view["kinds"].get(name) == "numeric"]},
        {"name": "business_context_signature_20bins", "kind": "value_signature",
         "categorical_features": groups["business_context"],
         "numeric_features": [name for name in groups["financial"] if view["kinds"].get(name) == "numeric"],
         "bin_count": 20,
         "features": groups["business_context"] +
                      [name for name in groups["financial"] if view["kinds"].get(name) == "numeric"]},
        {"name": "reviewed_value_tree_d3_leaf2", "kind": "reviewed_value_tree", "features": groups["all"], "max_depth": 3, "min_samples_leaf": 2},
        {"name": "reviewed_value_tree_d5_leaf2", "kind": "reviewed_value_tree", "features": groups["all"], "max_depth": 5, "min_samples_leaf": 2},
        {"name": "reviewed_value_tree_d8_leaf2", "kind": "reviewed_value_tree", "features": groups["all"], "max_depth": 8, "min_samples_leaf": 2},
        {"name": "reviewed_value_tree_d5_leaf5", "kind": "reviewed_value_tree", "features": groups["all"], "max_depth": 5, "min_samples_leaf": 5},
        {"name": "reviewed_value_tree_d5_balanced", "kind": "reviewed_value_tree", "features": groups["all"], "max_depth": 5, "min_samples_leaf": 2, "balanced_classes": True},
    ]
    for bins in (5, 10, 40):
        specs.append({
            "name": f"business_context_signature_{bins}bins", "kind": "value_signature",
            "categorical_features": groups["business_context"],
            "numeric_features": [name for name in groups["financial"] if view["kinds"].get(name) == "numeric"],
            "bin_count": bins,
            "features": groups["business_context"] +
                        [name for name in groups["financial"] if view["kinds"].get(name) == "numeric"],
        })
    specs.append({
        "name": "business_context_signature_5bins_uncapped", "kind": "value_signature",
        "categorical_features": groups["business_context"],
        "numeric_features": [name for name in groups["financial"] if view["kinds"].get(name) == "numeric"],
        "bin_count": 5, "propagation_cap": None,
        "features": groups["business_context"] +
                    [name for name in groups["financial"] if view["kinds"].get(name) == "numeric"],
    })
    specs.append({
        "name": "all_value_box_tau050_uncapped", "kind": "value_box",
        "features": groups["all"], "tolerance": 0.50, "propagation_cap": None,
    })
    signature_trials = (
        ("gl1_signature", ["gl_account_name_1"], [], 20),
        ("gl1_user_signature", ["gl_account_name_1", "user"], [], 20),
        ("gl1_gl2_cd1_signature", ["gl_account_name_1", "gl_account_name_2", "cd_flag_1"], [], 20),
        ("gl1_cd1_amount1_bin5", ["gl_account_name_1", "cd_flag_1"], ["amount_1"], 5),
        ("gl1_gl2_cd1_amount1_bin10", ["gl_account_name_1", "gl_account_name_2", "cd_flag_1"], ["amount_1"], 10),
        ("gl1_user_cd1_amount1_bin5", ["gl_account_name_1", "user", "cd_flag_1"], ["amount_1"], 5),
        ("gl1_user_cd1_amounts_bin5", ["gl_account_name_1", "user", "cd_flag_1"],
         ["amount_1", "amount_2", "amount_3"], 5),
        ("gl1_gl2_user_cd1_bins10", ["gl_account_name_1", "gl_account_name_2", "user", "cd_flag_1"],
         ["amount_1", "date_difference"], 10),
    )
    for name, cats, nums, bins in signature_trials:
        cats = [feature for feature in cats if feature in view["values"]]
        nums = [feature for feature in nums if view["kinds"].get(feature) == "numeric"]
        specs.append({
            "name": name, "kind": "value_signature", "categorical_features": cats,
            "numeric_features": nums, "bin_count": bins, "features": cats + nums,
        })
    return specs


def make_kmeans_policy(X, value_view, seed, clusters=1000):
    #build families from model-input values only
    model = MiniBatchKMeans(
        n_clusters=clusters, random_state=seed, batch_size=4096, n_init=3,
        max_iter=100, reassignment_ratio=0.01,
    )
    labels = model.fit_predict(X.to_numpy(dtype=float))
    return {"name": f"kmeans_value_families_{clusters}", "kind": "kmeans_family",
            "features": list(value_view["values"]), "cluster_count": clusters}, labels


def setup_trusted(frame, seed):
    ids = frame["posting_id"].astype(str)
    uncertainty = 1.0 - 2.0 * np.abs(frame["pred_score"].to_numpy(dtype=float) - 0.5)
    tie_break = np.random.RandomState(seed).random_sample(len(frame))
    order = np.lexsort((tie_break, uncertainty))
    trusted_ids = set(ids.iloc[order[:round(TRUSTED_FRACTION * len(frame))]])
    targets = dict(zip(ids, frame["pred_label"].to_numpy(dtype=int)))
    trusted_targets = {pid: targets[pid] for pid in trusted_ids}
    return trusted_ids, trusted_targets, tie_break


def save_outputs(all_results, run_seed, protocols, stem):
    checkpoints = []
    policies = []
    reviews = []
    events = []
    summaries = []
    for key, result in all_results.items():
        seed, schedule_name, arm_name = key.split("|")
        checkpoints.extend(result["checkpoints"])
        policies.extend(result["seed_policies"])
        reviews.extend(result["direct_reviews"])
        events.extend(result["propagation_events"])
        final = result["checkpoints"][-1]
        wrong = sum(event["propagated_label"] != event["true_label_offline"]
                    for event in result["propagation_events"])
        summaries.append({
            "seed": int(seed), "schedule": schedule_name, "arm": arm_name,
            "runtime_seconds": result["runtime_seconds"],
            "reviews": max(row["review_budget"] for row in result["checkpoints"]),
            "full_precision": final["full_precision"], "full_recall": final["full_recall"],
            "full_fp": final["full_fp"], "full_fn": final["full_fn"],
            "active_propagated": final["active_propagated"],
            "propagation_events": len(result["propagation_events"]),
            "wrong_propagations": wrong,
        })
    pd.DataFrame(checkpoints).to_csv(f"{stem}_checkpoints.csv", index=False)
    pd.DataFrame(policies).to_csv(f"{stem}_seed_policies.csv", index=False)
    pd.DataFrame(reviews).to_csv(f"{stem}_direct_reviews.csv", index=False)
    pd.DataFrame(events).to_csv(f"{stem}_propagated.csv", index=False)
    pd.DataFrame(summaries).to_csv(f"{stem}_summary.csv", index=False)
    payload = {
        "protocol": protocols,
        "run_seeds": list(run_seed),
        "completed_trajectories": list(all_results),
        "summary": summaries,
    }
    Path(f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pd.ExcelWriter(f"{stem}.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(checkpoints).to_excel(writer, sheet_name="Full-data P-R", index=False)
        pd.DataFrame(summaries).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(policies).to_excel(writer, sheet_name="Seed policies", index=False)
        pd.DataFrame(reviews).to_excel(writer, sheet_name="Direct reviews", index=False)
        pd.DataFrame(events).to_excel(writer, sheet_name="Recipients", index=False)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42", help="comma-separated run seeds")
    parser.add_argument("--schedules", default="10x50", help="comma-separated schedules")
    parser.add_argument("--policies", default="screen", help="screen, all, or comma-separated policy names")
    parser.add_argument("--tag", default="screen", help="unique artifact tag")
    parser.add_argument("--recipient-cap", type=int, default=PROPAGATION_CAP,
                        help="maximum new propagations per seed; 0 means uncapped")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    output_stem = RESULTS / f"{STEM.name}_{args.tag}"
    seeds = tuple(int(value) for value in args.seeds.split(","))
    requested_schedules = tuple(tuple(int(part) for part in value.lower().split("x"))
                                 for value in args.schedules.split(","))
    all_results = {}
    protocols = {
        "direct_only_comparator": True,
        "direct_reviews": REVIEW_BUDGET,
        "schedules": [{"rounds": r, "batch_size": b} for r, b in SCHEDULES],
        "timing": "post_review_shap_next_round (same fit→match→next-fit order; value matching itself uses no SHAP)",
        "weights": {"direct": CORRECTED_WEIGHT, "propagated": 1},
        "recipient_cap": args.recipient_cap,
        "metrics": "full-dataset P/R and FP/FN each checkpoint; non-fit P/R omitted",
        "feature_inputs": "existing model-ready transaction values; no labels/IDs/predictions/SHAP in family matching",
    }

    #use seed 42 for screening
    #add confirmation seeds only after a candidate passes screening
    for seed in seeds:
        frame, initial_model = _initial_cache(seed)
        frame["posting_id"] = frame["posting_id"].astype(str)
        frame["label"] = frame["label"].astype(int)
        frame["pred_label"] = frame["pred_label"].astype(int)
        X = feature_matrix(frame)
        if len(frame) != 51125 or not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError(f"Unexpected/non-finite initial state for run seed {seed}")
        view = build_value_view(X)
        trusted_ids, trusted_targets, tie_break = setup_trusted(frame, seed)
        policies = policies_for_view(view)
        if args.policies in {"screen", "all"}:
            selected_policies = policies
        elif args.policies == "direct_only":
            selected_policies = []
        else:
            wanted = set(args.policies.split(","))
            selected_policies = [policy for policy in policies if policy["name"] in wanted]
            custom_tokens = {token for token in wanted if token.startswith("kmeans")}
            unknown = wanted - {policy["name"] for policy in selected_policies} - custom_tokens
            if unknown:
                raise ValueError(f"Unknown value policies: {sorted(unknown)}")
        need_kmeans = args.policies in {"screen", "all"} or any(
            token.startswith("kmeans") for token in args.policies.split(",")
        )
        kmeans_by_clusters = {}
        if need_kmeans:
            clusters_needed = {1000} if args.policies in {"screen", "all"} else {
                int(token[len("kmeans"):].split("_")[0]) for token in args.policies.split(",")
                if token.startswith("kmeans")
            }
            for count in sorted(clusters_needed):
                base_policy, labels = make_kmeans_policy(X, view, seed, clusters=count)
                kmeans_by_clusters[count] = (base_policy, labels)
                if args.policies in {"screen", "all"} and count == 1000:
                    selected_policies.append(base_policy)
            if args.policies not in {"screen", "all"}:
                for token in sorted(custom_tokens):
                    cluster_count = int(token[len("kmeans"):].split("_")[0])
                    suffix = token[len(f"kmeans{cluster_count}_"):]
                    uncapped = suffix.endswith("_uncapped")
                    if uncapped:
                        suffix = suffix[:-len("_uncapped")]
                    label_filter = None
                    if suffix.endswith("_label0") or suffix.endswith("_label1"):
                        label_filter = int(suffix[-1])
                        suffix = suffix[:-7]
                    purity_gate = suffix.endswith("_pure_reviewed")
                    base_suffix = suffix.removesuffix("_pure_reviewed")
                    if base_suffix == "family":
                        policy = {"name": token, "kind": "kmeans_family",
                                  "features": list(view["values"]), "cluster_count": cluster_count}
                    elif base_suffix == "uncapped":
                        policy = {"name": token, "kind": "kmeans_family",
                                  "features": list(view["values"]), "cluster_count": cluster_count,
                                  "propagation_cap": None}
                    elif base_suffix.startswith("allbox") or base_suffix.startswith("corebox"):
                        scope, delta_spec = base_suffix.split("box", 1)
                        cap_by_label = None
                        if "_n" in delta_spec:
                            delta_spec, cap_spec = delta_spec.split("_n", 1)
                            normal_cap, positive_cap = (int(part) for part in cap_spec.split("p", 1))
                            cap_by_label = {"0": normal_cap, "1": positive_cap}
                        feature_set = (view["groups"]["all"] if scope == "all"
                                       else [name for name in view["groups"]["all"] if not name.startswith(("promptly_", "weekend_", "nwh_", "top_n_", "high_cash_"))])
                        policy = {"name": token, "kind": "kmeans_value_box",
                                  "features": feature_set, "cluster_count": cluster_count,
                                  "tolerance": int(delta_spec) / 100.0}
                        if cap_by_label is not None:
                            policy["recipient_cap_by_label"] = cap_by_label
                    else:
                        raise ValueError(f"Unknown k-means family policy: {token}")
                    if purity_gate:
                        policy["reviewed_family_purity"] = True
                    if label_filter is not None:
                        policy["seed_label_filter"] = label_filter
                    if uncapped:
                        policy["propagation_cap"] = None
                    selected_policies.append(policy)

        schedules = requested_schedules
        for rounds, batch_size in schedules:
            schedule_name = f"{rounds}x{batch_size}"
            arms = [{"name": "direct_only", "policy": None}]
            arms.extend({"name": policy["name"], "policy": policy} for policy in selected_policies)
            for arm in arms:
                key = f"{seed}|{schedule_name}|{arm['name']}"
                print(f"\n[trajectory] seed={seed} {schedule_name} {arm['name']}", flush=True)
                policy = arm["policy"] or {"name": "direct_only", "kind": "direct_only", "features": []}
                result = run_trajectory(
                    seed, (rounds, batch_size), arm["name"], policy,
                    frame, X, initial_model, trusted_ids, trusted_targets, tie_break,
                    view, propagation_cap=(None if args.recipient_cap == 0 else args.recipient_cap),
                    kmeans_labels=(kmeans_by_clusters[int(policy["cluster_count"])][1]
                                   if policy["kind"].startswith("kmeans") else None),
                )
                all_results[key] = result
                save_outputs(all_results, seeds, protocols, output_stem)
        print(f"[seed complete] {seed}", flush=True)

    print(f"All trajectories complete: {len(all_results)}", flush=True)


if __name__ == "__main__":
    main()

"""Test whether Oracle-discovered large SHAP family helps split-trained model."""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.kittyboost import CORRECTED_WEIGHT, feature_matrix, fit_split_model

SEED = 42
TRUSTED_FRACTION = 0.50
REVIEW_ROUNDS = 10
REVIEW_BATCH_SIZE = 50
ROOT_CACHE = Path.home() / ".cache" / "ad2_split_training"
ARTIFACTS = ROOT / "legacy" / "probe_out"
RESULTS = ROOT / "legacy" / "experiments" / "results"


def summarize(name: str, model: CatBoostClassifier, frame: pd.DataFrame, mask=None) -> dict:
    if mask is None:
        mask = np.ones(len(frame), dtype=bool)
    subset = frame.loc[mask]
    pool = Pool(feature_matrix(subset))
    score = model.predict_proba(pool)[:, 1]
    pred = np.asarray(model.predict(pool)).reshape(-1).astype(int)
    y = subset["label"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "arm": name,
        "n_evaluated": int(len(y)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, score)) if np.unique(y).size == 2 else None,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def load_initial() -> pd.DataFrame:
    stem = f"initial_seed_{SEED}"
    with (ROOT_CACHE / f"{stem}_predictions.pkl").open("rb") as handle:
        frame = pickle.load(handle)
    if len(frame) != 51125 or not {"posting_id", "label", "pred_label", "pred_score"}.issubset(frame.columns):
        raise ValueError("Initial model cache does not match expected 51,125-row split-training run.")
    if not (ROOT_CACHE / f"{stem}.cbm").exists():
        raise FileNotFoundError(f"Missing initial model cache: {ROOT_CACHE / f'{stem}.cbm'}")
    return frame.reset_index(drop=True)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = load_initial()
    ids = frame["posting_id"].astype(str)
    frame["posting_id"] = ids
    frame["label"] = frame["label"].astype(int)
    frame["pred_label"] = frame["pred_label"].astype(int)

    #rebuild the notebook's fixed bottom-half pseudo-label pool
    uncertainty = 1.0 - 2.0 * np.abs(frame["pred_score"].to_numpy(dtype=float) - 0.5)
    tie_break = np.random.RandomState(SEED).random_sample(len(frame))
    trusted_order = np.lexsort((tie_break, uncertainty))
    trusted_ids = set(ids.iloc[trusted_order[: round(TRUSTED_FRACTION * len(frame))]])
    trusted_targets = dict(zip(ids, frame["pred_label"].to_numpy(dtype=int)))
    trusted_targets = {pid: trusted_targets[pid] for pid in trusted_ids}
    trusted_mask = ids.isin(trusted_ids).to_numpy()

    #replay the 10-by-50 review run used to discover this family
    discovery_reviews: dict[str, int] = {}
    discovery_ids: set[str] = set()
    discovery_model = None
    for round_i in range(REVIEW_ROUNDS):
        if round_i == 0:
            query_u = uncertainty
        else:
            score = discovery_model.predict_proba(Pool(feature_matrix(frame)))[:, 1]
            query_u = 1.0 - 2.0 * np.abs(score - 0.5)
        eligible = ~(trusted_mask | ids.isin(discovery_ids).to_numpy())
        rows = np.flatnonzero(eligible)
        order = rows[np.lexsort((tie_break[rows], -query_u[rows]))]
        batch_ids = ids.iloc[order[:REVIEW_BATCH_SIZE]].tolist()
        discovery_ids.update(batch_ids)
        discovery_reviews.update({
            pid: int(frame.loc[ids == pid, "label"].iloc[0]) for pid in batch_ids
        })
        discovery_model = fit_split_model(frame, trusted_ids, trusted_targets, discovery_reviews)
        print(f"Family-discovery replay {round_i + 1}/{REVIEW_ROUNDS}: reviewed={len(discovery_ids)}")

    #use stored labels to correct remaining errors
    oracle_reviews = dict(discovery_reviews)
    cleanup_model = discovery_model
    for cleanup_round in range(5):
        pred = np.asarray(cleanup_model.predict(Pool(feature_matrix(frame)))).reshape(-1).astype(int)
        errors = np.flatnonzero(pred != frame["label"].to_numpy(dtype=int))
        new_ids = [ids.iloc[i] for i in errors if ids.iloc[i] not in oracle_reviews]
        print(f"Oracle cleanup {cleanup_round + 1}: residual_errors={len(errors)}, new_reviews={len(new_ids)}")
        if not new_ids:
            break
        oracle_reviews.update({pid: int(frame.loc[ids == pid, "label"].iloc[0]) for pid in new_ids})
        cleanup_model = fit_split_model(frame, trusted_ids, trusted_targets, oracle_reviews)

    #find the largest normal box and recover members from its SHAP bounds
    boxes = pd.read_csv(ARTIFACTS / "ideal_shap_pure_boxes.csv")
    boxes = boxes.loc[(boxes["status"] == "maximum_found") & (boxes["label"] == 0)]
    box = boxes.sort_values(["support", "anchor_count"], ascending=False).iloc[0]
    shap_features = json.loads(box["features"])
    lower = np.asarray(json.loads(box["lower"]), dtype=float)
    upper = np.asarray(json.loads(box["upper"]), dtype=float)
    metadata = json.loads((ARTIFACTS / "ideal_shap_pure_boxes_meta.json").read_text(encoding="utf-8"))
    if metadata.get("run_seed") != SEED or metadata.get("shap_space") != "shap_raw":
        raise ValueError("Family artifact metadata does not match expected seed/SHAP space.")

    shap_started = time.perf_counter()
    pool = Pool(feature_matrix(frame))
    shap_all = cleanup_model.get_feature_importance(pool, type="ShapValues")[:, :-1]
    all_features = list(feature_matrix(frame).columns)
    flag_prefixes = ("promptly_", "weekend_", "nwh_", "top_n_", "high_cash_")
    raw_idx = [i for i, name in enumerate(all_features) if not name.startswith(flag_prefixes)]
    raw_features = [all_features[i] for i in raw_idx]
    raw_shap = shap_all[:, raw_idx]
    if not set(shap_features).issubset(raw_features):
        raise ValueError("Recorded family references features absent from recomputed SHAP space.")
    feature_idx = np.asarray([raw_features.index(name) for name in shap_features], dtype=int)
    family_mask = np.all(
        (raw_shap[:, feature_idx] >= lower) & (raw_shap[:, feature_idx] <= upper), axis=1
    )
    family_ids = set(ids[family_mask])
    family_truth = frame.loc[family_mask, "label"].to_numpy(dtype=int)
    expected_support = int(box["support"])
    if int(family_mask.sum()) != expected_support:
        raise ValueError(
            f"Recomputed family size {int(family_mask.sum())} != artifact support {expected_support}; "
            "SHAP model/data does not reproduce discovery artifact."
        )
    if np.any(family_truth != 0):
        raise ValueError("Reconstructed Oracle family contains non-normal labels.")
    print(
        f"Family {box['box_id']}: support={family_mask.sum()}, "
        f"features={len(shap_features)}, SHAP recompute={time.perf_counter() - shap_started:.1f}s"
    )

    #run three independent active-learning trajectories
    #each arm selects from its current model and starts with family knowledge
    arms = {
        "direct_only": {"family": False, "fixed_weight": False},
        "family_recomputed_weight": {"family": True, "fixed_weight": False},
        "family_fixed_direct_weight": {"family": True, "fixed_weight": True},
    }
    trajectories = {}
    direct_weights_by_round = {}
    for arm, config in arms.items():
        review_labels: dict[str, int] = {}
        reviewed_ids: set[str] = set()
        model = None
        checkpoints = []
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

            direct_fit_ids = trusted_ids | set(review_labels)
            family_added = family_ids - direct_fit_ids if config["family"] else set()
            direct_positive_count = sum(trusted_targets.values()) + sum(review_labels.values())
            direct_class_weight = max(5.0, len(direct_fit_ids) / direct_positive_count)
            if arm == "direct_only":
                direct_weights_by_round[round_i] = direct_class_weight
            fixed_weight = (
                direct_weights_by_round[round_i]
                if config["fixed_weight"] else None
            )

            fit_started = time.perf_counter()
            model = fit_split_model(
                frame, trusted_ids, trusted_targets, review_labels,
                extra_ids=family_added,
                class_weight_anomaly=fixed_weight,
            )
            metrics = summarize(arm, model, frame)
            actual_fit_ids = direct_fit_ids | family_added
            target_positive_count = direct_positive_count
            actual_class_weight = (
                float(fixed_weight)
                if fixed_weight is not None
                else max(5.0, len(actual_fit_ids) / target_positive_count)
            )
            metrics.update({
                "round": round_i + 1,
                "review_budget": len(reviewed_ids),
                "reviewed_anomalies": int(sum(review_labels.values())),
                "fit_rows": len(actual_fit_ids),
                "family_rows_added": len(family_added),
                "class_weight_anomaly": actual_class_weight,
                "fit_seconds": round(time.perf_counter() - fit_started, 2),
            })
            checkpoints.append(metrics)
            print(json.dumps(metrics, sort_keys=True))
        trajectories[arm] = checkpoints

    result_path = RESULTS / "shap_family_propagation.json"
    payload = {
        "protocol": {
            "run_seed": SEED,
            "review_policy": "three independent 10 x 50 trajectories; each selects using its own latest model",
            "checkpoint_metrics": "full-data predictions after each 50-label batch; full data includes fit rows",
            "family_box_id": str(box["box_id"]),
            "family_label": 0,
            "family_support_artifact": expected_support,
            "family_features": shap_features,
            "family_bounds": {name: [float(lo), float(hi)] for name, lo, hi in zip(shap_features, lower, upper)},
            "family_reconstruction_purity": float(np.mean(family_truth == 0)),
            "trusted_rows": len(trusted_ids),
            "family_rows_overlapping_trusted": len(family_ids & trusted_ids),
            "family_label_source": "Oracle-selected ideal family; label 0",
            "oracle_discovery_reviews": len(discovery_reviews),
            "oracle_residual_corrections_for_family_discovery": len(oracle_reviews) - len(discovery_reviews),
            "family_available_from_round": 1,
            "class_weight_control": "family arm uses direct-only arm weight at matching review budget",
            "metrics_threshold": "CatBoost model.predict default threshold",
            "oracle_simulation": True,
        },
        "trajectories": trajectories,
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    checkpoint_rows = [row for checkpoints in trajectories.values() for row in checkpoints]
    pd.DataFrame(checkpoint_rows).to_csv(RESULTS / "shap_family_propagation_metrics.csv", index=False)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()

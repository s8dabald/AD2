"""Discover human-readable, one-feature pure tails in Oracle-clean SHAP_ALL."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool

from legacy.experiments.shap_all_groups import reconstruct_oracle_clean_model
from legacy.experiments.shap_family_propagation import feature_matrix


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "legacy" / "experiments" / "results"
MIN_SUPPORT = 2


def extreme_tail(values: np.ndarray, labels: np.ndarray, feature: str,
                 direction: str) -> dict | None:
    """Return longest pure extreme prefix, split only between distinct SHAP values."""
    ascending = direction == "low"
    order = np.argsort(values if ascending else -values, kind="mergesort")
    sorted_values = values[order]
    sorted_labels = labels[order]

    #merge ties because a threshold cannot split equal SHAP values
    starts = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
    unique_values = sorted_values[starts]
    counts = np.diff(np.r_[starts, len(sorted_values)])
    first_labels = np.unique(sorted_labels[starts[0]:starts[0] + counts[0]])
    if len(first_labels) != 1:
        return None
    tail_label = int(first_labels[0])
    boundary = None
    for i in range(1, len(unique_values)):
        tied_labels = np.unique(sorted_labels[starts[i]:starts[i] + counts[i]])
        if np.any(tied_labels != tail_label):
            boundary = i
            break
    if boundary is None:
        return None
    support = int(counts[:boundary].sum())
    if support < MIN_SUPPORT:
        return None

    last_in = float(unique_values[boundary - 1])
    first_out = float(unique_values[boundary])
    span = float(np.max(values) - np.min(values))
    if span <= 0:
        return None
    gap = abs(first_out - last_in)
    threshold = last_in + (first_out - last_in) / 2.0
    op = "<=" if ascending else ">"
    mask = values <= threshold if ascending else values > threshold
    if int(mask.sum()) != support or np.any(labels[mask] != tail_label):
        raise AssertionError("Extreme-tail boundary failed source-purity check")

    return {
        "feature": feature,
        "direction": direction,
        "group_label": tail_label,
        "support": support,
        "margin": gap / span,
        "raw_gap": gap,
        "threshold": float(threshold),
        "op": op,
        "mask": mask,
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frame, model, residual_corrections = reconstruct_oracle_clean_model()
    frame["posting_id"] = frame["posting_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    X = feature_matrix(frame)
    labels = frame["label"].to_numpy(dtype=int)
    shap_values = np.asarray(
        model.get_feature_importance(Pool(X), type="ShapValues")[:, :-1], dtype=float
    )
    if shap_values.shape != X.shape or not np.isfinite(shap_values).all():
        raise ValueError(f"Invalid Oracle-clean SHAP_ALL matrix: {shap_values.shape}")

    policies = []
    for col, feature in enumerate(X.columns):
        for direction in ("low", "high"):
            result = extreme_tail(shap_values[:, col], labels, feature, direction)
            if result is not None:
                policies.append(result)
    policies.sort(key=lambda p: (-p["margin"], -p["support"], p["feature"], p["direction"]))

    rows = []
    membership_candidates: dict[int, list[tuple[int, dict]]] = {}
    for index, policy in enumerate(policies):
        group_id = f"E{index + 1:04d}"
        policy["group_id"] = group_id
        path = [{
            "feature": policy["feature"],
            "op": policy["op"],
            "threshold": policy["threshold"],
        }]
        rows.append({
            "group_id": group_id,
            "feature": policy["feature"],
            "group_label": policy["group_label"],
            "support": policy["support"],
            "purity": 1.0,
            "pure": True,
            "path": json.dumps(path, ensure_ascii=False),
            "features": json.dumps([policy["feature"]], ensure_ascii=False),
            "n_rule_features": 1,
            "depth": 1,
            "direction": policy["direction"],
            "normalized_margin": policy["margin"],
            "raw_gap": policy["raw_gap"],
            "rank": index + 1,
        })
        for row_i in np.flatnonzero(policy["mask"]):
            membership_candidates.setdefault(int(row_i), []).append((index, policy))

    #map each seed to its clearest pure tail; keep other cases as exceptions
    memberships = []
    for row_i, options in membership_candidates.items():
        _, selected = min(options, key=lambda item: (
            -item[1]["margin"], -item[1]["support"], item[1]["group_id"]
        ))
        memberships.append({
            "posting_id": frame.iloc[row_i]["posting_id"],
            "group_id": selected["group_id"],
            "label": int(labels[row_i]),
        })
    membership = pd.DataFrame(memberships, columns=["posting_id", "group_id", "label"])
    groups = pd.DataFrame(rows)

    stem = "shap_extreme_families"
    groups.to_csv(RESULTS / f"{stem}.csv", index=False)
    membership.to_csv(RESULTS / f"{stem}_membership.csv", index=False)
    meta = {
        "run_seed": 42,
        "shap_space": "shap_all",
        "n_cases": len(frame),
        "n_dimensions": X.shape[1],
        "feature_names": list(X.columns),
        "discovery": "per-feature low/high longest pure extreme prefix; threshold only between distinct SHAP values",
        "score": "normalized gap to first opposite-label SHAP value, then support descending",
        "minimum_support": MIN_SUPPORT,
        "family_size_cap": None,
        "candidate_policies": len(groups),
        "assigned_seed_cases": len(membership),
        "singleton_exception_cases": len(frame) - len(membership),
        "oracle_reviewed_direct": 500,
        "oracle_residual_corrections": residual_corrections,
        "uses_oracle_labels_for_source_purity": True,
        "recipient_selection": "canonical runner evaluates one-feature SHAP condition on live model SHAP",
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (RESULTS / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(groups[["group_id", "group_label", "support", "feature", "direction", "normalized_margin"]].head(30).to_string(index=False)
          if "feature" in groups else groups.head(30).to_string(index=False))


if __name__ == "__main__":
    main()

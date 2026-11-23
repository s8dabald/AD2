"""Canonical propagation methods and shared SHAP-space mechanics."""

from __future__ import annotations

import numpy as np

#exclude rule-like flags from raw-SHAP selection
FLAG_PREFIXES = ("promptly_", "weekend_", "nwh_", "top_n_", "high_cash_")

def get_feature_cols(df):
    """All model feature columns (everything except label, posting_id, pred_label, pred_score)."""
    return df.drop(columns=["label", "posting_id", "pred_label", "pred_score"]).columns.tolist()


def _adaptive_topk(row, pred_score):
    """Return adaptive decision-relevant SHAP prefix length and column order."""
    p = min(max(pred_score, 1e-7), 1.0 - 1e-7)
    logit = float(np.log(p / (1.0 - p)))
    base = logit - float(row.sum())
    order = np.argsort(-np.abs(row))
    cum = base + np.cumsum(row[order])
    tol = 1e-9 * max(1.0, abs(logit))
    side = np.where(cum >= tol, 1, -1)
    final = side[-1] if len(side) else 0
    dev = np.where(side != final)[0]
    k = int(dev[-1]) + 2 if len(dev) else 0
    return int(min(k, len(row))), order[:k]


def apply_shap_box_policy(shap_values, feature_names, seed_idx, policy):
    """Apply reviewer/discovered tags to live SHAP rows without labels."""
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim != 2 or shap_values.shape[1] != len(feature_names):
        raise ValueError("SHAP matrix columns must align with feature_names")
    if not 0 <= seed_idx < len(shap_values):
        raise IndexError(f"seed_idx out of range: {seed_idx}")
    tags = policy.get("tags")
    if not isinstance(tags, list) or not tags:
        raise ValueError("SHAP-box policy must contain non-empty tags list")

    mask = np.ones(len(shap_values), dtype=bool)
    feature_index = {name: i for i, name in enumerate(feature_names)}
    for tag in tags:
        feature = tag.get("feature")
        choice = tag.get("choice")
        if choice in ("exception", "exception_no_relevant_features") or feature == "*":
            mask[:] = False
            mask[seed_idx] = True
            break
        if choice == "ignore":
            continue
        if feature not in feature_index:
            raise ValueError(f"Policy feature {feature!r} absent from current SHAP space")
        bounds = tag.get("bounds")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"Policy tag for {feature!r} requires two bounds")
        lo, hi = map(float, bounds)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
            raise ValueError(f"Invalid bounds for policy feature {feature!r}: {bounds}")
        values = shap_values[:, feature_index[feature]]
        mask &= (values >= lo) & (values <= hi)

    if not mask[seed_idx]:
        raise ValueError("Stored SHAP-box policy excludes its own seed")
    return mask


def shap_box_selection(shap_values, feature_names, pred_score, seed_idx,
                       seed_label, true_labels):
    """Build this method's seed-centered SHAP box, then match live SHAP rows."""
    from shap_box_policy import create_shap_box_policy

    policy = create_shap_box_policy(
        shap_values, feature_names, pred_score, seed_idx, seed_label, true_labels
    )
    mask = apply_shap_box_policy(shap_values, feature_names, seed_idx, policy)
    return mask, policy

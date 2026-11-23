"""Acquire and replay per-seed SHAP-box policies."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from propagation import _adaptive_topk

POLICY_SCHEMA_VERSION = 1
POLICY_CONFIG_FIELDS = (
    "strategy", "l", "corrected_weights", "compute_shap", "early_stop",
    "chunk_size", "min_delta", "stop_patience", "seed", "retrain_every",
    "propagation_mode", "split_training",
)


def _optimize_shap_box_tags(shap_values, topk, seed_idx, seed_label, true_labels):
    """Choose the widest per-feature tags whose matches share the seed's label."""
    labels = np.asarray(true_labels, dtype=int)
    seed = shap_values[seed_idx]
    pure = labels == int(seed_label)
    per_feature = []
    for j in topk:
        value = float(seed[j])
        options = [("ignore", np.ones(len(labels), dtype=bool), None)]
        for delta in (0.25, 0.10, 0.05):
            if value == 0.0:
                lo, hi = -1e-9, 1e-9
            else:
                lo, hi = sorted((value * (1.0 - delta), value * (1.0 + delta)))
            allowed = (shap_values[:, j] >= lo) & (shap_values[:, j] <= hi)
            options.append((f"delta_{delta:.2f}", allowed, [float(lo), float(hi)]))
        unique = {}
        for choice, allowed, bounds in options:
            unique.setdefault(np.packbits(allowed).tobytes(), (choice, allowed, bounds))
        per_feature.append(list(unique.values()))

    best_mask = np.zeros(len(labels), dtype=bool)
    best_mask[seed_idx] = True
    best_choices = ["ignore"] * len(topk)
    best_support = 0
    visited = set()

    def search(pos, current, choices):
        nonlocal best_mask, best_choices, best_support
        support = int(current.sum())
        if support < best_support or support == 0:
            return
        state_key = (pos, np.packbits(current).tobytes())
        if state_key in visited:
            return
        visited.add(state_key)
        if np.all(pure[current]):
            if support > best_support:
                best_support = support
                best_mask = current.copy()
                best_choices = choices + ["ignore"] * (len(per_feature) - pos)
            return
        if pos == len(per_feature):
            return
        options = sorted(
            per_feature[pos],
            key=lambda item: int((current & item[1]).sum()),
            reverse=True,
        )
        for choice, allowed, _bounds in options:
            narrowed = current & allowed
            if int(narrowed.sum()) >= best_support:
                search(pos + 1, narrowed, choices + [choice])

    search(0, np.ones(len(labels), dtype=bool), [])
    if best_support <= 1:
        only_seed = np.zeros(len(labels), dtype=bool)
        only_seed[seed_idx] = True
        return [{"feature": "*", "choice": "exception", "bounds": None}], only_seed

    policy = []
    for j, choice in zip(topk, best_choices):
        bounds = None
        if choice != "ignore":
            delta = float(choice.removeprefix("delta_"))
            value = float(seed[j])
            if value == 0.0:
                bounds = [-1e-9, 1e-9]
            else:
                bounds = [
                    float(x) for x in sorted(
                        (value * (1.0 - delta), value * (1.0 + delta))
                    )
                ]
        policy.append({"feature": int(j), "choice": choice, "bounds": bounds})
    return policy, best_mask


def create_shap_box_policy(shap_values, feature_names, pred_score, seed_idx,
                           seed_label, true_labels):
    """Find one seed's SHAP-box policy using stored labels."""
    shap_values = np.asarray(shap_values, dtype=float)
    scores = np.asarray(pred_score, dtype=float)
    if shap_values.ndim != 2 or shap_values.shape[1] != len(feature_names):
        raise ValueError("SHAP matrix columns must align with feature_names")
    if len(scores) != len(shap_values) or len(true_labels) != len(shap_values):
        raise ValueError("SHAP scores and labels must align with SHAP rows")
    if not 0 <= seed_idx < len(shap_values):
        raise IndexError(f"seed_idx out of range: {seed_idx}")

    k, topk = _adaptive_topk(shap_values[seed_idx], float(scores[seed_idx]))
    if k == 0:
        tags = [{"feature": "*", "choice": "exception_no_relevant_features", "bounds": None}]
        raw_family_size = 1
    else:
        tags, mask = _optimize_shap_box_tags(
            shap_values, topk, seed_idx, seed_label, true_labels
        )
        raw_family_size = int(mask.sum())
        for tag in tags:
            if isinstance(tag.get("feature"), (int, np.integer)):
                tag["feature"] = feature_names[int(tag["feature"])]
    return {
        "k": int(k),
        "features": [feature_names[int(j)] for j in topk],
        "tags": tags,
        "raw_family_size": raw_family_size,
        "exception": bool(raw_family_size <= 1),
    }


def create_reviewer_shap_box_policy(shap_values, feature_names, pred_score,
                                   seed_idx, feature_choices):
    """Turn reviewer choices into SHAP ranges around the seed, without labels."""
    shap_values = np.asarray(shap_values, dtype=float)
    scores = np.asarray(pred_score, dtype=float)
    if shap_values.ndim != 2 or shap_values.shape[1] != len(feature_names):
        raise ValueError("SHAP matrix columns must align with feature_names")
    if len(scores) != len(shap_values):
        raise ValueError("SHAP scores must align with SHAP rows")
    if not 0 <= seed_idx < len(shap_values):
        raise IndexError(f"seed_idx out of range: {seed_idx}")

    k, topk = _adaptive_topk(shap_values[seed_idx], float(scores[seed_idx]))
    if k == 0:
        return {
            "k": 0,
            "features": [],
            "tags": [{
                "feature": "*",
                "choice": "exception_no_relevant_features",
                "bounds": None,
            }],
            "raw_family_size": 1,
            "exception": True,
        }

    choices = feature_choices or {}
    allowed = {"ignore", "delta_0.25", "delta_0.10", "delta_0.05"}
    tags = []
    for position in topk:
        feature = feature_names[int(position)]
        choice = choices.get(feature, "ignore")
        if choice not in allowed:
            raise ValueError(f"Unsupported reviewer tag {choice!r} for {feature!r}")
        bounds = None
        if choice != "ignore":
            delta = float(choice.removeprefix("delta_"))
            value = float(shap_values[seed_idx, position])
            if value == 0.0:
                bounds = [-1e-9, 1e-9]
            else:
                bounds = [
                    float(bound) for bound in sorted(
                        (value * (1.0 - delta), value * (1.0 + delta))
                    )
                ]
        tags.append({"feature": feature, "choice": choice, "bounds": bounds})

    return {
        "k": int(k),
        "features": [feature_names[int(position)] for position in topk],
        "tags": tags,
        "raw_family_size": None,
        "exception": False,
    }


class PolicyReplay:
    def __init__(self, source_run_id, policies):
        self.source_run_id = source_run_id
        self._policies = policies
        self._consumed = set()

    def policy_for(self, seed_id):
        key = str(seed_id)
        if key not in self._policies:
            raise ValueError(f"No stored SHAP-box policy for reviewed seed {key!r}")
        if key in self._consumed:
            raise ValueError(f"Stored SHAP-box policy already consumed for seed {key!r}")
        self._consumed.add(key)
        return self._policies[key]

    def assert_complete(self):
        unused = sorted(set(self._policies) - self._consumed)
        if unused:
            sample = unused[:5]
            raise ValueError(
                f"Replay ended with {len(unused)} unused policies; first seed IDs: {sample}"
            )


def load_policy_replay(path, expected_config):
    """Load discovery-run policy records and validate config before replay."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Policy discovery JSONL not found: {path}")
    run_record = None
    summary_record = None
    policy_records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if record.get("kind") == "run":
                if run_record is not None:
                    raise ValueError(f"Multiple run headers in policy log: {path}")
                run_record = record
            elif record.get("kind") == "summary":
                summary_record = record
            elif record.get("kind") == "policy":
                policy_records.append(record)

    if run_record is None:
        raise ValueError(f"No run header in policy discovery log: {path}")
    if summary_record is None or "final_precision" not in summary_record:
        raise ValueError(f"Policy discovery run is incomplete: {path}")
    source_config = run_record.get("config", {})
    source_run_id = str(run_record.get("run_id", ""))
    if source_config.get("policy_mode") != "discover":
        raise ValueError("Replay source must be a policy_mode='discover' run")
    for key in POLICY_CONFIG_FIELDS:
        if source_config.get(key) != expected_config.get(key):
            raise ValueError(
                f"Policy replay config mismatch for {key}: "
                f"source={source_config.get(key)!r}, current={expected_config.get(key)!r}"
            )

    policies = {}
    for record in policy_records:
        if str(record.get("run_id", "")) != source_run_id:
            raise ValueError("Policy records do not belong to the discovery run header")
        if record.get("policy_schema_version") != POLICY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported/missing policy schema in discovery log: "
                f"{record.get('policy_schema_version')!r}"
            )
        seed_id = str(record.get("seed_id", ""))
        policy = record.get("policy")
        if not seed_id or not isinstance(policy, dict):
            raise ValueError("Malformed policy record: expected seed_id and policy object")
        if seed_id in policies:
            raise ValueError(f"Duplicate stored policy for seed {seed_id!r}")
        policies[seed_id] = policy
    if not policies:
        raise ValueError(f"No policy records in discovery log: {path}")
    return PolicyReplay(source_run_id, policies)

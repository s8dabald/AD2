"""Compare canonical timing arms with live shap_box / shap_box_u propagation."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from propagation import FLAG_PREFIXES, _adaptive_topk
from legacy.experiments.shap_family_propagation import (
    CORRECTED_WEIGHT,
    REVIEW_BATCH_SIZE,
    REVIEW_ROUNDS,
    ROOT_CACHE,
    SEED,
    TRUSTED_FRACTION,
    feature_matrix,
    fit_split_model,
    load_initial,
)


RESULTS = ROOT / "legacy" / "experiments" / "results"
REVIEW_BUDGET = REVIEW_BATCH_SIZE * REVIEW_ROUNDS
FIXED_DELTA = 0.25
SCHEDULES = ((10, 50), (20, 25))
MODES = ("shap_box", "shap_box_u")
TIMINGS = ("immediate_pre_review_shap", "post_review_shap_next_round")


def metric_pack(y_true, y_pred, y_score, prefix: str) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return {f"{prefix}_{key}": None for key in
                ("n", "precision", "recall", "f1", "auc", "tp", "fp", "fn", "tn")}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y_true)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size == 2 else None,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    return {f"{prefix}_{key}": value for key, value in out.items()}


def model_metrics(model, X, frame, fit_ids):
    pool = Pool(X)
    score = model.predict_proba(pool)[:, 1]
    pred = np.asarray(model.predict(pool)).reshape(-1).astype(int)
    y = frame["label"].to_numpy(dtype=int)
    ids = frame["posting_id"].astype(str).to_numpy()
    fit_mask = np.isin(ids, list(fit_ids))
    metrics = metric_pack(y, pred, score, "model_full")
    metrics.update(metric_pack(y[~fit_mask], pred[~fit_mask], score[~fit_mask], "model_nonfit"))
    return score, pred, metrics


def select_batch(frame, trusted_ids, reviewed_ids, uncertainty, tie_break, batch_size):
    ids = frame["posting_id"].astype(str)
    eligible = ~ids.isin(trusted_ids | reviewed_ids).to_numpy()
    rows = np.flatnonzero(eligible)
    order = rows[np.lexsort((tie_break[rows], -uncertainty[rows]))]
    chosen = order[:batch_size]
    if len(chosen) != batch_size:
        raise ValueError(f"Expected {batch_size} reviews, selected {len(chosen)}")
    return ids.iloc[chosen].tolist()


def candidate_delta(mode, uncertainty, fixed_delta=FIXED_DELTA):
    if mode == "shap_box":
        return np.full(len(uncertainty), fixed_delta, dtype=float)
    ranks = pd.Series(uncertainty).rank(method="average", pct=True).to_numpy(dtype=float)
    return 0.5 * np.clip((ranks - 0.5) / 0.5, 0.0, 1.0)


def seed_box(shap_values, feature_names, pred_score, pred_label, seed_i,
             seed_label, deltas, mode, fixed_delta=FIXED_DELTA,
             true_labels=None):
    """Match exact production shap_box geometry on production shap_raw features."""
    row = shap_values[seed_i]
    k, topk = _adaptive_topk(row, float(pred_score[seed_i]))
    mask = np.zeros(len(shap_values), dtype=bool)
    features = [feature_names[int(j)] for j in topk]
    seed_delta = float(deltas[seed_i]) if mode == "shap_box_u" else fixed_delta
    bands = []
    oracle_choices = []
    if k == 0:
        mask[seed_i] = True
        if mode == "oracle_feature_policy":
            oracle_choices = [{"feature": "*", "choice": "exception_no_relevant_features", "bounds": None}]
    elif mode == "oracle_feature_policy":
        if true_labels is None:
            raise ValueError("Oracle feature policy requires labels for offline policy discovery")
        choices, oracle_mask = optimize_oracle_feature_policy(
            shap_values, topk, seed_i, int(seed_label), true_labels
        )
        for choice in choices:
            if isinstance(choice.get("feature"), int):
                choice["feature"] = feature_names[choice["feature"]]
        oracle_choices = choices
        mask = oracle_mask
        bands = [choice["bounds"] for choice in choices if choice["choice"] != "ignore"]
    else:
        mask[:] = True
        for j in topk:
            v = float(row[j])
            if v == 0.0:
                lo, hi = -1e-9, 1e-9
                mask &= np.abs(shap_values[:, j]) <= 1e-9
            else:
                d = deltas if mode == "shap_box_u" else seed_delta
                raw_lo = v * (1.0 - d)
                raw_hi = v * (1.0 + d)
                lo = np.minimum(raw_lo, raw_hi)
                hi = np.maximum(raw_lo, raw_hi)
                mask &= (shap_values[:, j] >= lo) & (shap_values[:, j] <= hi)
                if mode == "shap_box_u":
                    lo_seed = sorted((v * (1.0 - seed_delta), v * (1.0 + seed_delta)))
                    bands.append([float(lo_seed[0]), float(lo_seed[1])])
                    continue
            bands.append([float(np.min(lo)), float(np.max(hi))]
                         if isinstance(lo, np.ndarray) else [float(lo), float(hi)])

    seed_pred = int(pred_label[seed_i])
    if mode == "shap_box_u":
        #expand error seeds only and admit same-prediction candidates
        #never overwrite directly reviewed cases
        mask &= pred_label == seed_pred
        if int(seed_label) == seed_pred:
            mask[:] = False
        mask[seed_i] = True
    elif not mask.any():
        mask[seed_i] = True

    return mask, {
        "k": int(k), "features": features, "bands": bands,
        "oracle_choices": oracle_choices,
        "oracle_exception": bool(mode == "oracle_feature_policy" and mask.sum() <= 1),
        "seed_delta": seed_delta, "seed_pred_label": seed_pred,
        "seed_is_error": int(seed_label) != seed_pred,
        "raw_family_size": int(mask.sum()),
    }


ORACLE_DELTAS = (0.25, 0.10, 0.05)


def optimize_oracle_feature_policy(shap_values, topk, seed_i, seed_label, true_labels):
    """Choose per-feature tags with the broadest same-label SHAP support.

    Each selected dimension may be ignored or use one of three signed relative
    bands. Candidate matching applies emitted tags to live SHAP values.
    """
    labels = np.asarray(true_labels, dtype=int)
    seed = shap_values[seed_i]
    pure = labels == int(seed_label)
    per_feature = []
    for j in topk:
        value = float(seed[j])
        options = [("ignore", np.ones(len(labels), dtype=bool), None)]
        for delta in ORACLE_DELTAS:
            if value == 0.0:
                lo, hi = -1e-9, 1e-9
            else:
                lo, hi = sorted((value * (1.0 - delta), value * (1.0 + delta)))
            allowed = (shap_values[:, j] >= lo) & (shap_values[:, j] <= hi)
            options.append((f"delta_{delta:.2f}", allowed, [float(lo), float(hi)]))
        #remove duplicate masks and keep the broadest option first
        unique = {}
        for choice, allowed, bounds in options:
            key = np.packbits(allowed).tobytes()
            unique.setdefault(key, (choice, allowed, bounds))
        per_feature.append(list(unique.values()))

    best_mask = np.zeros(len(labels), dtype=bool)
    best_mask[seed_i] = True
    best_choices = ["ignore"] * len(topk)
    best_support = 0
    visited = set()

    def search(pos, current, choices):
        nonlocal best_mask, best_choices, best_support
        support = int(current.sum())
        if support < best_support:
            return
        if support == 0:
            return
        state_key = (pos, np.packbits(current).tobytes())
        if state_key in visited:
            return
        visited.add(state_key)
        if np.all(pure[current]):
            #prefer the broader policy when tighter bands cannot improve support
            if support > best_support:
                best_support = support
                best_mask = current.copy()
                best_choices = choices + ["ignore"] * (len(per_feature) - pos)
            return
        if pos == len(per_feature):
            return
        #narrow the impure set with remaining reviewed features
        options = sorted(per_feature[pos], key=lambda item: int((current & item[1]).sum()), reverse=True)
        for choice, allowed, _bounds in options:
            narrowed = current & allowed
            if int(narrowed.sum()) >= best_support:
                search(pos + 1, narrowed, choices + [choice])

    search(0, np.ones(len(labels), dtype=bool), [])
    if best_support <= 1:
        only_seed = np.zeros(len(labels), dtype=bool)
        only_seed[seed_i] = True
        return ([{"feature": "*", "choice": "exception", "bounds": None}], only_seed)

    policy = []
    for j, choice in zip(topk, best_choices):
        bounds = None
        if choice != "ignore":
            delta = float(choice.removeprefix("delta_"))
            value = float(seed[j])
            if value == 0.0:
                bounds = [-1e-9, 1e-9]
            else:
                bounds = [float(x) for x in sorted((value * (1.0 - delta), value * (1.0 + delta)))]
        policy.append({"feature": int(j), "choice": choice, "bounds": bounds})
    return policy, best_mask


def propagate_batch(mode, timing, round_i, batch_ids, direct_labels, ids,
                    true_by_id, shap_values, feature_names, pred_score,
                    pred_label, uncertainty, fit_ids, prop_targets,
                    fixed_delta=FIXED_DELTA, true_labels=None,
                    propagation_limit=None, recipient_selection="all"):
    """Apply each reviewed seed's live-SHAP box; first matching seed wins."""
    id_to_row = {pid: i for i, pid in enumerate(ids)}
    deltas = candidate_delta(mode, uncertainty, fixed_delta=fixed_delta)
    review_rows, events = [], []
    new_ids = set()
    candidate_sum = 0
    fit_blocked = 0
    conflicts = 0
    for rank, seed_id in enumerate(batch_ids, start=1):
        seed_i = id_to_row[seed_id]
        mask, rule = seed_box(
            shap_values, feature_names, pred_score, pred_label, seed_i,
            direct_labels[seed_id], deltas, mode, fixed_delta=fixed_delta,
            true_labels=true_labels,
        )
        matched = set(ids[mask])
        candidate_sum += len(matched)
        blocked = matched & fit_ids
        fit_blocked += len(blocked)
        eligible = matched - fit_ids
        conflicts += len(eligible & set(prop_targets))
        recipients = eligible - set(prop_targets)
        eligible_unpropagated = len(recipients)
        selected_rows = np.sort(np.asarray([id_to_row[pid] for pid in recipients], dtype=int))
        selection_scores = {}
        if propagation_limit is not None and len(selected_rows) > propagation_limit:
            if recipient_selection == "nearest_shap":
                _k, topk = _adaptive_topk(
                    shap_values[seed_i], float(pred_score[seed_i])
                )
                if len(topk):
                    seed_profile = shap_values[seed_i, topk]
                    candidate_profiles = shap_values[np.ix_(selected_rows, topk)]
                    scale = np.maximum(0.10 * np.abs(seed_profile), 1e-9)
                    distances = np.sqrt(np.sum(((candidate_profiles - seed_profile) / scale) ** 2, axis=1))
                else:
                    distances = np.zeros(len(selected_rows), dtype=float)
                order = np.lexsort((selected_rows, distances))
                chosen_positions = order[:propagation_limit]
                selection_scores = {
                    int(selected_rows[pos]): float(distances[pos]) for pos in chosen_positions
                }
            elif recipient_selection == "highest_uncertainty":
                scores = np.asarray(uncertainty, dtype=float)[selected_rows]
                order = np.lexsort((selected_rows, -scores))
                chosen_positions = order[:propagation_limit]
                selection_scores = {
                    int(selected_rows[pos]): float(scores[pos]) for pos in chosen_positions
                }
            else:
                raise ValueError(f"Unknown recipient selection: {recipient_selection}")
            selected_rows = selected_rows[chosen_positions]
            recipients = {ids[row] for row in selected_rows}
        elif recipient_selection == "nearest_shap" and len(selected_rows):
            _k, topk = _adaptive_topk(shap_values[seed_i], float(pred_score[seed_i]))
            if len(topk):
                scale = np.maximum(0.10 * np.abs(shap_values[seed_i, topk]), 1e-9)
                profiles = shap_values[np.ix_(selected_rows, topk)]
                distances = np.sqrt(np.sum(((profiles - shap_values[seed_i, topk]) / scale) ** 2, axis=1))
            else:
                distances = np.zeros(len(selected_rows), dtype=float)
            selection_scores = {int(row): float(distance) for row, distance in zip(selected_rows, distances)}
        elif recipient_selection == "highest_uncertainty" and len(selected_rows):
            selection_scores = {
                int(row): float(uncertainty[row]) for row in selected_rows
            }

        recipient_order = [ids[row] for row in selected_rows]
        for pid in recipient_order:
            row_i = id_to_row[pid]
            prop_targets[pid] = int(direct_labels[seed_id])
            new_ids.add(pid)
            events.append({
                "mode": mode, "timing": timing, "round": round_i,
                "posting_id": pid, "propagated_label": int(direct_labels[seed_id]),
                "true_label": int(true_by_id[pid]),
                "seed_posting_id": seed_id,
                "recipient_selection": recipient_selection,
                "selection_score": selection_scores.get(row_i),
            })
        rule.update({
            "mode": mode, "timing": timing, "round": round_i,
            "review_rank_in_batch": rank,
            "review_budget": (round_i - 1) * len(batch_ids) + rank,
            "seed_posting_id": seed_id,
            "seed_label": int(direct_labels[seed_id]),
            "candidate_matches": len(matched),
            "fit_blocked": len(blocked),
            "eligible_unpropagated": eligible_unpropagated,
            "recipient_selection": recipient_selection,
            "propagation_limit": propagation_limit,
            "new_propagated": len(recipients),
            "new_propagated_correct": sum(true_by_id[pid] == direct_labels[seed_id] for pid in recipients),
        })
        review_rows.append(rule)
    return review_rows, events, new_ids, candidate_sum, fit_blocked, conflicts


def flatten_metrics(row):
    flattened = {key: value for key, value in row.items() if not isinstance(value, dict)}
    for scope in (
        "model_metrics", "final_model_metrics",
        "fit_target_metrics", "final_fit_target_metrics",
    ):
        for key, value in (row.get(scope) or {}).items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flattened[f"{scope}_{key}_{subkey}"] = subvalue
            else:
                flattened[f"{scope}_{key}"] = value
    return flattened


def fit_target_agreement(pred, score, ids, trusted_targets, direct_labels,
                         propagated_targets):
    """Measure model agreement with labels it was trained on, by label source."""
    targets = dict(trusted_targets)
    targets.update(propagated_targets)
    targets.update(direct_labels)
    n_fit = len(targets)
    n_positive = int(sum(targets.values()))
    anomaly_weight = max(5.0, n_fit / n_positive) if n_positive else 5.0
    row_by_id = {pid: row for row, pid in enumerate(ids)}
    source_targets = {
        "pseudo": {
            pid: label for pid, label in trusted_targets.items()
            if pid not in direct_labels and pid not in propagated_targets
        },
        "propagated": propagated_targets,
        "direct": direct_labels,
    }
    result = {}
    probs = np.clip(np.asarray(score, dtype=float), 1e-15, 1.0 - 1e-15)
    predictions = np.asarray(pred, dtype=int)
    for source, labels in source_targets.items():
        if not labels:
            result[source] = {
                "n": 0, "agreement": None, "log_loss": None,
                "objective_weighted_agreement": None,
                "objective_weighted_log_loss": None,
            }
            continue
        rows = np.fromiter((row_by_id[pid] for pid in labels), dtype=int, count=len(labels))
        y = np.fromiter((labels[ids[row]] for row in rows), dtype=int, count=len(rows))
        matched = predictions[rows] == y
        p = probs[rows]
        losses = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        weights = np.where(y == 1, anomaly_weight, 1.0)
        if source == "direct":
            weights *= CORRECTED_WEIGHT
        result[source] = {
            "n": int(len(rows)),
            "agreement": float(np.mean(matched)),
            "log_loss": float(np.mean(losses)),
            "objective_weighted_agreement": float(np.average(matched, weights=weights)),
            "objective_weighted_log_loss": float(np.average(losses, weights=weights)),
        }
    return result


def run_arm(mode, timing, batch_size, rounds, frame, X, pool, initial_model,
            initial_shap, initial_score, trusted_ids, trusted_targets, tie_break,
            *, fixed_delta=FIXED_DELTA, propagation_enabled=True,
            propagation_limit=None, recipient_selection="all",
            print_nonfit_metrics=True, feature_weight_mode="uniform",
            feature_weight_strength=0.5, seed=SEED,
            track_fit_target_agreement=False, query_policy="uncertainty",
            exclude_trusted_from_queries=True,
            exclude_trusted_from_propagation=True):
    ids = frame["posting_id"].astype(str).to_numpy()
    true_by_id = dict(zip(ids, frame["label"].to_numpy(dtype=int)))
    feature_names_all = list(X.columns)
    row_by_id = {pid: row for row, pid in enumerate(ids)}
    raw_cols = [i for i, name in enumerate(feature_names_all)
                if not name.startswith(FLAG_PREFIXES)]
    raw_names = [feature_names_all[i] for i in raw_cols]
    direct_labels: dict[str, int] = {}
    reviewed_ids: set[str] = set()
    prop_targets: dict[str, int] = {}
    model = initial_model
    query_score = np.asarray(initial_score, dtype=float)
    query_pred = frame["pred_label"].to_numpy(dtype=int)
    weight_query_pred = query_pred.copy()
    shap_all = np.asarray(initial_shap, dtype=float)
    if shap_all.shape != X.shape:
        raise ValueError(f"SHAP shape {shap_all.shape} does not match feature matrix {X.shape}")
    shap_raw = shap_all[:, raw_cols]
    random_priority = np.random.RandomState(seed).random_sample(len(ids))
    round_rows, review_rows, propagation_events, direct_review_rows = [], [], [], []
    feature_weight_rows = []
    reviewed_error_ids: set[str] = set()
    last_feature_weights = None
    started = time.perf_counter()

    def compute_feature_weights():
        if feature_weight_mode == "uniform":
            return None
        if feature_weight_mode == "static_initial_shap":
            source = np.abs(np.asarray(initial_shap, dtype=float))
        else:
            source_ids = set(direct_labels)
            if feature_weight_mode == "reviewed_error_shap":
                source_ids = reviewed_error_ids or source_ids
            source_rows = np.asarray([row_by_id[pid] for pid in source_ids], dtype=int)
            source = np.abs(shap_all[source_rows])
        importance = np.mean(source, axis=0)
        mean_importance = float(np.mean(importance))
        if not np.isfinite(mean_importance) or mean_importance <= 1e-15:
            return np.ones(X.shape[1], dtype=float).tolist()
        normalized = importance / mean_importance
        weights = np.clip(
            np.power(np.maximum(normalized, 1e-6), feature_weight_strength),
            0.25, 4.0,
        )
        return weights.astype(float).tolist()

    for round_i in range(1, rounds + 1):
        uncertainty = 1.0 - 2.0 * np.abs(query_score - 0.5)
        if query_policy not in ("uncertainty", "random"):
            raise ValueError(f"Unsupported query policy: {query_policy}")
        query_values = uncertainty if query_policy == "uncertainty" else random_priority
        query_excluded = trusted_ids if exclude_trusted_from_queries else set()
        batch_ids = select_batch(
            frame, query_excluded, reviewed_ids, query_values, tie_break, batch_size
        )
        reviewed_error_ids.update(
            pid for pid in batch_ids
            if int(weight_query_pred[row_by_id[pid]]) != int(true_by_id[pid])
        )
        direct_labels.update({pid: true_by_id[pid] for pid in batch_ids})
        reviewed_ids.update(batch_ids)
        direct_review_rows.extend({
            "mode": mode, "timing": timing, "round": round_i,
            "review_budget": (round_i - 1) * batch_size + rank,
            "posting_id": pid, "reviewed_label": int(direct_labels[pid]),
        } for rank, pid in enumerate(batch_ids, start=1))

        if timing == "immediate_pre_review_shap":
            current_fit_ids = trusted_ids | set(direct_labels) | set(prop_targets)
            propagation_fit_ids = (
                trusted_ids if exclude_trusted_from_propagation else set()
            ) | set(direct_labels) | set(prop_targets)
            if propagation_enabled:
                review_part, events, new_ids, candidates, blocked, conflicts = propagate_batch(
                    mode, timing, round_i, batch_ids, direct_labels, ids, true_by_id,
                    shap_raw, raw_names, query_score, query_pred,
                    uncertainty, propagation_fit_ids, prop_targets, fixed_delta=fixed_delta,
                    true_labels=frame["label"].to_numpy(dtype=int),
                    propagation_limit=propagation_limit,
                    recipient_selection=recipient_selection,
                )
            else:
                review_part, events, new_ids = [], [], set()
                candidates = blocked = conflicts = 0
            review_rows.extend(review_part)
            propagation_events.extend(events)
            fit_started = time.perf_counter()
            last_feature_weights = compute_feature_weights()
            model = fit_split_model(frame, trusted_ids, trusted_targets, direct_labels,
                                    extra_targets=prop_targets,
                                    feature_weights=last_feature_weights, seed=seed)
            query_score, pred, metrics = model_metrics(
                model, X, frame, trusted_ids | set(direct_labels) | set(prop_targets)
            )
            target_metrics = (
                fit_target_agreement(
                    pred, query_score, ids, trusted_targets, direct_labels, prop_targets
                ) if track_fit_target_agreement else None
            )
            query_pred = pred
            weight_query_pred = pred
            if propagation_enabled or feature_weight_mode != "uniform":
                shap_all = np.asarray(model.get_feature_importance(pool, type="ShapValues")[:, :-1], dtype=float)
                shap_raw = shap_all[:, raw_cols]
            row = {
                "mode": mode, "timing": timing, "round": round_i,
                "review_budget": len(direct_labels), "batch_size": batch_size,
                "fit_rows": len(trusted_ids | set(direct_labels) | set(prop_targets)),
                "new_propagated": len(new_ids), "active_propagated": len(prop_targets),
                "new_propagation_accuracy": (
                    float(sum(true_by_id[pid] == prop_targets[pid] for pid in new_ids) / len(new_ids))
                    if new_ids else None
                ),
                "candidate_matches_sum": candidates, "fit_blocked_sum": blocked,
                "overlap_conflicts": conflicts,
                "fit_seconds": round(time.perf_counter() - fit_started, 2),
                "model_metrics": metrics,
                "fit_target_metrics": target_metrics,
                "feature_weight_mode": feature_weight_mode,
                "feature_weight_strength": feature_weight_strength,
                "feature_weights": last_feature_weights,
            }
        else:
            #fit direct and prior propagated targets before matching fresh SHAP
            #queue new targets for the next fit
            for pid in batch_ids:
                prop_targets.pop(pid, None)
            fit_started = time.perf_counter()
            last_feature_weights = compute_feature_weights()
            model = fit_split_model(frame, trusted_ids, trusted_targets, direct_labels,
                                    extra_targets=prop_targets,
                                    feature_weights=last_feature_weights, seed=seed)
            score, pred, metrics = model_metrics(
                model, X, frame, trusted_ids | set(direct_labels) | set(prop_targets)
            )
            target_metrics = (
                fit_target_agreement(
                    pred, score, ids, trusted_targets, direct_labels, prop_targets
                ) if track_fit_target_agreement else None
            )
            weight_query_pred = pred
            next_shap_all = (
                np.asarray(model.get_feature_importance(pool, type="ShapValues")[:, :-1], dtype=float)
                if propagation_enabled or feature_weight_mode != "uniform" else None
            )
            fit_ids = trusted_ids | set(direct_labels) | set(prop_targets)
            propagation_fit_ids = (
                trusted_ids if exclude_trusted_from_propagation else set()
            ) | set(direct_labels) | set(prop_targets)
            current_prop = dict(prop_targets)
            queued_targets = dict(prop_targets)
            if propagation_enabled:
                direct_review_row, events, new_ids, candidates, blocked, conflicts = propagate_batch(
                    mode, timing, round_i, batch_ids, direct_labels, ids, true_by_id,
                    next_shap_all[:, raw_cols], raw_names, score,
                    pred,
                    1.0 - 2.0 * np.abs(score - 0.5), propagation_fit_ids, queued_targets,
                    fixed_delta=fixed_delta, true_labels=frame["label"].to_numpy(dtype=int),
                    propagation_limit=propagation_limit,
                    recipient_selection=recipient_selection,
                )
            else:
                direct_review_row, events, new_ids = [], [], set()
                candidates = blocked = conflicts = 0
            review_rows.extend(direct_review_row)
            propagation_events.extend(events)
            new_only = set(queued_targets) - set(prop_targets)
            prop_targets = queued_targets
            row = {
                "mode": mode, "timing": timing, "round": round_i,
                "review_budget": len(direct_labels), "batch_size": batch_size,
                "fit_rows": len(fit_ids), "new_propagated": len(new_only),
                "active_propagated": len(prop_targets),
                "propagated_in_fit": len(current_prop),
                "new_propagation_accuracy": (
                    float(sum(true_by_id[pid] == prop_targets[pid] for pid in new_only) / len(new_only))
                    if new_only else None
                ),
                "candidate_matches_sum": candidates, "fit_blocked_sum": blocked,
                "overlap_conflicts": conflicts,
                "fit_seconds": round(time.perf_counter() - fit_started, 2),
                "model_metrics": metrics,
                "fit_target_metrics": target_metrics,
            }
            query_score = score
            if propagation_enabled:
                shap_all = next_shap_all
                shap_raw = shap_all[:, raw_cols]
            elif feature_weight_mode != "uniform":
                shap_all = next_shap_all

            if round_i == rounds:
                final_started = time.perf_counter()
                model = fit_split_model(frame, trusted_ids, trusted_targets, direct_labels,
                                        extra_targets=prop_targets,
                                        feature_weights=last_feature_weights, seed=seed)
                final_fit_ids = trusted_ids | set(direct_labels) | set(prop_targets)
                final_score, final_pred, final_metrics = model_metrics(model, X, frame, final_fit_ids)
                row["final_fit_rows"] = len(final_fit_ids)
                row["final_fit_seconds"] = round(time.perf_counter() - final_started, 2)
                row["final_model_metrics"] = final_metrics
                if track_fit_target_agreement:
                    row["final_fit_target_metrics"] = fit_target_agreement(
                        final_pred, final_score, ids, trusted_targets,
                        direct_labels, prop_targets,
                    )

        row["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        row.setdefault("feature_weight_mode", feature_weight_mode)
        row.setdefault("feature_weight_strength", feature_weight_strength)
        row.setdefault("feature_weights", last_feature_weights)
        if last_feature_weights is not None:
            feature_weight_rows.extend({
                "mode": feature_weight_mode,
                "strength": feature_weight_strength,
                "round": round_i,
                "review_budget": len(direct_labels),
                "feature": name,
                "feature_weight": weight,
            } for name, weight in zip(feature_names_all, last_feature_weights))
        round_rows.append(row)
        progress = {
            "mode": mode, "timing": timing, "batch": f"{round_i}/{rounds}",
            "reviews": len(direct_labels),
            "new_prop": row["new_propagated"], "fit_rows": row["fit_rows"],
            "P_full": row["model_metrics"].get("model_full_precision"),
            "R_full": row["model_metrics"].get("model_full_recall"),
        }
        if print_nonfit_metrics:
            progress["P_nonfit"] = row["model_metrics"].get("model_nonfit_precision")
            progress["R_nonfit"] = row["model_metrics"].get("model_nonfit_recall")
        print(json.dumps(progress, sort_keys=True), flush=True)

    return {
        "rounds": round_rows,
        "reviews": review_rows,
        "direct_reviews": direct_review_rows,
        "propagation_events": propagation_events,
        "active_propagated_targets": prop_targets,
        "feature_weights": feature_weight_rows,
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }


def write_report(path, protocol, results):
    lines = [
        "# SHAP-box timing comparison",
        "",
        "Seed 42; 500 direct reviews per trajectory; SHAP_RAW (52 dimensions); "
        "direct weight 100, propagated weight 1.",
        "Metrics reported at every fit checkpoint; full-data and non-fit metrics shown separately.",
        "",
        "## Trajectory runtime",
        "",
        "| Schedule | Mode | Timing | Fit checkpoints | Total seconds | Mean fit seconds |",
        "|---|---|---|---:|---:|---:|",
    ]
    for key, result in results.items():
        sched, mode, timing = key.split("|")
        fits = sum(1 + int(bool(row.get("final_model_metrics"))) for row in result["rounds"])
        mean_fit = sum(row["fit_seconds"] for row in result["rounds"]) / len(result["rounds"])
        lines.append(
            f"| {sched} | {mode} | {timing} | {fits} | "
            f"{result['runtime_seconds']:.2f} | {mean_fit:.2f} |"
        )
    lines.extend([
        "",
        "## Per-retrain checkpoints",
        "",
        "| Schedule | Mode | Timing | Fit | Reviews | P full | R full | FP | FN | P non-fit | R non-fit | New prop | Active prop | Prop acc | Fit rows | Fit sec | Elapsed sec |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key, result in results.items():
        sched, mode, timing = key.split("|")
        for row in result["rounds"]:
            m = row["model_metrics"]
            lines.append(
                f"| {sched} | {mode} | {timing} | {row['round']} | {row['review_budget']} | "
                f"{m.get('model_full_precision', 0):.5f} | {m.get('model_full_recall', 0):.5f} | "
                f"{m.get('model_full_fp', 0)} | {m.get('model_full_fn', 0)} | "
                f"{m.get('model_nonfit_precision', 0):.5f} | {m.get('model_nonfit_recall', 0):.5f} | "
                f"{row['new_propagated']} | {row['active_propagated']} | "
                f"{'' if row['new_propagation_accuracy'] is None else f'{row['new_propagation_accuracy']:.5f}'} | "
                f"{row['fit_rows']} | {row['fit_seconds']:.2f} | {row['elapsed_seconds']:.2f} |"
            )
            if row.get("final_model_metrics"):
                fm = row["final_model_metrics"]
                lines.append(
                    f"| {sched} | {mode} | {timing} | {row['round']} final | {row['review_budget']} | "
                    f"{fm.get('model_full_precision', 0):.5f} | {fm.get('model_full_recall', 0):.5f} | "
                    f"{fm.get('model_full_fp', 0)} | {fm.get('model_full_fn', 0)} | "
                    f"{fm.get('model_nonfit_precision', 0):.5f} | {fm.get('model_nonfit_recall', 0):.5f} | "
                    f"0 | {row['active_propagated']} |  | {row['final_fit_rows']} | "
                    f"{row['final_fit_seconds']:.2f} | {row['elapsed_seconds']:.2f} |"
                )
    lines.extend([
        "",
        f"`shap_box`: adaptive decision-space top-k, fixed signed SHAP bands δ={protocol.get('fixed_shap_box_delta', FIXED_DELTA):.2f}; no candidate-label gate.",
        "`direct_only`: no propagation; same query, fitting cadence, and review budget.",
        "`shap_box_u`: production candidate-rank δ formula (0…0.5), error seeds only, same-prediction candidates.",
        "Within-batch overlap uses canonical runner precedence: first reviewed seed assigns; direct labels override propagated targets; previous propagated targets are not overwritten.",
        "Oracle labels appear only on direct review seeds and offline propagation accuracy. Candidate matching uses live SHAP_RAW only.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    frame = load_initial()
    frame["posting_id"] = frame["posting_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    frame["pred_label"] = frame["pred_label"].astype(int)
    X = feature_matrix(frame)
    pool = Pool(X)
    ids = frame["posting_id"].astype(str)
    initial_model = CatBoostClassifier()
    initial_model.load_model(str(ROOT_CACHE / f"initial_seed_{SEED}.cbm"))
    initial_shap = np.asarray(np.load(ROOT_CACHE / f"initial_seed_{SEED}_shap.npy"), dtype=float)
    if initial_shap.shape != X.shape:
        raise ValueError(f"Cached SHAP shape {initial_shap.shape} != feature matrix {X.shape}")
    initial_score = frame["pred_score"].to_numpy(dtype=float)

    uncertainty = 1.0 - 2.0 * np.abs(initial_score - 0.5)
    tie_break = np.random.RandomState(SEED).random_sample(len(frame))
    trusted_order = np.lexsort((tie_break, uncertainty))
    trusted_ids = set(ids.iloc[trusted_order[:round(TRUSTED_FRACTION * len(frame))]])
    initial_targets = dict(zip(ids, frame["pred_label"].to_numpy(dtype=int)))
    trusted_targets = {pid: initial_targets[pid] for pid in trusted_ids}

    results = {}
    protocol = {
        "seed": SEED,
        "review_budget": REVIEW_BUDGET,
        "trusted_fraction": TRUSTED_FRACTION,
        "shap_space": "shap_raw",
        "shap_raw_features": [name for name in X.columns if not name.startswith(FLAG_PREFIXES)],
        "fixed_shap_box_delta": FIXED_DELTA,
        "shap_box_u_delta": "0.5 * clip((candidate uncertainty percentile rank - 0.5) / 0.5, 0, 1)",
        "timing_arms": list(TIMINGS),
        "schedules": [{"rounds": rounds, "batch_size": size} for rounds, size in SCHEDULES],
        "direct_weight": CORRECTED_WEIGHT,
        "propagated_weight": 1,
        "candidate_id_family_lookup": False,
        "conflict_policy": "canonical first-matching seed wins; direct labels override; existing propagated targets never overwritten",
        "candidate_fit_exclusion": "exclude trusted/direct/prior-propagated after live SHAP box match",
        "metrics": "full data plus non-fit subset at every model fit checkpoint",
    }
    for rounds, batch_size in SCHEDULES:
        schedule = f"{rounds}x{batch_size}"
        print(f"\n[schedule] {schedule}; budget={rounds * batch_size}", flush=True)
        for mode in MODES:
            for timing in TIMINGS:
                key = f"{schedule}|{mode}|{timing}"
                print(f"\n[run] {key}", flush=True)
                results[key] = run_arm(
                    mode, timing, batch_size, rounds, frame, X, pool,
                    initial_model, initial_shap, initial_score,
                    trusted_ids, trusted_targets, tie_break,
                )
                #save each completed trajectory in case the run is interrupted
                stem = RESULTS / "split_training_shap_box_timing"
                compact_results = {
                    key: {
                        "rounds": data["rounds"],
                        "runtime_seconds": data["runtime_seconds"],
                        "review_count": len(data["reviews"]),
                        "propagation_event_count": len(data["propagation_events"]),
                    }
                    for key, data in results.items()
                }
                payload = {
                    "protocol": protocol,
                    "completed_trajectories": list(results),
                    "results": compact_results,
                    "elapsed_seconds": round(time.perf_counter() - total_started, 2),
                }
                (stem.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
                rows = [flatten_metrics(row) | {"schedule": k.split("|")[0], "mode": k.split("|")[1], "timing": k.split("|")[2]}
                        for k, data in results.items() for row in data["rounds"]]
                pd.DataFrame(rows).to_csv(f"{stem}_checkpoints.csv", index=False)
                review_rows = [item for data in results.values() for item in data["reviews"]]
                pd.DataFrame(review_rows).to_csv(f"{stem}_seed_rules.csv", index=False)
                prop_rows = [item for data in results.values() for item in data["propagation_events"]]
                pd.DataFrame(prop_rows).to_csv(f"{stem}_propagated.csv", index=False)
                report = RESULTS / "split_training_shap_box_timing_report.md"
                write_report(report, protocol, results)
                print(f"[saved] {stem} + checkpoint report", flush=True)

    print(f"Complete in {time.perf_counter() - total_started:.1f}s", flush=True)


if __name__ == "__main__":
    main()

"""Compare immediate vs one-round-delayed SHAP_ALL-family matching in 10x50."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score

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


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "legacy" / "experiments" / "results"


def metric_pack(y_true, y_pred, y_score, prefix: str) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return {f"{prefix}_{key}": None for key in
                ("n", "precision", "recall", "auc", "tp", "fp", "fn", "tn")}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y_true)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size == 2 else None,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    return {f"{prefix}_{key}": value for key, value in out.items()}


def predict_and_explain(model, X: pd.DataFrame, pool: Pool | None = None):
    if pool is None:
        pool = Pool(X)
    score = model.predict_proba(pool)[:, 1]
    pred = np.asarray(model.predict(pool)).reshape(-1).astype(int)
    shap = np.asarray(model.get_feature_importance(pool, type="ShapValues")[:, :-1], dtype=float)
    if shap.shape != X.shape or not np.isfinite(shap).all():
        raise ValueError(f"Invalid SHAP_ALL matrix from current model: {shap.shape}")
    return score, pred, shap


def load_family_policies(
    frame: pd.DataFrame,
    feature_names: list[str],
    groups_path: Path,
    membership_path: Path,
    meta_path: Path,
    allow_exceptions: bool,
):
    groups = pd.read_csv(groups_path)
    membership = pd.read_csv(
        membership_path,
        dtype={"posting_id": str, "group_id": str},
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("shap_space") != "shap_all" or int(meta.get("n_dimensions", -1)) != len(feature_names):
        raise ValueError("Family brain and model SHAP dimensions differ.")
    if membership.posting_id.duplicated().any():
        raise ValueError("Family membership contains duplicate posting IDs.")
    ids = frame["posting_id"].astype(str)
    unknown_ids = set(membership.posting_id) - set(ids)
    if unknown_ids:
        raise ValueError(f"Family membership contains {len(unknown_ids)} unknown posting IDs.")
    if len(membership) != len(frame) and not allow_exceptions:
        raise ValueError(
            "Family membership is partial; pass --allow-exceptions to leave unmatched cases as singletons."
        )
    seed_groups = membership.set_index("posting_id").reindex(ids)["group_id"]
    if seed_groups.isna().any() and not allow_exceptions:
        raise ValueError("Some review seeds have no Oracle-family mapping.")

    labels_by_group = {}
    paths_by_group = {}
    support_by_group = {}
    feature_pos = {name: i for i, name in enumerate(feature_names)}
    for row in groups.itertuples(index=False):
        gid = str(row.group_id)
        path = json.loads(row.path)
        for condition in path:
            if condition["feature"] not in feature_pos:
                raise ValueError(f"Unknown SHAP feature in family {gid}: {condition['feature']}")
            if condition["op"] not in ("<=", ">"):
                raise ValueError(f"Unsupported family operator {condition['op']!r}")
        labels_by_group[gid] = int(row.group_label)
        paths_by_group[gid] = path
        support_by_group[gid] = int(row.support)
    seed_group_by_id = {
        posting_id: (None if pd.isna(group_id) else str(group_id))
        for posting_id, group_id in zip(ids, seed_groups)
    }
    if set(membership.group_id.astype(str)) - set(labels_by_group):
        raise ValueError("Family membership references an unknown group ID.")
    membership_labels = membership.set_index("posting_id").reindex(ids)["label"]
    mapped = membership_labels.notna().to_numpy()
    if not np.array_equal(
        membership_labels.loc[mapped].to_numpy(dtype=int),
        frame.loc[mapped, "label"].to_numpy(dtype=int),
    ):
        raise ValueError("Family membership labels do not match timing-run labels.")
    return seed_group_by_id, labels_by_group, paths_by_group, support_by_group, feature_pos, meta


def match_family_masks(shap_values: np.ndarray, group_ids: set[str], paths_by_group, feature_pos):
    """Apply selected source-family SHAP paths to current model SHAP vectors."""
    masks = {}
    for group_id in group_ids:
        mask = np.ones(len(shap_values), dtype=bool)
        for condition in paths_by_group[group_id]:
            col = feature_pos[condition["feature"]]
            threshold = float(condition["threshold"])
            if condition["op"] == "<=":
                mask &= shap_values[:, col] <= threshold
            else:
                mask &= shap_values[:, col] > threshold
        masks[group_id] = mask
    return masks


def model_and_policy_metrics(model, X, frame, fit_ids, direct_labels, propagated_targets):
    pool = Pool(X)
    score = model.predict_proba(pool)[:, 1]
    model_pred = np.asarray(model.predict(pool)).reshape(-1).astype(int)
    y = frame["label"].to_numpy(dtype=int)
    ids = frame["posting_id"].astype(str).to_numpy()
    fit_mask = np.isin(ids, list(fit_ids))
    model_metrics = metric_pack(y, model_pred, score, "model_full")
    model_metrics.update(metric_pack(y[~fit_mask], model_pred[~fit_mask], score[~fit_mask], "model_nonfit"))

    policy_pred = model_pred.copy()
    row_index = {pid: i for i, pid in enumerate(ids)}
    for pid, label in direct_labels.items():
        policy_pred[row_index[pid]] = int(label)
    for pid, label in propagated_targets.items():
        if pid not in direct_labels:
            policy_pred[row_index[pid]] = int(label)
    policy_metrics = metric_pack(y, policy_pred, score, "policy_full")
    policy_metrics.update(metric_pack(y[~fit_mask], policy_pred[~fit_mask], score[~fit_mask], "policy_nonfit"))
    return model_metrics, policy_metrics


def select_batch(frame, trusted_ids, reviewed_ids, uncertainty, tie_break, batch_size):
    ids = frame["posting_id"].astype(str)
    blocked = trusted_ids | reviewed_ids
    eligible = ~ids.isin(blocked).to_numpy()
    rows = np.flatnonzero(eligible)
    order = rows[np.lexsort((tie_break[rows], -uncertainty[rows]))]
    chosen = order[:batch_size]
    if len(chosen) != batch_size:
        raise ValueError(f"Expected {batch_size} direct reviews, got {len(chosen)}")
    return ids.iloc[chosen].tolist()


def apply_review_families(
    method, round_i, batch_ids, review_labels, frame_ids, true_by_id,
    current_shap, current_fit_ids, existing_prop_targets,
    seed_group_by_id, labels_by_group, paths_by_group, support_by_group,
    feature_pos, batch_size,
):
    selected_groups = {seed_group_by_id.get(pid) for pid in batch_ids}
    selected_groups.discard(None)
    masks = match_family_masks(current_shap, selected_groups, paths_by_group, feature_pos)
    assignments = []
    propagation_events = []
    newly_added = set()
    fit_blocked_total = 0
    family_candidates_total = 0
    for rank, seed_id in enumerate(batch_ids, start=1):
        family_id = seed_group_by_id.get(seed_id)
        if family_id is None:
            #review unassigned cases directly without activating propagation
            assignments.append({
                "method": method,
                "round": round_i,
                "review_rank_in_batch": rank,
                "review_budget": (round_i - 1) * batch_size + rank,
                "seed_posting_id": seed_id,
                "seed_label": int(review_labels[seed_id]),
                "family_id": None,
                "family_label": int(review_labels[seed_id]),
                "family_support_oracle": 1,
                "family_conditions": "[]",
                "family_status": "singleton_exception",
                "candidate_shap_matches": 1,
                "matches_already_in_fit": 1,
                "new_propagated_cases": 0,
                "new_propagated_correct": 0,
            })
            continue
        family_label = labels_by_group[family_id]
        seed_label = int(review_labels[seed_id])
        if family_label != seed_label:
            raise ValueError(f"Seed {seed_id}: family label != direct Oracle review label")
        member_mask = masks[family_id]
        matched_ids = set(frame_ids[member_mask])
        family_candidates_total += len(matched_ids)
        blocked = matched_ids & current_fit_ids
        fit_blocked_total += len(blocked)
        new_ids = matched_ids - current_fit_ids - set(existing_prop_targets)
        for posting_id in new_ids:
            existing_prop_targets[posting_id] = seed_label
            propagation_events.append({
                "method": method,
                "round": round_i,
                "posting_id": posting_id,
                "propagated_label": seed_label,
                "family_id": family_id,
                "seed_posting_id": seed_id,
            })
        newly_added.update(new_ids)
        assignments.append({
            "method": method,
            "round": round_i,
            "review_rank_in_batch": rank,
            "review_budget": (round_i - 1) * batch_size + rank,
            "seed_posting_id": seed_id,
            "seed_label": seed_label,
            "family_id": family_id,
            "family_label": family_label,
            "family_support_oracle": support_by_group[family_id],
            "family_conditions": json.dumps(paths_by_group[family_id], ensure_ascii=False),
            "family_status": "matched",
            "candidate_shap_matches": len(matched_ids),
            "matches_already_in_fit": len(blocked),
            "new_propagated_cases": len(new_ids),
            "new_propagated_correct": sum(true_by_id[pid] == seed_label for pid in new_ids),
        })
    return assignments, newly_added, family_candidates_total, fit_blocked_total, propagation_events


def run_immediate(
    frame, X, X_pool, initial_model, initial_shap, initial_score,
    trusted_ids, trusted_targets, tie_break, seed_group_by_id,
    labels_by_group, paths_by_group, support_by_group, feature_pos,
    batch_size, review_rounds,
):
    method = "immediate_pre_review_shap"
    ids = frame["posting_id"].astype(str).to_numpy()
    true_by_id = dict(zip(frame["posting_id"].astype(str), frame["label"].astype(int)))
    direct_labels = {}
    reviewed_ids = set()
    prop_targets = {}
    model = initial_model
    query_score = initial_score
    shap_values = initial_shap
    round_rows, review_rows, propagation_events = [], [], []
    round_started = time.perf_counter()

    for round_i in range(1, review_rounds + 1):
        print(f"[immediate] round {round_i}/{review_rounds}: select reviews from current model", flush=True)
        uncertainty = 1.0 - 2.0 * np.abs(query_score - 0.5)
        batch_ids = select_batch(frame, trusted_ids, reviewed_ids, uncertainty, tie_break, batch_size)
        direct_labels.update({pid: true_by_id[pid] for pid in batch_ids})
        reviewed_ids.update(batch_ids)

        current_fit_ids = trusted_ids | set(direct_labels) | set(prop_targets)
        assignments, new_ids, candidates, blocked, events = apply_review_families(
            method, round_i, batch_ids, direct_labels, ids, true_by_id,
            shap_values, current_fit_ids, prop_targets,
            seed_group_by_id, labels_by_group, paths_by_group, support_by_group,
            feature_pos, batch_size,
        )
        review_rows.extend(assignments)
        propagation_events.extend(events)
        print(
            f"[immediate] round {round_i}: family candidates={candidates:,}, "
            f"fit-blocked={blocked:,}, new labels={len(new_ids):,}; fitting...",
            flush=True,
        )

        model = fit_split_model(frame, trusted_ids, trusted_targets, direct_labels,
                                extra_targets=prop_targets)
        score, pred, next_shap = predict_and_explain(model, X, X_pool)
        fit_ids = trusted_ids | set(direct_labels) | set(prop_targets)
        model_metrics, policy_metrics = model_and_policy_metrics(
            model, X, frame, fit_ids, direct_labels, prop_targets)
        positives = sum(trusted_targets.values()) + sum(direct_labels.values()) + sum(prop_targets.values())
        fit_rows = len(fit_ids)
        row = {
            "method": method, "round": round_i, "review_budget": len(direct_labels),
            "reviewed_anomalies": int(sum(direct_labels.values())),
            "fit_rows": fit_rows, "outside_fit_rows": len(frame) - fit_rows,
            "new_propagated": len(new_ids), "active_propagated": len(prop_targets),
            "new_propagation_accuracy": (
                float(sum(true_by_id[pid] == prop_targets[pid] for pid in new_ids) / len(new_ids))
                if new_ids else None
            ),
            "class_weight_anomaly": max(5.0, fit_rows / positives),
            "direct_weight": CORRECTED_WEIGHT, "propagated_weight": 1,
            "model_metrics": model_metrics, "policy_metrics": policy_metrics,
            "fit_seconds": round(time.perf_counter() - round_started, 2),
        }
        round_rows.append(row)
        print(json.dumps({
            "method": method, "reviews": len(direct_labels), "new_prop": len(new_ids),
            "fit_rows": fit_rows, "P": model_metrics["model_full_precision"],
            "R": model_metrics["model_full_recall"],
        }, sort_keys=True), flush=True)
        query_score = score
        if round_i < review_rounds:
            shap_values = next_shap
        round_started = time.perf_counter()
    return round_rows, review_rows, prop_targets, propagation_events


def run_delayed(
    frame, X, X_pool, initial_model, initial_score, trusted_ids, trusted_targets,
    tie_break, seed_group_by_id, labels_by_group, paths_by_group,
    support_by_group, feature_pos, batch_size, review_rounds,
):
    method = "post_review_shap_next_round"
    ids = frame["posting_id"].astype(str).to_numpy()
    true_by_id = dict(zip(frame["posting_id"].astype(str), frame["label"].astype(int)))
    direct_labels = {}
    reviewed_ids = set()
    prop_targets = {}  #targets collected here enter the following fit
    model = initial_model
    query_score = initial_score
    round_rows, review_rows, propagation_events = [], [], []
    round_started = time.perf_counter()

    for round_i in range(1, review_rounds + 1):
        print(f"[delayed] round {round_i}/{review_rounds}: select reviews from current model", flush=True)
        uncertainty = 1.0 - 2.0 * np.abs(query_score - 0.5)
        batch_ids = select_batch(frame, trusted_ids, reviewed_ids, uncertainty, tie_break, batch_size)
        direct_labels.update({pid: true_by_id[pid] for pid in batch_ids})
        reviewed_ids.update(batch_ids)
        #direct review overrides earlier targets and weights
        for pid in batch_ids:
            prop_targets.pop(pid, None)

        #fit current reviews and labels queued by the previous round
        model = fit_split_model(frame, trusted_ids, trusted_targets, direct_labels,
                                extra_targets=prop_targets)
        score, pred, shap_values = predict_and_explain(model, X, X_pool)
        fit_ids = trusted_ids | set(direct_labels) | set(prop_targets)
        current_prop = dict(prop_targets)
        stage_model_metrics, _ = model_and_policy_metrics(
            model, X, frame, fit_ids, direct_labels, current_prop)

        #queue new SHAP matches for the next fit
        queued_targets = dict(prop_targets)
        assignments, new_ids, candidates, blocked, events = apply_review_families(
            method, round_i, batch_ids, direct_labels, ids, true_by_id,
            shap_values, fit_ids, queued_targets,
            seed_group_by_id, labels_by_group, paths_by_group, support_by_group,
            feature_pos, batch_size,
        )
        review_rows.extend(assignments)
        propagation_events.extend(events)
        new_only = set(queued_targets) - set(prop_targets)
        prop_targets = queued_targets
        stage_model_metrics, stage_policy_metrics = model_and_policy_metrics(
            model, X, frame, fit_ids, direct_labels, prop_targets)
        print(
            f"[delayed] round {round_i}: trained on reviews + prior propagation; "
            f"SHAP candidates={candidates:,}, fit-blocked={blocked:,}, "
            f"queued for next round={len(new_only):,}",
            flush=True,
        )

        fit_rows = len(fit_ids)
        positives = sum(trusted_targets.values()) + sum(direct_labels.values()) + sum(current_prop.values())
        row = {
            "method": method, "round": round_i, "review_budget": len(direct_labels),
            "reviewed_anomalies": int(sum(direct_labels.values())),
            "fit_rows": fit_rows, "outside_fit_rows": len(frame) - fit_rows,
            "new_propagated_queued": len(new_only),
            "propagated_currently_in_fit": len(current_prop),
            "active_propagation_queue": len(prop_targets),
            "new_propagation_accuracy": (
                float(sum(true_by_id[pid] == queued_targets[pid] for pid in new_only) / len(new_only))
                if new_only else None
            ),
            "class_weight_anomaly": max(5.0, fit_rows / positives),
            "direct_weight": CORRECTED_WEIGHT, "propagated_weight": 1,
            "stage_model_metrics": stage_model_metrics,
            "stage_policy_metrics": stage_policy_metrics,
            "final_model_metrics": None,
            "fit_seconds": round(time.perf_counter() - round_started, 2),
        }
        query_score = score
        if round_i == review_rounds:
            #fit the final queue because no next round will consume it
            final_fit_ids = trusted_ids | set(direct_labels) | set(prop_targets)
            model = fit_split_model(frame, trusted_ids, trusted_targets, direct_labels,
                                    extra_targets=prop_targets)
            final_score, final_pred, _ = predict_and_explain(model, X, X_pool)
            row["final_fit_rows"] = len(final_fit_ids)
            row["final_outside_fit_rows"] = len(frame) - len(final_fit_ids)
            row["final_model_metrics"] = model_and_policy_metrics(
                model, X, frame, final_fit_ids, direct_labels, prop_targets)[0]
            row["final_fit_seconds"] = round(time.perf_counter() - round_started, 2)
            print(
                f"[delayed] final double-fit: included {len(new_only):,} final-round "
                "propagation labels.", flush=True,
            )
        round_rows.append(row)
        print(json.dumps({
            "method": method, "reviews": len(direct_labels),
            "new_prop_queued": len(new_only), "fit_rows_before_final": fit_rows,
            "P": row["stage_model_metrics"]["model_full_precision"],
            "R": row["stage_model_metrics"]["model_full_recall"],
        }, sort_keys=True), flush=True)
        round_started = time.perf_counter()
    return round_rows, review_rows, prop_targets, propagation_events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SHAP_ALL family timing comparison; defaults preserve canonical 10x50 protocol."
    )
    parser.add_argument("--batch-size", type=int, default=REVIEW_BATCH_SIZE,
                        help="direct reviews per fit/matching checkpoint (default: 50)")
    parser.add_argument("--review-budget", type=int, default=REVIEW_BATCH_SIZE * REVIEW_ROUNDS,
                        help="total direct review budget; must be divisible by batch size (default: 500)")
    parser.add_argument("--allow-exceptions", action="store_true",
                        help="allow partial seed membership; unmatched reviewed cases become non-propagating singletons")
    parser.add_argument("--groups-path", type=Path, default=RESULTS / "shap_all_groups.csv",
                        help="SHAP path policy table")
    parser.add_argument("--membership-path", type=Path, default=RESULTS / "shap_all_group_membership.csv",
                        help="seed ID to family mapping; partial mapping requires --allow-exceptions")
    parser.add_argument("--meta-path", type=Path, default=RESULTS / "shap_all_groups_meta.json",
                        help="family source metadata")
    parser.add_argument("--output-tag", default="",
                        help="optional suffix to keep alternate runs from overwriting canonical outputs")
    return parser


def write_checkpoint_report(stem: Path, methods: dict, protocol: dict) -> Path:
    lines = [
        "# SHAP_ALL family timing checkpoints",
        "",
        f"Seed {protocol['seed']}; {protocol['review_budget']} direct reviews; "
        f"batch size {protocol['batch_size']}; {protocol['review_rounds']} fit checkpoints.",
        f"Families: {protocol['families']['n_groups']}; assigned seeds: "
        f"{protocol['families']['assigned_cases']}; singleton exceptions: "
        f"{protocol['families']['exception_cases']}.",
        "",
        "P/R and confusion counts below are full-data model metrics at every checkpoint. "
        "Delayed arm final fit shown separately where applicable.",
        "",
        "| Arm | Checkpoint | Reviews | P | R | FP | FN | New prop. | Active prop. | Prop. accuracy | Fit rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def add_row(method: str, checkpoint: str, budget: int, metrics: dict,
                new_count, active_count, prop_accuracy, fit_rows) -> None:
        lines.append(
            f"| {method} | {checkpoint} | {budget} | "
            f"{metrics.get('model_full_precision', 0):.5f} | "
            f"{metrics.get('model_full_recall', 0):.5f} | "
            f"{metrics.get('model_full_fp', 0)} | {metrics.get('model_full_fn', 0)} | "
            f"{new_count if new_count is not None else 0} | {active_count} | "
            f"{'' if prop_accuracy is None else f'{prop_accuracy:.5f}'} | {fit_rows} |"
        )

    for method, result in methods.items():
        for row in result["rounds"]:
            metrics = row.get("model_metrics") or row.get("stage_model_metrics") or {}
            add_row(
                method, str(row["round"]), row["review_budget"], metrics,
                row.get("new_propagated", row.get("new_propagated_queued", 0)),
                row.get("active_propagated", row.get("active_propagation_queue", 0)),
                row.get("new_propagation_accuracy"), row.get("fit_rows", 0),
            )
            final_metrics = row.get("final_model_metrics")
            if final_metrics:
                add_row(method, f"{row['round']} final-fit", row["review_budget"],
                        final_metrics, 0, row.get("active_propagation_queue", 0), None,
                        row.get("final_fit_rows", row.get("fit_rows", 0)))
    report_path = stem.with_name(f"{stem.name}_trajectory.md")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> None:
    started = time.perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.review_budget <= 0:
        parser.error("--batch-size and --review-budget must be positive")
    if args.review_budget % args.batch_size:
        parser.error("--review-budget must be divisible by --batch-size")
    review_rounds = args.review_budget // args.batch_size
    defaults = {
        "groups": RESULTS / "shap_all_groups.csv",
        "membership": RESULTS / "shap_all_group_membership.csv",
        "meta": RESULTS / "shap_all_groups_meta.json",
    }
    custom_family_files = (
        args.groups_path.resolve() != defaults["groups"].resolve()
        or args.membership_path.resolve() != defaults["membership"].resolve()
        or args.meta_path.resolve() != defaults["meta"].resolve()
    )
    output_parts = []
    if args.allow_exceptions:
        output_parts.append("exceptions")
    if custom_family_files:
        output_parts.append("custom_families")
    if args.output_tag:
        tag = re.sub(r"[^A-Za-z0-9_-]+", "_", args.output_tag.strip()).strip("_")
        if not tag:
            parser.error("--output-tag must contain at least one letter or digit")
        output_parts.append(tag)
    output_suffix = "_" + "_".join(output_parts) if output_parts else ""

    print(
        f"[setup] Loading SHAP_ALL timing data; {review_rounds} checkpoints × "
        f"{args.batch_size} reviews...", flush=True
    )
    frame = load_initial()
    frame["posting_id"] = frame["posting_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    frame["pred_label"] = frame["pred_label"].astype(int)
    ids = frame["posting_id"].astype(str)
    X = feature_matrix(frame)
    X_pool = Pool(X)
    feature_names = list(X.columns)
    initial_model = CatBoostClassifier()
    stem = f"initial_seed_{SEED}"
    initial_model.load_model(str(ROOT_CACHE / f"{stem}.cbm"))
    initial_shap = np.load(ROOT_CACHE / f"{stem}_shap.npy")
    if initial_shap.shape != X.shape:
        raise ValueError(f"Cached initial SHAP shape {initial_shap.shape} != X shape {X.shape}")

    (seed_group_by_id, labels_by_group, paths_by_group, support_by_group,
     feature_pos, family_meta) = load_family_policies(
        frame, feature_names,
        args.groups_path, args.membership_path, args.meta_path,
        allow_exceptions=args.allow_exceptions,
    )
    if family_meta.get("shap_space") != "shap_all" or int(family_meta.get("n_dimensions", -1)) != X.shape[1]:
        raise ValueError("SHAP_ALL rule definitions do not match feature matrix.")
    true_by_id = dict(zip(ids, frame.label.astype(int)))
    for pid, gid in seed_group_by_id.items():
        if gid is not None and true_by_id[pid] != labels_by_group[gid]:
            raise ValueError(f"Seed family label mismatch for {pid}/{gid}")
    if not args.allow_exceptions:
        initial_masks = match_family_masks(initial_shap, set(paths_by_group), paths_by_group, feature_pos)
        mask_coverage = np.sum(np.stack(list(initial_masks.values()), axis=0), axis=0)
        if not np.all(mask_coverage == 1):
            raise ValueError("SHAP family paths do not form a disjoint, exhaustive partition.")

    uncertainty = 1.0 - 2.0 * np.abs(frame.pred_score.to_numpy(dtype=float) - 0.5)
    tie_break = np.random.RandomState(SEED).random_sample(len(frame))
    trusted_order = np.lexsort((tie_break, uncertainty))
    trusted_ids = set(ids.iloc[trusted_order[:round(TRUSTED_FRACTION * len(frame))]])
    initial_targets = dict(zip(ids, frame.pred_label.astype(int)))
    trusted_targets = {pid: initial_targets[pid] for pid in trusted_ids}
    print(
        f"[setup] {len(frame):,} cases, {X.shape[1]} SHAP_ALL dims, {len(paths_by_group)} families, "
        f"{len(trusted_ids):,} trusted rows; "
        f"{sum(gid is not None for gid in seed_group_by_id.values()):,} assigned, "
        f"{sum(gid is None for gid in seed_group_by_id.values()):,} singleton exceptions. "
        "Candidate matching applies path conditions to current model SHAP.",
        flush=True,
    )

    immediate_rounds, immediate_reviews, immediate_targets, immediate_events = run_immediate(
        frame, X, X_pool, initial_model, initial_shap, frame.pred_score.to_numpy(dtype=float),
        trusted_ids, trusted_targets, tie_break, seed_group_by_id,
        labels_by_group, paths_by_group, support_by_group, feature_pos,
        args.batch_size, review_rounds,
    )
    print("\n[method] Starting post-review / next-round propagation...", flush=True)
    delayed_rounds, delayed_reviews, delayed_targets, delayed_events = run_delayed(
        frame, X, X_pool, initial_model, frame.pred_score.to_numpy(dtype=float),
        trusted_ids, trusted_targets, tie_break, seed_group_by_id,
        labels_by_group, paths_by_group, support_by_group, feature_pos,
        args.batch_size, review_rounds,
    )

    methods = {
        "immediate_pre_review_shap": {
            "rounds": immediate_rounds, "reviews": immediate_reviews,
            "active_propagated_targets": immediate_targets, "propagation_events": immediate_events,
        },
        "post_review_shap_next_round": {
            "rounds": delayed_rounds, "reviews": delayed_reviews,
            "active_propagated_targets": delayed_targets, "propagation_events": delayed_events,
        },
    }
    output = {
        "protocol": {
            "name": f"SHAP_ALL Oracle-family propagation timing comparison ({review_rounds}x{args.batch_size})",
            "seed": SEED,
            "review_rounds": review_rounds,
            "batch_size": args.batch_size,
            "review_budget": args.review_budget,
            "query_source": "initial proposal for first batch; current best_model after each retrain",
            "family_seed_lookup": "posting_id -> Oracle SHAP_ALL family ID",
            "candidate_matching": "evaluate saved family path inequalities against each round's current model SHAP_ALL vectors",
            "candidate_id_membership_lookup_used_for_propagation": False,
            "fit_filter": "apply after SHAP matching; exclude trusted/direct/prior-propagated fit rows",
            "direct_weight": CORRECTED_WEIGHT,
            "propagated_weight": 1,
            "delayed_rule": "train direct + previous propagation; compute SHAP/match current seeds; add matches next retrain",
            "delayed_final_round": "stage fit on direct + previous propagation, then SHAP-match, then final fit including last propagation",
            "family_files": {
                "groups": str(args.groups_path),
                "membership": str(args.membership_path),
                "meta": str(args.meta_path),
                "exceptions_allowed": bool(args.allow_exceptions),
            },
            "families": {
                "n_groups": int(len(paths_by_group)),
                "cases": int(len(frame)),
                "assigned_cases": int(sum(gid is not None for gid in seed_group_by_id.values())),
                "exception_cases": int(sum(gid is None for gid in seed_group_by_id.values())),
                "oracle_setup_direct_reviews_not_charged": int(family_meta.get("oracle_reviewed_direct", 0)),
                "oracle_residual_corrections_not_charged": int(family_meta.get("oracle_residual_corrections", 0)),
            },
            "runtime_seconds": round(time.perf_counter() - started, 2),
        },
        "methods": methods,
    }
    outdir = RESULTS
    stem = outdir / f"split_training_ideal_shap_all_timing_{review_rounds}x{args.batch_size}{output_suffix}"
    (stem.with_suffix(".json")).write_text(json.dumps(output, indent=2), encoding="utf-8")
    round_rows = [
        {key: value for key, value in row.items() if not isinstance(value, dict)}
        | {f"{scope}_{metric}": metric_value
           for scope in ("model_metrics", "policy_metrics", "stage_model_metrics", "final_model_metrics")
           for metric, metric_value in (row.get(scope) or {}).items()}
        for data in methods.values() for row in data["rounds"]
    ]
    pd.DataFrame(round_rows).to_csv(f"{stem}_rounds.csv", index=False)
    pd.DataFrame([review for data in methods.values() for review in data["reviews"]]).to_csv(
        f"{stem}_reviews.csv", index=False
    )
    prop_rows = [event for data in methods.values() for event in data["propagation_events"]]
    pd.DataFrame(prop_rows).to_csv(f"{stem}_propagated_cases.csv", index=False)
    trajectory_report = write_checkpoint_report(stem, methods, output["protocol"])
    print(f"Saved: {stem}.json", flush=True)
    print(f"Saved: {stem}_rounds.csv", flush=True)
    print(f"Saved: {stem}_reviews.csv", flush=True)
    print(f"Saved: {stem}_propagated_cases.csv", flush=True)
    print(f"Saved: {trajectory_report}", flush=True)


if __name__ == "__main__":
    main()

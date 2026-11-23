"""Streaming, multi-run analysis for uncertainty-trace sidecars.

Example:
    python -m legacy.analyze_uncertainty_trace runs/traces/run_a.jsonl runs/traces/run_b.jsonl

Trace files can be large. Records are processed one line at a time; raw events
and cohort snapshots are never accumulated as full JSON objects.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _iter_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc


class Distribution:
    def __init__(self):
        self.values = []

    def add(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if np.isfinite(value):
            self.values.append(value)

    def report(self):
        if not self.values:
            return {"n": 0}
        arr = np.asarray(self.values, dtype=float)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "q10": float(np.quantile(arr, 0.10)),
            "q25": float(np.quantile(arr, 0.25)),
            "q75": float(np.quantile(arr, 0.75)),
            "q90": float(np.quantile(arr, 0.90)),
            "q95": float(np.quantile(arr, 0.95)),
            "max": float(arr.max()),
        }


class RunningMoments:
    """Small-memory aggregate for large cohort/control streams."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = None
        self.maximum = None

    def add(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(value):
            return
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def report(self):
        if self.n == 0:
            return {"n": 0}
        return {
            "n": self.n,
            "mean": self.mean,
            "std": float(np.sqrt(self.m2 / self.n)),
            "min": self.minimum,
            "max": self.maximum,
        }


def _safe_int(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_run_metadata(run_path):
    config = {}
    actual_retrains = set()
    final_iter = None
    run_summary = None
    if not run_path or not os.path.isfile(run_path):
        return config, actual_retrains, final_iter, run_summary

    for rec in _iter_jsonl(run_path):
        if rec.get("kind") == "run":
            config = rec.get("config") or {}
        elif rec.get("kind") == "iter":
            iteration = _safe_int(rec.get("iteration"))
            if rec.get("mode") in ("retrain", "incremental"):
                actual_retrains.add(iteration)
            if final_iter is None or iteration > _safe_int(final_iter.get("iteration")):
                final_iter = rec
        elif rec.get("kind") == "summary":
            run_summary = rec
    return config, actual_retrains, final_iter, run_summary


def _family_report(family_rows, min_family_size):
    family_csv_rows = []
    total_changed = 0
    total_wrong = 0
    total_correct = 0
    high_flagged = high_wrong = high_correct = 0
    low_flagged = low_wrong = low_correct = 0
    either_flagged = either_wrong = either_correct = 0

    for (retrain_iter, seed_id), rows in family_rows.items():
        scores = np.asarray([r["score"] for r in rows], dtype=float)
        q1, q3 = np.quantile(scores, [0.25, 0.75])
        iqr = float(q3 - q1)
        high_threshold = float(q3 + 1.5 * iqr)
        low_threshold = float(q1 - 1.5 * iqr)
        can_flag = len(rows) >= min_family_size
        high_flags = [bool(can_flag and r["score"] > high_threshold and r["score"] > 0)
                      for r in rows]
        low_flags = [bool(can_flag and r["score"] < low_threshold)
                     for r in rows]
        either_flags = [hi or lo for hi, lo in zip(high_flags, low_flags)]
        family_wrong = sum(r["group"] == "prop_changed_wrong" for r in rows)
        family_correct = sum(r["group"] == "prop_changed_correct" for r in rows)

        def flag_counts(flags):
            wrong = sum(flag and row["group"] == "prop_changed_wrong"
                        for row, flag in zip(rows, flags))
            correct = sum(flag and row["group"] == "prop_changed_correct"
                          for row, flag in zip(rows, flags))
            return sum(flags), wrong, correct

        high_counts = flag_counts(high_flags)
        low_counts = flag_counts(low_flags)
        either_counts = flag_counts(either_flags)
        total_changed += len(rows)
        total_wrong += family_wrong
        total_correct += family_correct
        high_flagged += high_counts[0]
        high_wrong += high_counts[1]
        high_correct += high_counts[2]
        low_flagged += low_counts[0]
        low_wrong += low_counts[1]
        low_correct += low_counts[2]
        either_flagged += either_counts[0]
        either_wrong += either_counts[1]
        either_correct += either_counts[2]

        family_csv_rows.append({
            "retrain_iteration": retrain_iter,
            "seed_id": seed_id,
            "candidate_count": len(rows),
            "family_size": max((r["family_size"] for r in rows), default=0),
            "changed_correct": family_correct,
            "changed_wrong": family_wrong,
            "propagation_accuracy": (
                family_correct / (family_correct + family_wrong)
                if family_correct + family_wrong else None),
            "score_median": float(np.median(scores)),
            "score_q90": float(np.quantile(scores, 0.90)),
            "score_max": float(scores.max()),
            "high_threshold": high_threshold,
            "low_threshold": low_threshold,
            "threshold_applied": can_flag,
            "high_flagged": high_counts[0],
            "high_wrong": high_counts[1],
            "high_correct": high_counts[2],
            "low_flagged": low_counts[0],
            "low_wrong": low_counts[1],
            "low_correct": low_counts[2],
            "either_flagged": either_counts[0],
            "either_wrong": either_counts[1],
            "either_correct": either_counts[2],
        })

    def flag_report(flagged_n, wrong_n, correct_n):
        return {
            "flagged_count": flagged_n,
            "flagged_wrong": wrong_n,
            "flagged_correct": correct_n,
            "wrong_capture_rate": wrong_n / total_wrong if total_wrong else None,
            "flag_precision": wrong_n / flagged_n if flagged_n else None,
            "correct_flag_rate": correct_n / total_correct if total_correct else None,
        }

    return family_csv_rows, {
        "high_tail_rule": "per-seed Q3 + 1.5*IQR; score > 0; exploratory only",
        "low_tail_rule": "per-seed Q1 - 1.5*IQR; opposite-direction diagnostic",
        "min_family_size": min_family_size,
        "candidate_count": total_changed,
        "wrong_count": total_wrong,
        "correct_count": total_correct,
        "high_tail": flag_report(high_flagged, high_wrong, high_correct),
        "low_tail": flag_report(low_flagged, low_wrong, low_correct),
        "either_tail": flag_report(either_flagged, either_wrong, either_correct),
    }


def analyze_trace(trace_path, runs_dir="runs", run_override=None,
                  min_family_size=5):
    stem = os.path.splitext(os.path.basename(trace_path))[0]
    run_path = run_override or os.path.join(runs_dir, f"{stem}.jsonl")
    config, actual_retrains, final_iter, run_summary = _load_run_metadata(run_path)

    header = {}
    trace_summary = None
    trace_retrains = set()
    resolution_retrains = set()
    seed_retrains = set()
    invalid_resolution_retrains = set()
    invalid_seed_retrains = set()
    unresolved_events = 0
    unresolved_seeds = 0
    resolution_count = 0
    seed_resolution_count = 0
    cohort_count = 0
    control_sample_count = 0

    resolution_groups = defaultdict(Distribution)
    adjusted_groups = defaultdict(Distribution)
    seed_relative_groups = defaultdict(Distribution)
    control_sample_groups = defaultdict(RunningMoments)
    control_by_decile = defaultdict(RunningMoments)
    cohort_by_retrain_group = defaultdict(RunningMoments)
    cohort_trajectories = {}
    family_rows = defaultdict(list)
    y_true_wrong = []
    y_score_seed_relative = []
    y_score_delta_u = []
    u_event_gaps = Distribution()
    retrain_summaries = {}

    for rec in _iter_jsonl(trace_path):
        kind = rec.get("kind")
        if kind == "trace_header":
            header = rec
        elif kind == "trace_summary":
            trace_summary = rec
        elif kind == "retrain_summary":
            iteration = _safe_int(rec.get("retrain_iteration"))
            trace_retrains.add(iteration)
            retrain_summaries[iteration] = rec
        elif kind == "event_resolution":
            resolution_count += 1
            iteration = _safe_int(rec.get("retrain_iteration"))
            resolution_retrains.add(iteration)
            if actual_retrains and iteration not in actual_retrains:
                invalid_resolution_retrains.add(iteration)
            group = rec.get("group", "unknown")
            delta_u = rec.get("delta_u")
            resolution_groups[group].add(delta_u)
            gap = rec.get("u_event_gap")
            if gap is not None:
                u_event_gaps.add(abs(gap))

            control_mean = rec.get("control_delta_mean_decile")
            if control_mean is not None and delta_u is not None:
                adjusted_groups[group].add(float(delta_u) - float(control_mean))

            score = rec.get("seed_relative_score")
            if (rec.get("role") == "candidate"
                    and group in ("prop_changed_correct", "prop_changed_wrong")
                    and delta_u is not None):
                y_score_delta_u.append((
                    int(group == "prop_changed_wrong"), float(delta_u),
                    (float(delta_u) - float(control_mean))
                    if control_mean is not None else None,
                ))
            if score is None:
                seed_delta = rec.get("seed_delta_u")
                seed_control = rec.get("seed_control_delta_mean_decile")
                if (seed_delta is not None and seed_control is not None
                        and control_mean is not None and delta_u is not None):
                    score = ((float(delta_u) - float(control_mean))
                             - (float(seed_delta) - float(seed_control)))
            if rec.get("role") == "candidate" and score is not None:
                score = float(score)
                seed_relative_groups[group].add(score)
                if group in ("prop_changed_correct", "prop_changed_wrong"):
                    y_true_wrong.append(int(group == "prop_changed_wrong"))
                    y_score_seed_relative.append(score)
                    family_key = (iteration, str(rec.get("seed_id")))
                    family_rows[family_key].append({
                        "score": score,
                        "group": group,
                        "posting_id": rec.get("posting_id"),
                        "family_size": _safe_int(rec.get("family_size"), 0),
                        "delta_assigned": rec.get("delta_assigned"),
                        "iteration": _safe_int(rec.get("iteration")),
                        "delta_u": float(delta_u),
                        "control_adjusted_delta_u": (
                            float(delta_u) - float(control_mean)
                            if control_mean is not None else None),
                    })
        elif kind == "seed_resolution":
            seed_resolution_count += 1
            iteration = _safe_int(rec.get("retrain_iteration"))
            seed_retrains.add(iteration)
            if actual_retrains and iteration not in actual_retrains:
                invalid_seed_retrains.add(iteration)
        elif kind == "control_sample":
            control_sample_count += 1
            group = rec.get("group", "control_never")
            control_sample_groups[group].add(rec.get("delta_u"))
            key = (int(rec.get("retrain_iteration", -1)),
                   int(rec.get("u_decile_before", -1)))
            control_by_decile[key].add(rec.get("delta_u"))
        elif kind == "cohort_retrain":
            cohort_count += 1
            iteration = _safe_int(rec.get("retrain_iteration"))
            cohort = str(rec.get("cohort", "unknown"))
            latest = str(rec.get("latest_outcome", "unknown"))
            group_key = (iteration, cohort, latest)
            cohort_by_retrain_group[group_key].add(rec.get("delta_u"))

            pid = str(rec.get("posting_id"))
            trajectory = cohort_trajectories.get(pid)
            if trajectory is None:
                cohort_trajectories[pid] = {
                    "cohort": cohort,
                    "ever_wrong": bool(rec.get("ever_wrong")),
                    "first_retrain": iteration,
                    "last_retrain": iteration,
                    "first_delta_u": float(rec.get("delta_u", 0.0)),
                    "last_delta_u": float(rec.get("delta_u", 0.0)),
                    "first_is_correct": bool(rec.get("is_correct_after")),
                    "last_is_correct": bool(rec.get("is_correct_after")),
                    "n_retrains": 1,
                }
            else:
                trajectory["last_retrain"] = iteration
                trajectory["last_delta_u"] = float(rec.get("delta_u", 0.0))
                trajectory["last_is_correct"] = bool(rec.get("is_correct_after"))
                trajectory["ever_wrong"] = bool(
                    trajectory["ever_wrong"] or rec.get("ever_wrong"))
                trajectory["n_retrains"] += 1
        elif kind == "event_unresolved":
            unresolved_events += 1
        elif kind == "seed_unresolved":
            unresolved_seeds += 1

    invalid_retrains = sorted(invalid_resolution_retrains | invalid_seed_retrains)
    missing_trace_retrains = sorted(actual_retrains - trace_retrains)
    extra_trace_retrains = sorted(trace_retrains - actual_retrains) if actual_retrains else []
    gap_stats = u_event_gaps.report()
    gap_bad = sum(
        1 for value in u_event_gaps.values if value > 1e-8
    )

    wrong = np.asarray(y_true_wrong, dtype=int)
    scores = np.asarray(y_score_seed_relative, dtype=float)
    separation = {
        "n_changed_candidates": int(wrong.size),
        "wrong_prevalence": float(wrong.mean()) if wrong.size else None,
        "roc_auc": None,
        "roc_auc_reversed": None,
        "average_precision": None,
        "average_precision_reversed": None,
    }
    if wrong.size and len(np.unique(wrong)) == 2:
        separation["roc_auc"] = float(roc_auc_score(wrong, scores))
        separation["roc_auc_reversed"] = float(roc_auc_score(wrong, -scores))
        separation["average_precision"] = float(average_precision_score(wrong, scores))
        separation["average_precision_reversed"] = float(
            average_precision_score(wrong, -scores))

    delta_rows = [row for row in y_score_delta_u if row[2] is not None]

    def _separation_report(rows, score_index):
        labels = np.asarray([row[0] for row in rows], dtype=int)
        values = np.asarray([row[score_index] for row in rows], dtype=float)
        report = {
            "n_changed_candidates": int(labels.size),
            "wrong_prevalence": float(labels.mean()) if labels.size else None,
            "roc_auc": None,
            "roc_auc_reversed": None,
            "average_precision": None,
            "average_precision_reversed": None,
        }
        if labels.size and len(np.unique(labels)) == 2:
            report["roc_auc"] = float(roc_auc_score(labels, values))
            report["roc_auc_reversed"] = float(roc_auc_score(labels, -values))
            report["average_precision"] = float(
                average_precision_score(labels, values))
            report["average_precision_reversed"] = float(
                average_precision_score(labels, -values))
        return report

    candidate_delta_separation = _separation_report(y_score_delta_u, 1)
    adjusted_delta_separation = _separation_report(delta_rows, 2)

    family_csv_rows, reversion_probe = _family_report(family_rows, min_family_size)

    cohort_by_group = defaultdict(list)
    for trajectory in cohort_trajectories.values():
        cohort_by_group[(trajectory["cohort"], trajectory["ever_wrong"])].append(trajectory)
    cohort_trajectory_report = {}
    for (cohort, ever_wrong), trajectories in cohort_by_group.items():
        first_correct = sum(t["first_is_correct"] for t in trajectories)
        last_correct = sum(t["last_is_correct"] for t in trajectories)
        cohort_trajectory_report[f"{cohort}|ever_wrong={ever_wrong}"] = {
            "n_cases": len(trajectories),
            "first_delta_u": Distribution(),
            "last_delta_u": Distribution(),
            "delta_u_change": Distribution(),
            "first_correct_fraction": first_correct / len(trajectories),
            "last_correct_fraction": last_correct / len(trajectories),
            "mean_retrains_per_case": float(np.mean([t["n_retrains"] for t in trajectories])),
        }
        target = cohort_trajectory_report[f"{cohort}|ever_wrong={ever_wrong}"]
        for trajectory in trajectories:
            target["first_delta_u"].add(trajectory["first_delta_u"])
            target["last_delta_u"].add(trajectory["last_delta_u"])
            target["delta_u_change"].add(
                trajectory["last_delta_u"] - trajectory["first_delta_u"])
        for field in ("first_delta_u", "last_delta_u", "delta_u_change"):
            target[field] = target[field].report()

    cohort_retrain_report = {}
    for (iteration, cohort, latest), stats in cohort_by_retrain_group.items():
        cohort_retrain_report.setdefault(str(iteration), {})[
            f"{cohort}|{latest}"] = stats.report()

    family_csv_rows = [
        {"run_id": header.get("run_id", stem), **row}
        for row in family_csv_rows
    ]
    valid = bool(run_path and os.path.isfile(run_path))
    valid = (valid and not invalid_retrains and not missing_trace_retrains
             and not extra_trace_retrains)

    result = {
        "run_id": header.get("run_id", stem),
        "trace_path": trace_path,
        "run_path": run_path if os.path.isfile(run_path) else None,
        "trace_header": header,
        "trace_summary": trace_summary,
        "config": config,
        "complete": run_summary is not None and trace_summary is not None,
        "valid_boundary_alignment": valid,
        "actual_retrain_count": len(actual_retrains),
        "trace_retrain_count": len(trace_retrains),
        "invalid_retrain_iterations": invalid_retrains,
        "missing_trace_retrains": missing_trace_retrains,
        "extra_trace_retrains": extra_trace_retrains,
        "event_resolutions": resolution_count,
        "seed_resolutions": seed_resolution_count,
        "unresolved_events": unresolved_events,
        "unresolved_seeds": unresolved_seeds,
        "u_event_gap_abs": gap_stats,
        "u_event_gap_over_1e-8": gap_bad,
        "cohort_rows_streamed": cohort_count,
        "cohort_cases": len(cohort_trajectories),
        "cohort_trajectories": cohort_trajectory_report,
        "cohort_by_retrain_and_group": cohort_retrain_report,
        "control_sample_rows": control_sample_count,
        "control_sample_delta_u": {
            group: stats.report() for group, stats in control_sample_groups.items()
        },
        "full_control_delta_u_by_retrain_decile": {
            str(iteration): summary.get("control_by_decile", {})
            for iteration, summary in sorted(retrain_summaries.items())
        },
        "control_sample_delta_u_by_retrain_decile": {
            f"{iteration}|{decile}": stats.report()
            for (iteration, decile), stats in sorted(control_by_decile.items())
        },
        "resolution_delta_u_by_group": {
            group: stats.report() for group, stats in resolution_groups.items()
        },
        "control_adjusted_delta_u_by_group": {
            group: stats.report() for group, stats in adjusted_groups.items()
        },
        "seed_relative_score_by_group": {
            group: stats.report() for group, stats in seed_relative_groups.items()
        },
        "wrong_vs_correct_separation": separation,
        "candidate_delta_u_separation": candidate_delta_separation,
        "control_adjusted_delta_u_separation": adjusted_delta_separation,
        "exploratory_reversion_probe": reversion_probe,
        "final_run_metrics": {
            "iteration": _safe_int((final_iter or {}).get("iteration"), 0),
            "mode": (final_iter or {}).get("mode"),
            "precision": (final_iter or {}).get("precision"),
            "recall": (final_iter or {}).get("recall"),
            "tp": (final_iter or {}).get("tp"),
            "fp": (final_iter or {}).get("fp"),
            "tn": (final_iter or {}).get("tn"),
            "fn": (final_iter or {}).get("fn"),
        },
        "run_summary": run_summary,
    }
    return result, family_csv_rows


def _comparison_row(result):
    groups = result["resolution_delta_u_by_group"]
    scores = result["seed_relative_score_by_group"]
    metrics = result["final_run_metrics"]
    separation = result["wrong_vs_correct_separation"]
    delta_separation = result["candidate_delta_u_separation"]
    adjusted_separation = result["control_adjusted_delta_u_separation"]
    probe = result["exploratory_reversion_probe"]
    high_probe = probe["high_tail"]
    low_probe = probe["low_tail"]
    return {
        "run_id": result["run_id"],
        "propagation_mode": result["config"].get("propagation_mode"),
        "retrain_every": result["config"].get("retrain_every"),
        "seed": result["config"].get("seed"),
        "valid_boundary_alignment": result["valid_boundary_alignment"],
        "actual_retrains": result["actual_retrain_count"],
        "changed_correct_n": groups.get("prop_changed_correct", {}).get("n", 0),
        "changed_correct_delta_median": groups.get("prop_changed_correct", {}).get("median"),
        "changed_correct_score_median": scores.get("prop_changed_correct", {}).get("median"),
        "changed_wrong_n": groups.get("prop_changed_wrong", {}).get("n", 0),
        "changed_wrong_delta_median": groups.get("prop_changed_wrong", {}).get("median"),
        "changed_wrong_score_median": scores.get("prop_changed_wrong", {}).get("median"),
        "roc_auc": separation.get("roc_auc"),
        "roc_auc_reversed": separation.get("roc_auc_reversed"),
        "average_precision": separation.get("average_precision"),
        "average_precision_reversed": separation.get("average_precision_reversed"),
        "delta_u_roc_auc": delta_separation.get("roc_auc"),
        "delta_u_roc_auc_reversed": delta_separation.get("roc_auc_reversed"),
        "delta_u_average_precision": delta_separation.get("average_precision"),
        "delta_u_average_precision_reversed": delta_separation.get("average_precision_reversed"),
        "control_adjusted_delta_u_roc_auc": adjusted_separation.get("roc_auc"),
        "control_adjusted_delta_u_average_precision": adjusted_separation.get("average_precision"),
        "high_tail_flagged_n": high_probe.get("flagged_count"),
        "high_tail_wrong": high_probe.get("flagged_wrong"),
        "high_tail_correct": high_probe.get("flagged_correct"),
        "high_tail_wrong_capture": high_probe.get("wrong_capture_rate"),
        "high_tail_precision": high_probe.get("flag_precision"),
        "low_tail_flagged_n": low_probe.get("flagged_count"),
        "low_tail_wrong": low_probe.get("flagged_wrong"),
        "low_tail_correct": low_probe.get("flagged_correct"),
        "low_tail_wrong_capture": low_probe.get("wrong_capture_rate"),
        "low_tail_precision": low_probe.get("flag_precision"),
        "precision_final": metrics.get("precision"),
        "recall_final": metrics.get("recall"),
        "fp_final": metrics.get("fp"),
        "fn_final": metrics.get("fn"),
        "tp_final": metrics.get("tp"),
        "tn_final": metrics.get("tn"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stream and compare one or more uncertainty-trace JSONL files.")
    parser.add_argument("traces", nargs="+", help="explicit trace JSONL paths")
    parser.add_argument("--runs-dir", default="runs",
                        help="directory containing corresponding main run JSONLs")
    parser.add_argument("--out-dir", default="runs/traces/analysis",
                        help="output directory; timestamped reports written here")
    parser.add_argument("--min-family-size", type=int, default=5,
                        help="minimum changed candidates for exploratory IQR probe")
    args = parser.parse_args()

    results = []
    family_rows = []
    for path in args.traces:
        print(f"Analyzing {path} ...")
        result, families = analyze_trace(
            path, runs_dir=args.runs_dir,
            min_family_size=args.min_family_size)
        results.append(result)
        family_rows.extend(families)

    comparison = [_comparison_row(result) for result in results]
    out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method_notes": {
            "seed_relative_score": "(case Δu - control median for case decile) - (seed Δu - control median for seed decile)",
            "wrong_vs_correct_separation": "ROC-AUC/AP use seed_relative_score; candidate and control-adjusted Δu scores reported separately.",
            "exploratory_outlier_probe": "Per-seed upper and lower IQR fences; offline diagnostics only; no policy applied.",
            "cohort": "Per-case cohort rows streamed and aggregated; raw rows not retained.",
        },
        "runs": results,
        "comparison": comparison,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.out_dir, f"uncertainty_comparison_{stamp}.json")
    csv_path = os.path.join(args.out_dir, f"uncertainty_comparison_{stamp}.csv")
    family_path = os.path.join(args.out_dir, f"uncertainty_families_{stamp}.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, default=str)

    comparison_fields = list(comparison[0]) if comparison else []
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        if comparison_fields:
            writer.writeheader()
            writer.writerows(comparison)

    family_fields = [
        "run_id", "retrain_iteration", "seed_id", "candidate_count",
        "family_size", "changed_correct", "changed_wrong",
        "propagation_accuracy", "score_median", "score_q90", "score_max",
        "high_threshold", "low_threshold", "threshold_applied",
        "high_flagged", "high_wrong", "high_correct", "low_flagged",
        "low_wrong", "low_correct", "either_flagged", "either_wrong",
        "either_correct",
    ]
    with open(family_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=family_fields)
        writer.writeheader()
        writer.writerows(family_rows)

    print("\nRun comparison:")
    for row in comparison:
        print(
            f"{str(row['propagation_mode']):10} every={str(row['retrain_every']):>2} "
            f"wrong={row['changed_wrong_n']:>5} correct={row['changed_correct_n']:>6} "
            f"relAUC(+/-)={row['roc_auc']}/{row['roc_auc_reversed']} "
            f"rawAUC={row['delta_u_roc_auc']} "
            f"adjAUC={row['control_adjusted_delta_u_roc_auc']} "
            f"FP={row['fp_final']} FN={row['fn_final']} "
            f"valid={row['valid_boundary_alignment']}"
        )
    print("\nWrote:")
    print(" ", json_path)
    print(" ", csv_path)
    print(" ", family_path)


if __name__ == "__main__":
    main()

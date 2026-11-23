"""Direct-review-only experiment with SHAP-guided CatBoost feature weights.

No label propagation. CatBoost feature weights are global per fit; reviewed-case
SHAP vectors are aggregated into one feature-weight vector before each retrain.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "legacy" / "experiments"))

from legacy.experiments.shap_family_propagation import (  # noqa: E402
    CORRECTED_WEIGHT,
    TRUSTED_FRACTION,
    feature_matrix,
)
from legacy.experiments.split_training_feature_value_families import _initial_cache  # noqa: E402
from legacy.experiments.split_training_shap_box_timing import (  # noqa: E402
    model_metrics,
    run_arm,
)


RESULTS = ROOT / "legacy" / "experiments" / "results"
STEM = RESULTS / "split_training_shap_feature_weights"
SCHEDULES = {"10x50": (10, 50), "20x25": (20, 25)}
WEIGHT_MODES = (
    "uniform",
    "static_initial_shap",
    "reviewed_local_shap",
    "reviewed_error_shap",
)
DEFAULT_SEEDS = (42,)
DEFAULT_STRENGTH = 0.5


def check_feature_weight_api(frame: pd.DataFrame, seed: int) -> None:
    """Fail early if installed CatBoost rejects native feature_weights."""
    X = feature_matrix(frame)
    labels = frame["pred_label"].to_numpy(dtype=int)
    positive = np.flatnonzero(labels == 1)[:128]
    negative = np.flatnonzero(labels == 0)[:128]
    rows = np.concatenate((negative, positive))
    y = labels[rows]
    if np.unique(y).size != 2:
        raise ValueError("Feature-weight API smoke test needs both pseudo-label classes")
    probe = CatBoostClassifier(
        iterations=2,
        depth=2,
        learning_rate=0.1,
        loss_function="Logloss",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        feature_weights=[1.0] * X.shape[1],
    )
    probe.fit(Pool(X.iloc[rows], label=y))


def save_progress(results: dict, protocol: dict, stem: Path) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    compact = {
        key: {
            "rounds": result["rounds"],
            "runtime_seconds": result["runtime_seconds"],
            "direct_review_count": len(result["direct_reviews"]),
        }
        for key, result in results.items()
    }
    (stem.with_suffix(".json")).write_text(json.dumps({
        "protocol": protocol,
        "completed_trajectories": list(results),
        "results": compact,
    }, indent=2), encoding="utf-8")
    checkpoint_rows = []
    review_rows = []
    weight_rows = []
    for key, result in results.items():
        seed, schedule, mode = key.split("|")
        checkpoint_rows.extend(
            {"seed": int(seed), "schedule": schedule, **row, "weight_policy": mode}
            for row in result["rounds"]
        )
        review_rows.extend(
            {"seed": int(seed), "schedule": schedule, **row, "weight_policy": mode}
            for row in result["direct_reviews"]
        )
        weight_rows.extend(
            {"seed": int(seed), "schedule": schedule, **row}
            for row in result["feature_weights"]
        )
    pd.DataFrame(checkpoint_rows).to_csv(f"{stem}_checkpoints.csv", index=False)
    pd.DataFrame(review_rows).to_csv(f"{stem}_direct_reviews.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(f"{stem}_feature_weights.csv", index=False)
    lines = [
        "# SHAP-guided CatBoost feature-weight experiment",
        "",
        "Direct-review-only trajectories; propagation disabled. CatBoost receives one",
        "global feature-weight vector per fit. SHAP values from reviewed cases are",
        "aggregated; no per-case feature-weighting is claimed.",
        "",
        f"Strength: {protocol['strength']}; review budget: {protocol['reviews']};",
        f"seeds: {protocol['seeds']}; schedules: {', '.join(protocol['schedules'])}.",
        "",
        "## Final checkpoint by trajectory",
        "",
        "| Seed | Schedule | Policy | Reviews | Full P | Full R | Full F1 | Non-fit P | Non-fit R | Non-fit F1 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, result in results.items():
        seed, schedule, mode = key.split("|")
        row = result["rounds"][-1]
        metrics = row.get("final_model_metrics") or row["model_metrics"]
        def value(name):
            item = metrics.get(name)
            return "—" if item is None else f"{item:.5f}"
        lines.append(
            f"| {seed} | {schedule} | {mode} | {row['review_budget']} | "
            f"{value('model_full_precision')} | {value('model_full_recall')} | {value('model_full_f1')} | "
            f"{value('model_nonfit_precision')} | {value('model_nonfit_recall')} | {value('model_nonfit_f1')} |"
        )
    lines.extend([
        "", "## First checkpoint reaching full-data P/R=1/1", "",
        "| Seed | Schedule | Policy | Reviews |",
        "|---:|---|---|---:|",
    ])
    for key, result in results.items():
        seed, schedule, mode = key.split("|")
        first = next((row for row in result["rounds"]
                      if row["model_metrics"].get("model_full_precision") == 1.0
                      and row["model_metrics"].get("model_full_recall") == 1.0), None)
        reviews = "not reached" if first is None else str(first["review_budget"])
        lines.append(f"| {seed} | {schedule} | {mode} | {reviews} |")
    lines.extend([
        "", "## Checkpoint data", "",
        f"All full/non-fit checkpoint metrics: `{stem.name}_checkpoints.csv`.",
        f"Per-feature vectors: `{stem.name}_feature_weights.csv`.",
        f"Direct review sequences: `{stem.name}_direct_reviews.csv`.",
        "Existing direct-only and nearest-SHAP top-four curves are external references, not rerun here.",
        "",
    ])
    Path(f"{stem}_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42", help="Comma-separated run seeds")
    parser.add_argument("--schedules", default="10x50,20x25")
    parser.add_argument("--modes", default=",".join(WEIGHT_MODES))
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH,
                        help="Exponent on normalized mean-absolute-SHAP importance")
    parser.add_argument("--tag", default="screen42", help="Artifact tag to avoid overwriting prior runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    schedules = tuple(value for value in args.schedules.split(",") if value)
    modes = tuple(value for value in args.modes.split(",") if value)
    if not seeds or any(value not in SCHEDULES for value in schedules):
        raise ValueError("Provide valid seeds and schedules (10x50,20x25)")
    if not modes or any(value not in WEIGHT_MODES for value in modes):
        raise ValueError(f"Modes must be selected from {WEIGHT_MODES}")
    if not 0.0 <= args.strength <= 1.0:
        raise ValueError("--strength must be in [0, 1]")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = RESULTS / f"{STEM.name}_{args.tag}"
    started = time.perf_counter()
    results = {}
    protocol = {
        "purpose": "Test whether global SHAP-guided CatBoost feature weights recover part of feedback benefit without propagation",
        "propagation": "disabled; direct reviews only",
        "weight_mechanism": "CatBoost native global feature_weights, not sample weights or scaled input values",
        "weight_formula": "clip((mean_abs_shap / mean_feature_importance) ** strength, 0.25, 4.0)",
        "strength": args.strength,
        "modes": list(modes),
        "seeds": list(seeds),
        "schedules": list(schedules),
        "reviews": 500,
        "direct_review_weight": CORRECTED_WEIGHT,
        "trusted_fraction": TRUSTED_FRACTION,
        "metrics": "full and non-fit P/R, F1, AUC, confusion counts at every fit checkpoint",
        "dynamic_local_source": "current-model SHAP on directly reviewed cases only; error mode uses cases mispredicted before review (fallback to all reviewed cases)",
        "limitations": "CatBoost feature_weights are global per model fit; per-case values are aggregated, not individually applied",
        "propagation_reference": "Existing nearest-SHAP top-four delayed report; reference only, not rerun in this runner",
    }

    for seed in seeds:
        print(f"[init] seed={seed}", flush=True)
        frame, initial_model = _initial_cache(seed)
        frame = frame.reset_index(drop=True)
        frame["posting_id"] = frame["posting_id"].astype(str)
        frame["label"] = frame["label"].astype(int)
        frame["pred_label"] = frame["pred_label"].astype(int)
        X = feature_matrix(frame)
        pool = Pool(X)
        check_feature_weight_api(frame, seed)
        initial_shap = np.asarray(
            initial_model.get_feature_importance(pool, type="ShapValues")[:, :-1],
            dtype=float,
        )
        if initial_shap.shape != X.shape or not np.isfinite(initial_shap).all():
            raise ValueError(f"Seed {seed}: invalid initial SHAP shape/values {initial_shap.shape}")
        initial_score = frame["pred_score"].to_numpy(dtype=float)
        ids = frame["posting_id"]
        tie_break = np.random.RandomState(seed).random_sample(len(frame))
        uncertainty = 1.0 - 2.0 * np.abs(initial_score - 0.5)
        trusted_order = np.lexsort((tie_break, uncertainty))
        trusted_ids = set(ids.iloc[trusted_order[:round(TRUSTED_FRACTION * len(frame))]])
        initial_targets = dict(zip(ids, frame["pred_label"].to_numpy(dtype=int)))
        trusted_targets = {pid: initial_targets[pid] for pid in trusted_ids}

        for schedule in schedules:
            rounds, batch_size = SCHEDULES[schedule]
            for mode in modes:
                key = f"{seed}|{schedule}|{mode}"
                print(f"[run] {key}; strength={args.strength}", flush=True)
                result = run_arm(
                    "direct_only", "post_review_shap_next_round", batch_size, rounds,
                    frame, X, pool, initial_model, initial_shap, initial_score,
                    trusted_ids, trusted_targets, tie_break,
                    propagation_enabled=False,
                    print_nonfit_metrics=True,
                    feature_weight_mode=mode,
                    feature_weight_strength=args.strength,
                    seed=seed,
                )
                results[key] = result
                save_progress(results, protocol, stem)
                print(f"[saved] completed {key}", flush=True)

    print(f"Complete in {time.perf_counter() - started:.1f}s; artifacts: {stem}*", flush=True)


if __name__ == "__main__":
    main()

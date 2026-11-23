"""Delayed 10x50 / 20x25 SHAP-box δ=.10 vs direct-only controls."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from legacy.experiments.shap_family_propagation import (
    REVIEW_BATCH_SIZE,
    REVIEW_ROUNDS,
    ROOT_CACHE,
    SEED,
    TRUSTED_FRACTION,
    feature_matrix,
    load_initial,
)
from legacy.experiments.split_training_shap_box_timing import (
    RESULTS,
    flatten_metrics,
    run_arm,
    write_report,
    FLAG_PREFIXES,
)


FIXED_DELTA = 0.10
REVIEW_BUDGET = REVIEW_BATCH_SIZE * REVIEW_ROUNDS
SCHEDULES = ((10, 50), (20, 25))
TIMING = "post_review_shap_next_round"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
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
    if initial_shap.shape != X.shape or not np.isfinite(initial_shap).all():
        raise ValueError(f"Cached SHAP shape/values invalid for feature matrix {X.shape}")
    initial_score = frame["pred_score"].to_numpy(dtype=float)

    uncertainty = 1.0 - 2.0 * np.abs(initial_score - 0.5)
    tie_break = np.random.RandomState(SEED).random_sample(len(frame))
    trusted_order = np.lexsort((tie_break, uncertainty))
    trusted_ids = set(ids.iloc[trusted_order[:round(TRUSTED_FRACTION * len(frame))]])
    initial_targets = dict(zip(ids, frame["pred_label"].to_numpy(dtype=int)))
    trusted_targets = {pid: initial_targets[pid] for pid in trusted_ids}

    protocol = {
        "seed": SEED,
        "review_budget": REVIEW_BUDGET,
        "trusted_fraction": TRUSTED_FRACTION,
        "timing_arm": TIMING,
        "shap_space": "shap_raw",
        "shap_raw_features": [name for name in X.columns if not name.startswith(FLAG_PREFIXES)],
        "fixed_shap_box_delta": FIXED_DELTA,
        "direct_only_control": "same query/review/retrain protocol; propagation disabled; SHAP not recomputed after fit",
        "schedules": [{"rounds": rounds, "batch_size": size} for rounds, size in SCHEDULES],
        "propagation": "live seed shap_box; first matching seed wins; fit exclusion after match",
        "direct_weight": 100.0,
        "propagated_weight": 1,
        "metrics": "full data and non-fit P/R, FP/FN, propagated counts, fit rows, each retrain",
        "oracle_role": "direct review labels only; candidate truth used only for offline propagation-accuracy reporting",
    }
    results = {}
    stem = RESULTS / "split_training_shap_box_delta010_delayed"

    for rounds, batch_size in SCHEDULES:
        schedule = f"{rounds}x{batch_size}"
        for mode, enabled in (("shap_box", True), ("direct_only", False)):
            key = f"{schedule}|{mode}|{TIMING}"
            print(f"\n[run] {key}; delta={FIXED_DELTA if enabled else 'n/a'}", flush=True)
            results[key] = run_arm(
                mode, TIMING, batch_size, rounds, frame, X, pool,
                initial_model, initial_shap, initial_score,
                trusted_ids, trusted_targets, tie_break,
                fixed_delta=FIXED_DELTA,
                propagation_enabled=enabled,
            )

            compact_results = {
                name: {
                    "rounds": data["rounds"],
                    "runtime_seconds": data["runtime_seconds"],
                    "direct_review_count": len(data["direct_reviews"]),
                    "seed_rule_count": len(data["reviews"]),
                    "propagation_event_count": len(data["propagation_events"]),
                }
                for name, data in results.items()
            }
            payload = {
                "protocol": protocol,
                "completed_trajectories": list(results),
                "results": compact_results,
                "elapsed_seconds": round(time.perf_counter() - started, 2),
            }
            stem.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

            checkpoint_rows = [
                {**flatten_metrics(row), "schedule": name.split("|")[0],
                 "mode": name.split("|")[1], "timing": name.split("|")[2]}
                for name, data in results.items() for row in data["rounds"]
            ]
            pd.DataFrame(checkpoint_rows).to_csv(f"{stem}_checkpoints.csv", index=False)
            seed_rows = [row for data in results.values() for row in data["reviews"]]
            pd.DataFrame(seed_rows).to_csv(f"{stem}_seed_rules.csv", index=False)
            direct_rows = [row for data in results.values() for row in data["direct_reviews"]]
            pd.DataFrame(direct_rows).to_csv(f"{stem}_direct_reviews.csv", index=False)
            events = [row for data in results.values() for row in data["propagation_events"]]
            pd.DataFrame(events).to_csv(f"{stem}_propagated.csv", index=False)
            write_report(Path(f"{stem}_report.md"), protocol, results)
            print(f"[saved] {stem} artifacts", flush=True)

    print(f"Complete in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()

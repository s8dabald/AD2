"""Paired multi-seed confirmation of delayed nearest-SHAP top-four vs direct-only."""

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
    TRUSTED_FRACTION,
    feature_matrix,
)
from legacy.experiments.split_training_feature_value_families import _initial_cache  # noqa: E402
from legacy.experiments.split_training_shap_box_timing import (  # noqa: E402
    RESULTS,
    flatten_metrics,
    run_arm,
)


SCHEDULE = "20x25"
ROUNDS, BATCH_SIZE = 20, 25
FIXED_DELTA = 0.10
RECIPIENT_LIMIT = 4
TIMING = "post_review_shap_next_round"
STEM = RESULTS / "split_training_shap_box_top4_confirm"


def save_results(results: dict, protocol: dict, stem: Path, started: float) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    compact = {
        key: {
            "rounds": value["rounds"],
            "runtime_seconds": value["runtime_seconds"],
            "direct_review_count": len(value["direct_reviews"]),
            "propagation_event_count": len(value["propagation_events"]),
        }
        for key, value in results.items()
    }
    payload = {
        "protocol": protocol,
        "completed_trajectories": list(results),
        "results": compact,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    checkpoints, direct_reviews, propagation_events = [], [], []
    for key, value in results.items():
        seed, arm = key.split("|")
        for row in value["rounds"]:
            checkpoints.append({"seed": int(seed), "arm": arm, **flatten_metrics(row)})
        direct_reviews.extend(
            {"seed": int(seed), "arm": arm, **row}
            for row in value["direct_reviews"]
        )
        propagation_events.extend(
            {"seed": int(seed), "arm": arm, **row}
            for row in value["propagation_events"]
        )
    pd.DataFrame(checkpoints).to_csv(f"{stem}_checkpoints.csv", index=False)
    pd.DataFrame(direct_reviews).to_csv(f"{stem}_direct_reviews.csv", index=False)
    pd.DataFrame(propagation_events).to_csv(f"{stem}_propagated.csv", index=False)

    lines = [
        "# Paired confirmation: SHAP-box top-four vs direct-only",
        "",
        "Delayed post-review SHAP; 20×25; 500 direct reviews per arm; run seeds "
        + ", ".join(str(seed) for seed in protocol["seeds"]) + ".",
        "Top-four arm: adaptive SHAP top-k, δ=.10, nearest normalized SHAP distance, "
        "maximum four new recipients per seed. Direct-only arm has propagation disabled.",
        "Full-data metrics include fit rows; non-fit metrics are recorded as secondary.",
        "",
        "## Final and first-perfect checkpoint",
        "",
        "| Seed | Arm | Final P | Final R | First P/R=1/1 reviews | Propagated | Wrong events |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for key, value in results.items():
        seed_text, arm = key.split("|")
        rows = value["rounds"]
        final_row = rows[-1]
        final = final_row.get("final_model_metrics") or final_row["model_metrics"]
        first = next((row for row in rows
                      if row["model_metrics"]["model_full_precision"] == 1.0
                      and row["model_metrics"]["model_full_recall"] == 1.0), None)
        first_reviews = "not reached" if first is None else str(first["review_budget"])
        events = value["propagation_events"]
        wrong = sum(event["propagated_label"] != event["true_label"] for event in events)
        lines.append(
            f"| {seed_text} | {arm} | {final['model_full_precision']:.5f} | "
            f"{final['model_full_recall']:.5f} | {first_reviews} | {len(events)} | {wrong} |"
        )
    lines.extend([
        "", "## Full-data P/R at every retrain", "",
        "| Seed | Arm | Reviews | P | R | FP | FN | New propagated | Active propagated |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key, value in results.items():
        seed_text, arm = key.split("|")
        for row in value["rounds"]:
            metrics = row["model_metrics"]
            lines.append(
                f"| {seed_text} | {arm} | {row['review_budget']} | "
                f"{metrics['model_full_precision']:.5f} | {metrics['model_full_recall']:.5f} | "
                f"{metrics['model_full_fp']} | {metrics['model_full_fn']} | "
                f"{row['new_propagated']} | {row['active_propagated']} |"
            )
    lines.extend([
        "", f"Checkpoint CSV: `{stem.name}_checkpoints.csv`.",
        f"Direct-review IDs: `{stem.name}_direct_reviews.csv`.",
        f"Recipient ledger: `{stem.name}_propagated.csv`.",
        "",
    ])
    Path(f"{stem}_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,123,2024")
    parser.add_argument("--tag", default="seeds7_123_2024")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if not seeds:
        raise ValueError("At least one run seed is required")

    stem = RESULTS / f"{STEM.name}_{args.tag}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = {}
    protocol = {
        "seeds": list(seeds),
        "schedule": SCHEDULE,
        "direct_reviews": ROUNDS * BATCH_SIZE,
        "timing": TIMING,
        "top4_policy": "adaptive SHAP top-k box, fixed delta .10, nearest normalized SHAP distance, max 4 recipients per reviewed seed",
        "direct_only_control": "same query, initialization, fit cadence, and direct-review budget; propagation disabled",
        "trusted_fraction": TRUSTED_FRACTION,
        "direct_weight": 100,
        "propagated_weight": 1,
        "paired_seed_definition": "Both arms share seed-specific initial state/config; review IDs may diverge after fitted models differ",
        "metrics": "full/non-fit P/R/F1, AUC, FP/FN at every retrain; propagation event counts/correctness offline",
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
        initial_shap = np.asarray(
            initial_model.get_feature_importance(pool, type="ShapValues")[:, :-1],
            dtype=float,
        )
        if initial_shap.shape != X.shape or not np.isfinite(initial_shap).all():
            raise ValueError(f"Seed {seed}: invalid SHAP matrix {initial_shap.shape}")
        score = frame["pred_score"].to_numpy(dtype=float)
        ids = frame["posting_id"].astype(str)
        tie_break = np.random.RandomState(seed).random_sample(len(frame))
        uncertainty = 1.0 - 2.0 * np.abs(score - 0.5)
        order = np.lexsort((tie_break, uncertainty))
        trusted_ids = set(ids.iloc[order[:round(TRUSTED_FRACTION * len(frame))]])
        initial_targets = dict(zip(ids, frame["pred_label"].to_numpy(dtype=int)))
        trusted_targets = {pid: initial_targets[pid] for pid in trusted_ids}

        arms = (
            ("direct_only", False),
            ("nearest_shap_top4", True),
        )
        for arm, propagation_enabled in arms:
            key = f"{seed}|{arm}"
            print(f"[run] {key}; {SCHEDULE}; delta={FIXED_DELTA}; cap={RECIPIENT_LIMIT}", flush=True)
            results[key] = run_arm(
                "direct_only" if not propagation_enabled else "shap_box",
                TIMING,
                BATCH_SIZE,
                ROUNDS,
                frame,
                X,
                pool,
                initial_model,
                initial_shap,
                score,
                trusted_ids,
                trusted_targets,
                tie_break,
                fixed_delta=FIXED_DELTA,
                propagation_enabled=propagation_enabled,
                propagation_limit=RECIPIENT_LIMIT,
                recipient_selection="nearest_shap",
                print_nonfit_metrics=False,
                seed=seed,
            )
            save_results(results, protocol, stem, started)
            print(f"[saved] completed {key}", flush=True)

    print(f"Complete in {time.perf_counter() - started:.1f}s; artifacts: {stem}*", flush=True)


if __name__ == "__main__":
    main()

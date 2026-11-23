"""Delayed SHAP-box δ=.10 with at most four recipients per reviewed seed."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from legacy.experiments.shap_family_propagation import ROOT_CACHE, SEED, TRUSTED_FRACTION, feature_matrix, load_initial
from legacy.experiments.split_training_shap_box_timing import (
    RESULTS, flatten_metrics, run_arm,
)


FIXED_DELTA = 0.10
PROPAGATION_LIMIT = 4
SCHEDULES = ((10, 50), (20, 25))
TIMING = "post_review_shap_next_round"
SELECTORS = ("nearest_shap", "highest_uncertainty")
STEM = RESULTS / "split_training_shap_box_top4_delayed"


def full_metric_columns(frame):
    keep = [
        "schedule", "selector", "round", "review_budget", "batch_size",
        "new_propagated", "active_propagated", "fit_rows", "fit_seconds",
        "elapsed_seconds",
    ]
    keep.extend(key for key in frame.columns if key.startswith("model_metrics_model_full_"))
    keep.extend(key for key in frame.columns if key.startswith("final_model_metrics_model_full_"))
    return frame[[key for key in keep if key in frame.columns]]


def save_results(results, protocol, started):
    compact = {
        key: {
            "runtime_seconds": data["runtime_seconds"],
            "direct_review_count": len(data["direct_reviews"]),
            "seed_policy_count": len(data["reviews"]),
            "propagation_event_count": len(data["propagation_events"]),
        }
        for key, data in results.items()
    }
    (STEM.with_suffix(".json")).write_text(json.dumps({
        "protocol": protocol,
        "completed_trajectories": list(results),
        "results": compact,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }, indent=2), encoding="utf-8")

    checkpoints = []
    seed_rows = []
    direct_rows = []
    propagation_rows = []
    for key, data in results.items():
        schedule, selector = key.split("|")
        for row in data["rounds"]:
            flat = flatten_metrics(row)
            #save full-dataset metrics for this experiment
            flat = {name: value for name, value in flat.items()
                    if not ("nonfit" in name or name.endswith("_auc"))}
            checkpoints.append(flat | {"schedule": schedule, "selector": selector})
        seed_rows.extend({**row, "schedule": schedule, "selector": selector}
                         for row in data["reviews"])
        direct_rows.extend({**row, "schedule": schedule, "selector": selector}
                           for row in data["direct_reviews"])
        propagation_rows.extend({**row, "schedule": schedule, "selector": selector}
                                for row in data["propagation_events"])

    checkpoint_frame = pd.DataFrame(checkpoints)
    seed_frame = pd.DataFrame(seed_rows)
    direct_frame = pd.DataFrame(direct_rows)
    propagation_frame = pd.DataFrame(propagation_rows)
    checkpoint_frame.to_csv(f"{STEM}_checkpoints.csv", index=False)
    seed_frame.to_csv(f"{STEM}_seed_rules.csv", index=False)
    direct_frame.to_csv(f"{STEM}_direct_reviews.csv", index=False)
    propagation_frame.to_csv(f"{STEM}_propagated.csv", index=False)

    with pd.ExcelWriter(f"{STEM}.xlsx", engine="openpyxl") as writer:
        full_metric_columns(checkpoint_frame).to_excel(writer, sheet_name="Full-data P-R", index=False)
        seed_frame.to_excel(writer, sheet_name="Seed selection", index=False)
        direct_frame.to_excel(writer, sheet_name="Direct reviews", index=False)
        propagation_frame.to_excel(writer, sheet_name="Propagated cases", index=False)

    lines = [
        "# SHAP-box δ=.10 top-four recipient experiment", "",
        "Delayed post-review SHAP; seed 42; 500 direct reviews per trajectory.",
        "SHAP_RAW; adaptive decision-space top-k per seed; fixed relative δ=.10 bands.",
        "At most four eligible new recipients per seed, excluding seed; fit/reviewed/prior-propagated rows excluded.",
        "Selectors: nearest relative SHAP distance in that seed's top-k space; highest current-model uncertainty.",
        "Nearest distance is Euclidean norm of per-feature SHAP differences divided by the δ=.10 band half-width (floor 1e-9).",
        "First reviewed seed wins overlaps. Unreviewed labels used only for offline metrics.",
        "Metrics below use full dataset only; no non-fit metrics reported.", "",
        "## Final metrics", "",
        "| Schedule | Selector | Reviews | Full P | Full R | FP | FN | Active propagated | New propagation events | Wrong events | Runtime s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, data in results.items():
        schedule, selector = key.split("|")
        row = data["rounds"][-1]
        metrics = row.get("final_model_metrics", row["model_metrics"])
        events = data["propagation_events"]
        wrong = sum(event["propagated_label"] != event["true_label"] for event in events)
        lines.append(
            f"| {schedule} | {selector} | {len(data['direct_reviews'])} | "
            f"{metrics['model_full_precision']:.5f} | {metrics['model_full_recall']:.5f} | "
            f"{metrics['model_full_fp']} | {metrics['model_full_fn']} | "
            f"{row['active_propagated']} | {len(events)} | {wrong} | {data['runtime_seconds']:.1f} |"
        )
    lines.extend([
        "", "## Every fit checkpoint", "",
        "| Schedule | Selector | Reviews | Full P | Full R | FP | FN | New prop | Active prop | Fit rows |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key, data in results.items():
        schedule, selector = key.split("|")
        for row in data["rounds"]:
            metrics = row["model_metrics"]
            lines.append(
                f"| {schedule} | {selector} | {row['review_budget']} | "
                f"{metrics['model_full_precision']:.5f} | {metrics['model_full_recall']:.5f} | "
                f"{metrics['model_full_fp']} | {metrics['model_full_fn']} | "
                f"{row['new_propagated']} | {row['active_propagated']} | {row['fit_rows']} |"
            )
    Path(f"{STEM}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frame = load_initial()
    frame["posting_id"] = frame["posting_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    frame["pred_label"] = frame["pred_label"].astype(int)
    X = feature_matrix(frame)
    pool = Pool(X)
    ids = frame["posting_id"].astype(str)
    model = CatBoostClassifier()
    model.load_model(str(ROOT_CACHE / f"initial_seed_{SEED}.cbm"))
    shap = np.asarray(np.load(ROOT_CACHE / f"initial_seed_{SEED}_shap.npy"), dtype=float)
    if shap.shape != X.shape or not np.isfinite(shap).all():
        raise ValueError("Cached SHAP matrix does not match feature matrix")
    score = frame["pred_score"].to_numpy(dtype=float)
    uncertainty = 1.0 - 2.0 * np.abs(score - 0.5)
    tie_break = np.random.RandomState(SEED).random_sample(len(frame))
    order = np.lexsort((tie_break, uncertainty))
    trusted_ids = set(ids.iloc[order[:round(TRUSTED_FRACTION * len(frame))]])
    initial_targets = dict(zip(ids, frame["pred_label"].to_numpy(dtype=int)))
    trusted_targets = {pid: initial_targets[pid] for pid in trusted_ids}

    protocol = {
        "seed": SEED,
        "timing": TIMING,
        "schedules": [{"rounds": rounds, "batch_size": size} for rounds, size in SCHEDULES],
        "direct_reviews_each": 500,
        "propagation": "production adaptive top-k live SHAP_RAW box, fixed delta .10",
        "recipient_limit": PROPAGATION_LIMIT,
        "limit_scope": "per reviewed seed; excludes seed itself",
        "selectors": {
            "nearest_shap": "smallest Euclidean SHAP difference normalized by per-feature delta=.10 half-width, on seed adaptive top-k",
            "highest_uncertainty": "largest current-model margin uncertainty among box matches",
        },
        "eligibility": "trusted, directly reviewed and previously propagated cases excluded; first seed wins overlap",
        "weights": {"direct": 100, "propagated": 1},
        "metrics": "full-dataset P/R, FP/FN at every fit; offline propagation accuracy",
    }
    results = {}
    for rounds, batch_size in SCHEDULES:
        schedule = f"{rounds}x{batch_size}"
        for selector in SELECTORS:
            key = f"{schedule}|{selector}"
            print(f"\n[run] {key}; delta={FIXED_DELTA}; max_recipients={PROPAGATION_LIMIT}", flush=True)
            results[key] = run_arm(
                "shap_box", TIMING, batch_size, rounds, frame, X, pool, model, shap, score,
                trusted_ids, trusted_targets, tie_break,
                fixed_delta=FIXED_DELTA,
                propagation_limit=PROPAGATION_LIMIT,
                recipient_selection=selector,
                print_nonfit_metrics=False,
            )
            save_results(results, protocol, started)
            print(f"[saved] {STEM} artifacts", flush=True)
    print(f"Complete in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()

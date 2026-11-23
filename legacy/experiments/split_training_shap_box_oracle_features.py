"""Paired direct-only vs per-feature-tagged SHAP-box runs on matched seeds."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool

from legacy.experiments.split_training_feature_value_families import _initial_cache
from legacy.experiments.split_training_shap_box_timing import (
    RESULTS,
    flatten_metrics,
    run_arm,
)
from legacy.experiments.shap_family_propagation import TRUSTED_FRACTION, feature_matrix


SCHEDULES = ((10, 50), (20, 25))
TIMING = "post_review_shap_next_round"
TAGGED_ARM = "feature_tagged_shap_box"
DEFAULT_SEEDS = (42, 7, 123, 2024)
STEM = RESULTS / "split_training_shap_box_oracle_features_paired"


def _key_parts(key: str) -> tuple[int, str, str]:
    seed, schedule, arm = key.split("|")
    return int(seed), schedule, arm


def _source_metric(metrics: dict | None, source: str, metric: str) -> str:
    value = ((metrics or {}).get(source) or {}).get(metric)
    return "" if value is None else f"{value:.5f}"


def save_artifacts(results, protocol, started, stem):
    compact = {
        key: {
            "round_count": len(data["rounds"]),
            "runtime_seconds": data["runtime_seconds"],
            "direct_review_count": len(data["direct_reviews"]),
            "policy_count": len(data["reviews"]),
            "propagation_event_count": len(data["propagation_events"]),
        }
        for key, data in results.items()
    }
    stem.with_suffix(".json").write_text(json.dumps({
        "protocol": protocol,
        "completed_trajectories": list(results),
        "results": compact,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }, indent=2), encoding="utf-8")

    checkpoints = []
    seed_policies = []
    decisions = []
    reviews = []
    propagated = []
    for key, data in results.items():
        seed, schedule, arm = _key_parts(key)
        for row in data["rounds"]:
            checkpoint = flatten_metrics(row)
            checkpoint.update({
                "seed": seed, "schedule": schedule, "arm": arm,
                "fit_pass": "retrain",
            })
            checkpoints.append(checkpoint)
            if row.get("final_model_metrics"):
                final_row = {
                    "round": row["round"], "review_budget": row["review_budget"],
                    "fit_rows": row.get("final_fit_rows"),
                    "model_metrics": row["final_model_metrics"],
                    "fit_target_metrics": row.get("final_fit_target_metrics"),
                }
                final_checkpoint = flatten_metrics(final_row)
                final_checkpoint.update({
                    "seed": seed, "schedule": schedule, "arm": arm,
                    "fit_pass": "final_flush",
                })
                checkpoints.append(final_checkpoint)

        for row in data["reviews"]:
            base = {k: v for k, v in row.items() if k != "oracle_choices"}
            seed_policies.append({**base, "seed": seed, "schedule": schedule, "arm": arm})
            for choice in row.get("oracle_choices", []):
                decisions.append({
                    "seed": seed,
                    "schedule": schedule,
                    "arm": arm,
                    "review_budget": row.get("review_budget"),
                    "seed_posting_id": row.get("seed_posting_id"),
                    "seed_label": row.get("seed_label"),
                    "adaptive_k": row.get("k"),
                    "feature": choice.get("feature"),
                    "choice": choice.get("choice"),
                    "bounds": json.dumps(choice.get("bounds")),
                    "pure_group_support": row.get("raw_family_size"),
                    "exception": row.get("oracle_exception"),
                })
        reviews.extend({**row, "seed": seed, "schedule": schedule, "arm": arm}
                       for row in data["direct_reviews"])
        propagated.extend({**row, "seed": seed, "schedule": schedule, "arm": arm}
                          for row in data["propagation_events"])

    checkpoint_frame = pd.DataFrame(checkpoints)
    checkpoint_frame.to_csv(f"{stem}_checkpoints.csv", index=False)
    pd.DataFrame(decisions).to_csv(f"{stem}_oracle_decisions.csv", index=False)
    pd.DataFrame(reviews).to_csv(f"{stem}_direct_reviews.csv", index=False)
    pd.DataFrame(propagated).to_csv(f"{stem}_propagated.csv", index=False)

    with pd.ExcelWriter(f"{stem}.xlsx", engine="openpyxl") as writer:
        checkpoint_frame.to_excel(writer, sheet_name="Checkpoints", index=False)
        pd.DataFrame(seed_policies).to_excel(writer, sheet_name="Seed policies", index=False)
        pd.DataFrame(decisions).to_excel(writer, sheet_name="Feature choices", index=False)
        pd.DataFrame(reviews).to_excel(writer, sheet_name="Direct reviews", index=False)
        pd.DataFrame(propagated).to_excel(writer, sheet_name="Propagation", index=False)

    lines = [
        f"# {protocol.get('report_title', 'Paired per-feature SHAP-box and direct-only comparison')}", "",
        f"Seeds: {', '.join(map(str, protocol['seeds']))}; delayed post-review SHAP; "
        f"SHAP_RAW; pseudo-target pool={protocol.get('pseudo_target_fraction', TRUSTED_FRACTION):.0%}; "
        "500 direct reviews per schedule.",
        f"Arms: {', '.join(protocol.get('arms', {}).keys())}.",
        *(["Tagged arm uses adaptive top-k and per-feature ignore/δ=.25/.10/.05 settings; no recipient cap."]
          if TAGGED_ARM in protocol.get("arms", {}) else []),
        "At each retrain, fit-target agreement is measured separately for trusted "
        "pseudo-labels, direct reviews, and propagated labels.",
        "P/R is full-data P/R at each retrain. Agreement is in-sample model agreement "
        "with assigned fit targets, not a separate evaluation metric.", "",
        "## Final checkpoint by seed", "",
        "| Seed | Schedule | Arm | Reviews | Full P | Full R | First P/R=1/1 | "
        "Pseudo agreement | Direct agreement | Propagated agreement | Wrong propagated |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, data in results.items():
        seed, schedule, arm = _key_parts(key)
        final_row = data["rounds"][-1]
        final_metrics = final_row.get("final_model_metrics") or final_row["model_metrics"]
        fit_metrics = final_row.get("final_fit_target_metrics") or final_row.get("fit_target_metrics") or {}
        first_perfect = None
        for row in data["rounds"]:
            stage = row["model_metrics"]
            final = row.get("final_model_metrics")
            if (stage.get("model_full_precision") == 1.0
                    and stage.get("model_full_recall") == 1.0) or (
                    final and final.get("model_full_precision") == 1.0
                    and final.get("model_full_recall") == 1.0):
                first_perfect = row["review_budget"]
                break
        wrong = sum(int(event.get("propagated_label") != event.get("true_label"))
                    for event in data["propagation_events"])
        lines.append(
            f"| {seed} | {schedule} | {arm} | {len(data['direct_reviews'])} | "
            f"{final_metrics.get('model_full_precision'):.5f} | "
            f"{final_metrics.get('model_full_recall'):.5f} | "
            f"{first_perfect if first_perfect is not None else 'not reached'} | "
            f"{_source_metric(fit_metrics, 'pseudo', 'agreement')} | "
            f"{_source_metric(fit_metrics, 'direct', 'agreement')} | "
            f"{_source_metric(fit_metrics, 'propagated', 'agreement')} | {wrong} |"
        )

    lines.extend([
        "", "## P/R and fit-target agreement at every retrain", "",
        "| Seed | Schedule | Arm | Fit pass | Reviews | Full P | Full R | "
        "Pseudo n/agreement/loss | Direct n/agreement/loss | "
        "Propagated n/agreement/loss | Fit rows |",
        "|---:|---|---|---|---:|---:|---:|---|---|---|---:|",
    ])
    for row in checkpoints:
        pseudo = {
            "n": row.get("fit_target_metrics_pseudo_n"),
            "agreement": row.get("fit_target_metrics_pseudo_agreement"),
            "log_loss": row.get("fit_target_metrics_pseudo_log_loss"),
        }
        direct = {
            "n": row.get("fit_target_metrics_direct_n"),
            "agreement": row.get("fit_target_metrics_direct_agreement"),
            "log_loss": row.get("fit_target_metrics_direct_log_loss"),
        }
        propagated = {
            "n": row.get("fit_target_metrics_propagated_n"),
            "agreement": row.get("fit_target_metrics_propagated_agreement"),
            "log_loss": row.get("fit_target_metrics_propagated_log_loss"),
        }

        def source_cell(values):
            if values["n"] is None:
                return "—"
            agreement = "—" if values["agreement"] is None else f"{values['agreement']:.4f}"
            loss = "—" if values["log_loss"] is None else f"{values['log_loss']:.4f}"
            return f"{values['n']}/{agreement}/{loss}"

        p = row.get("model_metrics_model_full_precision")
        r = row.get("model_metrics_model_full_recall")
        lines.append(
            f"| {row['seed']} | {row['schedule']} | {row['arm']} | "
            f"{row.get('fit_pass', 'retrain')} | "
            f"{row.get('review_budget', '')} | {'' if p is None else f'{p:.5f}'} | "
            f"{'' if r is None else f'{r:.5f}'} | {source_cell(pseudo)} | "
            f"{source_cell(direct)} | {source_cell(propagated)} | {row.get('fit_rows', '')} |"
        )
    lines.extend([
        "", "Fit-source cells show count / raw agreement / mean binary log loss. "
        "Final-flush rows use final-model metrics after last delayed propagation is fitted. "
        "Repeated last-budget rows are final flushes, not extra reviewer checkpoints.",
        "Per-feature choices and recipient ledger are in the companion CSV/XLSX files.",
    ])
    Path(f"{stem}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--tag", default="seeds42_7_123_2024")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide a non-empty list of unique run seeds")

    stem = RESULTS / f"{STEM.name}_{args.tag}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = {}
    protocol = {
        "seeds": list(seeds),
        "timing": TIMING,
        "schedules": [{"rounds": rounds, "batch_size": batch_size}
                      for rounds, batch_size in SCHEDULES],
        "direct_reviews_each": 500,
        "trusted_fraction": TRUSTED_FRACTION,
        "arms": ["direct_only", TAGGED_ARM],
        "tagged_policy": "per-feature ignore/delta .25/.10/.05, adaptive top-k, maximum-pure-support policy, no recipient cap",
        "fit_target_metrics": "raw agreement and mean binary log loss by pseudo, direct, propagated source at each fit; objective-weighted fields also in checkpoint CSV",
        "comparison": "full-data P/R at every retrain; direct-only and tagged SHAP-box share seed-specific initialization and schedule",
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

        for rounds, batch_size in SCHEDULES:
            schedule = f"{rounds}x{batch_size}"
            arms = (
                ("direct_only", "direct_only", False),
                (TAGGED_ARM, "oracle_feature_policy", True),
            )
            for arm, mode, propagation_enabled in arms:
                key = f"{seed}|{schedule}|{arm}"
                print(f"[run] {key}; delayed; no recipient cap", flush=True)
                results[key] = run_arm(
                    mode,
                    TIMING,
                    batch_size,
                    rounds,
                    frame,
                    X,
                    pool,
                    initial_model,
                    initial_shap,
                    score,
                    trusted_ids,
                    trusted_targets,
                    tie_break,
                    propagation_enabled=propagation_enabled,
                    print_nonfit_metrics=False,
                    seed=seed,
                    track_fit_target_agreement=True,
                )
                save_artifacts(results, protocol, started, stem)
                print(f"[saved] completed {key}", flush=True)

    print(f"Complete in {time.perf_counter() - started:.1f}s; artifacts: {stem}*", flush=True)


if __name__ == "__main__":
    main()

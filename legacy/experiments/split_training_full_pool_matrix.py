"""Full-pool random, uncertainty, and tagged SHAP-box comparison arms.

Split-only and split-plus-tagged arms live in the paired split-training runner;
this runner supplies the three non-split arms for the four-seed comparison.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from catboost import Pool

from legacy.experiments.split_training_feature_value_families import _initial_cache
from legacy.experiments.split_training_shap_box_oracle_features import (
    TAGGED_ARM,
    TIMING,
    save_artifacts,
)
from legacy.experiments.split_training_shap_box_timing import RESULTS, run_arm
from legacy.experiments.shap_family_propagation import feature_matrix


DEFAULT_SEEDS = (42, 7, 123, 2024)
MATRIX_SCHEDULES = ((20, 25),)
ARM_SETTINGS = {
    "random_direct_only": ("direct_only", False, "random"),
    "uncertainty_direct_only": ("direct_only", False, "uncertainty"),
    TAGGED_ARM: ("oracle_feature_policy", True, "uncertainty"),
}
DEFAULT_ARMS = ("random_direct_only", TAGGED_ARM)
STEM = RESULTS / "split_training_full_pool_matrix"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--tag", default="matched4")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                        help="Comma-separated arms; defaults to missing random and tagged-SHAP arms")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide a non-empty list of unique run seeds")
    arms = tuple(value for value in args.arms.split(",") if value)
    if not arms or len(set(arms)) != len(arms) or any(value not in ARM_SETTINGS for value in arms):
        raise ValueError(f"Arms must be unique members of {tuple(ARM_SETTINGS)}")

    stem = RESULTS / f"{STEM.name}_{args.tag}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = {}
    protocol = {
        "report_title": "Full-pool random, uncertainty, and feature-tagged SHAP-box comparison",
        "seeds": list(seeds),
        "timing": TIMING,
        "schedules": [{"rounds": rounds, "batch_size": batch_size}
                      for rounds, batch_size in MATRIX_SCHEDULES],
        "direct_reviews_each": 500,
        "pseudo_target_fraction": 1.0,
        "arms": {
            arm: {
                "random_direct_only": "random query; full initial pseudo-target pool; no propagation",
                "uncertainty_direct_only": "uncertainty query; full initial pseudo-target pool; no propagation",
                TAGGED_ARM: "uncertainty query; full initial pseudo-target pool; per-feature SHAP tags; no recipient cap",
            }[arm]
            for arm in arms
        },
        "tagged_policy": "adaptive top-k; per-feature ignore/delta .25/.10/.05; delayed live-SHAP matching",
        "weights": "direct 100; all pseudo and propagated targets 1, with class weight recomputed from current fit targets",
        "fit_target_metrics": "model-to-assigned-target agreement and mean binary log loss, grouped by pseudo/direct/propagated target source",
        "comparison": "No split: every initial pseudo-target row enters fitting. Direct-only and tagged arms share per-seed initial state and schedule.",
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
        all_ids = set(ids)
        initial_targets = dict(zip(ids, frame["pred_label"].to_numpy(dtype=int)))

        for rounds, batch_size in MATRIX_SCHEDULES:
            schedule = f"{rounds}x{batch_size}"
            for arm in arms:
                mode, propagation_enabled, query_policy = ARM_SETTINGS[arm]
                key = f"{seed}|{schedule}|{arm}"
                print(
                    f"[run] {key}; full pseudo pool; query={query_policy}; "
                    f"propagation={propagation_enabled}", flush=True,
                )
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
                    all_ids,
                    initial_targets,
                    tie_break,
                    propagation_enabled=propagation_enabled,
                    query_policy=query_policy,
                    exclude_trusted_from_queries=False,
                    exclude_trusted_from_propagation=False,
                    print_nonfit_metrics=False,
                    seed=seed,
                    track_fit_target_agreement=True,
                )
                save_artifacts(results, protocol, started, stem)
                print(f"[saved] completed {key}", flush=True)

    print(f"Complete in {time.perf_counter() - started:.1f}s; artifacts: {stem}*", flush=True)


if __name__ == "__main__":
    main()

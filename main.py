import os
import time
from datetime import datetime

import numpy as np
from openpyxl import Workbook, load_workbook
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix

import logstore
from dataprep import full_dataprep
from isolation_forest import run_if
from models.kittyboost import train_catboost
from querystep import novelty_scores, select_batch
from propagation import (
    FLAG_PREFIXES,
    _adaptive_topk,
    apply_shap_box_policy,
    shap_box_selection,
)
from shap_box_policy import (
    POLICY_SCHEMA_VERSION,
    create_reviewer_shap_box_policy,
    load_policy_replay,
)

def _uncertainty_values(df, strategy, novelty_cache=None):
    if strategy == "margin":
        return 1.0 - 2.0 * np.abs(df["pred_score"].values - 0.5)
    elif strategy == "entropy":
        p = np.clip(df["pred_score"].values, 1e-12, 1.0 - 1e-12)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)
    elif strategy == "novelty":
        if novelty_cache is None:
            return None
        return np.asarray(novelty_cache, dtype=float)
    else:
        return None


def _drive_results_path():
    """Return the Colab Drive path for the results workbook, if mounted."""
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/test_results.xlsx"
    return None


def test_logger(header, results):
    drive_path = _drive_results_path()
    #use drive workbook, then local workbook, then create new
    base = drive_path if (drive_path and os.path.exists(drive_path)) else None
    if base is None and os.path.exists("test_results.xlsx"):
        base = "test_results.xlsx"
    try:
        wb = load_workbook(base) if base else Workbook()
        ws = wb.active
    except Exception:
        wb = Workbook()
        ws = wb.active

    ws.append([header])
    for result_str in results:
        ws.append([result_str])
    ws.append(["'" + "=" * 80])  #separate runs

    for path in ["test_results.xlsx"] + ([drive_path] if drive_path else []):
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            wb.save(path)
            print("Results saved to", path)
        except Exception as e:
            print("Failed to save", path, ":", e)


def retrain_catboost(df, l=10, corrected_weights=100, corrected_saved=True,
                     strategy="margin", greedy_batching=False, greedy_T=None,
                     compute_shap=False, skip_retrain_on_skip=False,
                     propagation_space=None, initial_shap=None, initial_model=None,
                     prop_weight_mode="uniform", selection_mode="uncertainty",
                     early_stop=False, chunk_size=50, min_delta=0.001,
                     stop_patience=2, seed=42, run=None, retrain_every=1,
                     propagation_mode=None, prop_k=2, prop_delta=0.25,
                     feature_space="off", basket_alpha=0.5,
                     n_representatives=3, rep_selection="farthest_point",
                     fallback="family", split_training=False,
                     policy_mode=None, policy_replay=None,
                     review_callback=None, phase_callback=None):
    """Canonical CatBoost retrain loop, called by run_supervised."""
    if propagation_mode not in (None, "shap_box_selection"):
        raise ValueError(
            f"Propagation mode {propagation_mode!r} lives in legacy/main.py"
        )
    tagged_selection = propagation_mode == "shap_box_selection"
    compute_shap = tagged_selection or compute_shap
    if tagged_selection and review_callback is not None:
        policy_mode = "manual"
    else:
        policy_mode = (policy_mode or "discover") if tagged_selection else "none"
    if retrain_every < 1:
        raise ValueError("retrain_every must be at least 1")
    df = df.reset_index(drop=True).copy()
    df["posting_id"] = df["posting_id"].astype(str)

    ids = df["posting_id"].astype(str).to_numpy()
    id_to_row = {pid: row for row, pid in enumerate(ids)}
    labels = df["label"].to_numpy(dtype=int)
    true_by_id = dict(zip(ids, labels))
    initial_targets = dict(zip(ids, df["pred_label"].to_numpy(dtype=int)))
    tie_break = np.random.RandomState(seed).random_sample(len(df))
    random_priority = np.random.RandomState(seed).random_sample(len(df))
    initial_uncertainty = 1.0 - 2.0 * np.abs(df["pred_score"].to_numpy(dtype=float) - 0.5)

    if split_training:
        trusted_order = np.lexsort((tie_break, initial_uncertainty))
        trusted_ids = set(ids[trusted_order[:round(0.5 * len(df))]])
        trusted_targets = {pid: initial_targets[pid] for pid in trusted_ids}
    else:
        trusted_ids = set(ids)
        trusted_targets = dict(initial_targets)

    direct_targets = {}
    reviewer_choices = {}
    reviewed_ids = set()
    propagated_targets = {}
    pending_batch = []
    current_batch = []
    novelty_cache = novelty_scores(df) if strategy == "novelty" else None
    feature_names = [name for name in df.columns
                     if name not in {"label", "posting_id", "pred_label", "pred_score"}]
    raw_cols = [i for i, name in enumerate(feature_names)
                if not name.startswith(FLAG_PREFIXES)]
    raw_names = [feature_names[i] for i in raw_cols]
    results = []
    cat_importances = None
    cat_model = None
    precision = recall = 0.0
    tn = fp = fn = tp = 0
    trees_this = 0
    query_score = df["pred_score"].to_numpy(dtype=float).copy()
    current_shap = initial_shap
    final_flush_metrics = None

    def fit_current_targets():
        fit_targets = dict(trusted_targets)
        fit_targets.update(propagated_targets)
        fit_targets.update(direct_targets)
        fit_ids = set(fit_targets)
        fitted = train_catboost(
            df,
            corrected_ids=set(direct_targets),
            corrected_weights=corrected_weights,
            compute_shap=tagged_selection or compute_shap,
            early_stop=early_stop,
            chunk_size=chunk_size,
            min_delta=min_delta,
            stop_patience=stop_patience,
            seed=seed,
            fit_ids=fit_ids,
            fit_targets=fit_targets,
        )
        return fitted, fit_targets, fit_ids

    for iteration in range(1, l + 1):
        if not current_batch:
            if strategy == "random":
                priority = random_priority
            else:
                priority = _uncertainty_values(df, strategy, novelty_cache)
                if priority is None:
                    raise ValueError(f"Unsupported strategy for batch workflow: {strategy}")
            current_batch = select_batch(
                df,
                trusted_ids if split_training else set(),
                reviewed_ids,
                priority,
                tie_break,
                min(retrain_every, l - iteration + 1),
            )
        posting_id = current_batch.pop(0)
        reviewed_ids.add(posting_id)
        if review_callback is None:
            direct_label = int(true_by_id[posting_id])
        else:
            row_idx = id_to_row[posting_id]
            shap_row = (
                np.asarray(current_shap[row_idx, raw_cols], dtype=float)
                if current_shap is not None else np.zeros(len(raw_names), dtype=float)
            )
            score = float(df.loc[row_idx, "pred_score"])
            adaptive_k, adaptive_positions = _adaptive_topk(shap_row, score)
            review_decision = review_callback({
                "posting_id": posting_id,
                "iteration": iteration,
                "review_count": iteration - 1,
                "review_target": retrain_every,
                "total_reviews": l,
                "model_probability": score,
                "predicted_label": int(df.loc[row_idx, "pred_label"]),
                "uncertainty": float(1.0 - 2.0 * abs(score - 0.5)),
                "shap": [
                    {"feature": feature, "value": float(value)}
                    for feature, value in zip(raw_names, shap_row)
                ],
                "adaptive_k": adaptive_k,
                "adaptive_features": [raw_names[int(pos)] for pos in adaptive_positions],
            })
            if not isinstance(review_decision, dict) or "label" not in review_decision:
                raise ValueError("review_callback must return dict with label")
            label_value = review_decision["label"]
            if isinstance(label_value, str):
                normalized = label_value.strip().lower()
                if normalized not in {"normal", "anomaly"}:
                    raise ValueError(f"Unsupported reviewer label: {label_value!r}")
                direct_label = int(normalized == "anomaly")
            else:
                direct_label = int(label_value)
            if direct_label not in (0, 1):
                raise ValueError(f"Reviewer label must be 0/1, got {label_value!r}")
            reviewer_choices[posting_id] = review_decision.get("feature_choices", {})

        direct_targets[posting_id] = direct_label
        propagated_targets.pop(posting_id, None)
        pending_batch.append(posting_id)
        df.loc[df["posting_id"].astype(str) == posting_id, "pred_label"] = direct_targets[posting_id]

        batch_end = iteration % retrain_every == 0 or iteration == l
        if not batch_end:
            y_true = df["label"].astype(int)
            y_pred = df["pred_label"].astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            if run is not None:
                run.write("iter", run_id=run.run_id, iteration=iteration,
                          mode="batch_review", strategy=strategy,
                          reviewed_id=posting_id, precision=precision, recall=recall)
            continue

        t_fit = time.time()
        if phase_callback is not None:
            phase_callback("retrain", iteration, l, retrain_every)
        fitted, fit_targets, fit_ids = fit_current_targets()
        (df, cat_importances, precision, recall, cat_model,
         tn, fp, fn, tp, shap_vals, early_stop_info) = fitted
        trees_this = cat_model.tree_count_
        query_score = df["pred_score"].to_numpy(dtype=float).copy()
        current_shap = shap_vals
        fit_metrics = _fit_target_agreement(
            df["pred_label"].to_numpy(dtype=int), query_score, ids,
            trusted_targets, direct_targets, propagated_targets,
        )

        new_targets = set()
        candidate_count = fit_blocked = conflicts = 0
        if tagged_selection:
            shap_all = np.asarray(shap_vals, dtype=float)
            shap_raw = shap_all[:, raw_cols]
            propagation_fit_ids = (trusted_ids if split_training else set()) | set(direct_targets) | set(propagated_targets)
            queued_targets = dict(propagated_targets)
            for rank, seed_id in enumerate(pending_batch, start=1):
                seed_row = id_to_row[seed_id]
                if policy_mode == "manual":
                    policy = create_reviewer_shap_box_policy(
                        shap_raw, raw_names, query_score, seed_row,
                        reviewer_choices.pop(seed_id, {}),
                    )
                    mask = apply_shap_box_policy(
                        shap_raw, raw_names, seed_row, policy
                    )
                elif policy_mode == "discover":
                    mask, policy = shap_box_selection(
                        shap_raw, raw_names, query_score, seed_row,
                        direct_targets[seed_id], labels,
                    )
                elif policy_mode == "replay":
                    policy = policy_replay.policy_for(seed_id)
                    mask = apply_shap_box_policy(shap_raw, raw_names, seed_row, policy)
                else:
                    raise ValueError(f"Unsupported tagged policy mode: {policy_mode!r}")
                matched = set(ids[mask])
                blocked = matched & propagation_fit_ids
                eligible = matched - propagation_fit_ids
                recipients = sorted(eligible - set(queued_targets), key=id_to_row.get)
                candidate_count += len(matched)
                fit_blocked += len(blocked)
                conflicts += len(eligible & set(queued_targets))
                for recipient in recipients:
                    queued_targets[recipient] = direct_targets[seed_id]
                    new_targets.add(recipient)
                if run is not None:
                    run.write("policy", run_id=run.run_id, seed_id=seed_id,
                              review_budget=iteration - len(pending_batch) + rank,
                              policy_schema_version=POLICY_SCHEMA_VERSION,
                              policy_source=(
                                  "reviewer" if policy_mode == "manual"
                                  else "replay" if policy_mode == "replay"
                                  else "automatic_discovery"
                              ),
                              policy_source_run_id=(
                                  policy_replay.source_run_id if policy_replay is not None else None
                              ),
                              policy=policy, candidate_matches=len(matched),
                              fit_blocked=len(blocked), new_propagated=len(recipients),
                              propagation_events=[{
                                  "posting_id": recipient,
                                  "propagated_label": direct_targets[seed_id],
                                  "true_label": true_by_id[recipient],
                                  "seed_posting_id": seed_id,
                              } for recipient in recipients])
            propagated_targets = queued_targets

        checkpoint = {
            "round": (iteration - 1) // retrain_every + 1,
            "review_budget": iteration,
            "reviewed_ids": list(pending_batch),
            "metrics": {"precision": precision, "recall": recall,
                        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "fit_target_metrics": fit_metrics,
            "fit_rows": len(fit_ids),
            "new_propagated": len(new_targets),
            "active_propagated": len(propagated_targets),
            "candidate_matches": candidate_count,
            "fit_blocked": fit_blocked,
            "overlap_conflicts": conflicts,
            "fit_seconds": time.time() - t_fit,
        }

        #fit recipients queued by final batch
        if iteration == l:
            t_flush = time.time()
            final_fit, _final_targets, final_fit_ids = fit_current_targets()
            (df, cat_importances, precision, recall, cat_model,
             tn, fp, fn, tp, _final_shap, early_stop_info) = final_fit
            final_flush_metrics = {
                "precision": precision, "recall": recall,
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "fit_rows": len(final_fit_ids),
                "fit_seconds": time.time() - t_flush,
            }
            checkpoint["final_flush"] = final_flush_metrics

        if run is not None:
            run.write("iter", run_id=run.run_id, iteration=iteration,
                      mode="retrain", strategy=strategy,
                      propagation_mode="shap_box_selection" if tagged_selection else None,
                      split_training=split_training, **checkpoint)
        results.append(
            f"Iteration: {iteration}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
            f"Reviews: {len(direct_targets)}, Fit rows: {len(fit_ids)}, "
            f"Propagated: {len(new_targets)}"
        )
        pending_batch = []

    if policy_replay is not None:
        policy_replay.assert_complete()
    return df, cat_importances, precision, recall, cat_model, results


def run_unsupervised(compute_shap=False, early_stop=False, chunk_size=50, min_delta=0.001, stop_patience=2, seed=42, run=None):
    df, labels = full_dataprep()
    df_if = run_if(df, labels, verbose=False, seed=seed)
    df_cat, cat_importances, precision, recall, cat_model, tn, fp, fn, tp, initial_shap, early_stop_info = train_catboost(
        df_if, verbose=True, compute_shap=compute_shap,
        early_stop=early_stop, chunk_size=chunk_size, min_delta=min_delta,
        stop_patience=stop_patience, seed=seed)
    print(f"\nInitial Precision: {precision:.4f}, Initial Recall: {recall:.4f}, Trees: {cat_model.tree_count_}")
    if run is not None:
        try:
            auc = float(roc_auc_score(df_cat["label"].astype(int), df_cat["pred_score"]))
        except ValueError:
            auc = None
        run.write("initial", run_id=run.run_id, iteration=0, mode="initial",
                  precision=precision, recall=recall, model_trees=cat_model.tree_count_,
                  auc=auc, early_stop=early_stop_info)
    return df_cat, cat_importances, precision, recall, cat_model, initial_shap


def _fit_target_agreement(pred, score, ids, trusted_targets,
                                    direct_labels, prop_targets):
    targets = dict(trusted_targets)
    targets.update(prop_targets)
    targets.update(direct_labels)
    positives = int(sum(targets.values()))
    anomaly_weight = max(5.0, len(targets) / positives) if positives else 5.0
    row_by_id = {pid: row for row, pid in enumerate(ids)}
    sources = {
        "pseudo": {pid: label for pid, label in trusted_targets.items()
                   if pid not in direct_labels and pid not in prop_targets},
        "propagated": prop_targets,
        "direct": direct_labels,
    }
    probs = np.clip(np.asarray(score, dtype=float), 1e-15, 1.0 - 1e-15)
    predictions = np.asarray(pred, dtype=int)
    result = {}
    for source, labels in sources.items():
        if not labels:
            result[source] = {"n": 0, "agreement": None, "log_loss": None,
                              "objective_weighted_agreement": None,
                              "objective_weighted_log_loss": None}
            continue
        rows = np.fromiter((row_by_id[pid] for pid in labels), dtype=int, count=len(labels))
        y = np.fromiter((labels[ids[row]] for row in rows), dtype=int, count=len(rows))
        matched = predictions[rows] == y
        losses = -(y * np.log(probs[rows]) + (1 - y) * np.log(1 - probs[rows]))
        weights = np.where(y == 1, anomaly_weight, 1.0)
        if source == "direct":
            weights *= 100.0
        result[source] = {
            "n": int(len(rows)), "agreement": float(np.mean(matched)),
            "log_loss": float(np.mean(losses)),
            "objective_weighted_agreement": float(np.average(matched, weights=weights)),
            "objective_weighted_log_loss": float(np.average(losses, weights=weights)),
        }
    return result

def run_supervised(training_strat="retrain", l=10, corrected_weights=100,
                   corrected_saved=True, strategy="margin", return_full_data=False,
                   greedy_batching=False, greedy_T=None, compute_shap=False,
                   skip_retrain_on_skip=False, propagation_space=None,
                   prop_weight_mode="uniform", selection_mode="uncertainty",
                   early_stop=False, chunk_size=50, min_delta=0.001,
                   stop_patience=2, seed=42, retrain_every=1,
                   propagation_mode=None, prop_k=2, prop_delta=0.25,
                   feature_space="off", basket_alpha=0.5,
                   n_representatives=3, rep_selection="farthest_point",
                   fallback="family", uncertainty_trace=False,
                   split_training=False, policy_mode=None, policy_path=None,
                   review_callback=None, phase_callback=None):
    """Run canonical workflow using established legacy-style call interface."""
    if training_strat != "retrain":
        raise ValueError("Root main.py canonical workflows use training_strat='retrain'")
    if propagation_mode not in (None, "shap_box_selection"):
        raise ValueError(
            f"Propagation mode {propagation_mode!r} lives in legacy/main.py"
        )
    tagged = propagation_mode == "shap_box_selection"
    if tagged and review_callback is not None:
        policy_mode = "manual"
    else:
        policy_mode = (policy_mode or "discover") if tagged else "none"
    if tagged and policy_mode not in ("discover", "replay"):
        if policy_mode != "manual":
            raise ValueError("policy_mode must be 'discover', 'replay', or 'manual'")
    needs_shap = tagged or compute_shap
    config = dict(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        training_strat=training_strat, l=l, corrected_weights=corrected_weights,
        corrected_saved=corrected_saved, strategy=strategy,
        return_full_data=return_full_data, greedy_batching=greedy_batching,
        greedy_T=greedy_T, compute_shap=needs_shap,
        skip_retrain_on_skip=skip_retrain_on_skip,
        propagation_space=propagation_space, prop_weight_mode=prop_weight_mode,
        selection_mode=selection_mode, early_stop=early_stop,
        chunk_size=chunk_size, min_delta=min_delta,
        stop_patience=stop_patience, seed=seed, retrain_every=retrain_every,
        propagation_mode=propagation_mode, prop_k=prop_k, prop_delta=prop_delta,
        feature_space=feature_space, basket_alpha=basket_alpha,
        n_representatives=n_representatives, rep_selection=rep_selection,
        fallback=fallback, uncertainty_trace=bool(uncertainty_trace),
        split_training=bool(split_training), policy_mode=policy_mode,
        policy_source_path=str(policy_path) if policy_path is not None else None,
    )
    if tagged and policy_mode == "replay" and not policy_path:
        raise ValueError("policy_path is required for policy_mode='replay'")
    if tagged and policy_mode == "discover" and policy_path is not None:
        raise ValueError("policy_path is only valid for policy_mode='replay'")
    if tagged and policy_mode == "manual" and policy_path is not None:
        raise ValueError("Manual reviewer policies are held in memory; policy_path is unused")
    if tagged and policy_mode == "manual" and review_callback is None:
        raise ValueError("policy_mode='manual' requires review_callback")
    policy_replay = (
        load_policy_replay(policy_path, config) if policy_mode == "replay" else None
    )
    config["policy_source_run_id"] = (
        policy_replay.source_run_id if policy_replay is not None else None
    )
    run = logstore.open_run(config)
    t0 = time.time()
    if tagged and policy_mode == "discover":
        print(f"SHAP-box policy discovery log: {logstore.RUNS_DIR}/{run.run_id}.jsonl")
    try:
        if phase_callback is not None:
            phase_callback("initial_training", 0, l, retrain_every)
        df, _, _, _, initial_model, initial_shap = run_unsupervised(
            compute_shap=needs_shap, early_stop=early_stop,
            chunk_size=chunk_size, min_delta=min_delta,
            stop_patience=stop_patience, seed=seed, run=run)
        if phase_callback is not None:
            phase_callback("review", 0, l, retrain_every)
        df, importances, precision, recall, model, results = retrain_catboost(
            df, l=l, corrected_weights=corrected_weights,
            corrected_saved=corrected_saved, strategy=strategy,
            greedy_batching=greedy_batching, greedy_T=greedy_T,
            compute_shap=needs_shap, skip_retrain_on_skip=skip_retrain_on_skip,
            propagation_space=propagation_space, initial_shap=initial_shap,
            initial_model=initial_model, prop_weight_mode=prop_weight_mode,
            selection_mode=selection_mode, early_stop=early_stop,
            chunk_size=chunk_size, min_delta=min_delta,
            stop_patience=stop_patience, seed=seed, run=run,
            retrain_every=retrain_every, propagation_mode=propagation_mode,
            split_training=split_training, policy_mode=policy_mode,
            policy_replay=policy_replay, review_callback=review_callback,
            phase_callback=phase_callback)
        run.close(wall_time_s=time.time() - t0,
                  final_precision=precision, final_recall=recall)
        if phase_callback is not None:
            phase_callback("complete", l, l, retrain_every)
        test_logger(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] | run_id={run.run_id} | "
            f"training_strat={training_strat}, l={l}, corrected_weights={corrected_weights}, "
            f"corrected_saved={corrected_saved}, strategy={strategy}, "
            f"greedy_batching={greedy_batching}, greedy_T={greedy_T}, "
            f"compute_shap={needs_shap}, propagation_space={propagation_space}, "
            f"prop_weight_mode={prop_weight_mode}, selection_mode={selection_mode}, "
            f"retrain_every={retrain_every}, propagation_mode={propagation_mode}, "
            f"split_training={split_training}, policy_mode={policy_mode}, "
            f"seed={seed}", results)
        return df, importances, precision, recall, model
    except Exception:
        run.close(wall_time_s=time.time() - t0)
        if phase_callback is not None:
            phase_callback("error", None, l, retrain_every)
        raise


if __name__ == "__main__":
    #choose one run configuration below
    # df, cat_importances, precision, recall, cat_model = run_supervised(
    #     training_strat="retrain", l=500, corrected_weights=100,
    #     corrected_saved=True, strategy="random", retrain_every=25)
    # df, cat_importances, precision, recall, cat_model = run_supervised(
    #     training_strat="retrain", l=500, corrected_weights=100,
    #     corrected_saved=True, strategy="margin", retrain_every=25)
    # df, cat_importances, precision, recall, cat_model = run_supervised(
    #     training_strat="retrain", l=500, corrected_weights=100,
    #     corrected_saved=True, strategy="margin", split_training=True,
    #     retrain_every=25)
    # df, cat_importances, precision, recall, cat_model = run_supervised(
    #     training_strat="retrain", l=500, corrected_weights=100,
    #     corrected_saved=True, strategy="margin",
    #     propagation_mode="shap_box_selection", retrain_every=25)
    # df, cat_importances, precision, recall, cat_model = run_supervised(
    #     training_strat="retrain", l=500, corrected_weights=100,
    #     corrected_saved=True, strategy="margin",
    #     propagation_mode="shap_box_selection", split_training=True,
    #     retrain_every=25)
    pass

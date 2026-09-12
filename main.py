import os
import time
from datetime import datetime

from sklearn.metrics import roc_auc_score

import logstore
from dataprep import full_dataprep
import numpy as np
import pandas as pd
from isolation_forest import run_if
from models.kittyboost import train_catboost
from querystep import uncertainty_query, novelty_scores
from greedy import greedy_iteration, new_state
from openpyxl import Workbook, load_workbook

_UNC_BUCKETS = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


def _uncertainty_distribution(df, strategy, novelty_cache=None, buckets=_UNC_BUCKETS):
    """Anteil Mispredicts in den Top-k der Uncertainty (alle Cases, kumulativ).

    Reines Reporting (nutzt echte Labels) - beeinflusst nie Training/Selektion.
    """
    if strategy == "margin":
        u = 1.0 - 2.0 * np.abs(df["pred_score"].values - 0.5)
    elif strategy == "entropy":
        p = np.clip(df["pred_score"].values, 1e-12, 1.0 - 1e-12)
        u = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)
    elif strategy == "novelty":
        if novelty_cache is None:
            return None
        u = np.asarray(novelty_cache)
    else:
        return None

    mis = df["label"].values.astype(int) != df["pred_label"].values.astype(int)
    order = np.argsort(-u, kind="stable")
    cum = np.cumsum(mis[order])
    pop = len(mis)
    counts = []
    for k in buckets:
        k = min(k, pop)
        if k <= 0:
            counts.append(0)
        else:
            counts.append(int(cum[k - 1]))
    print("unc-dist: " + ", ".join(f"unc@{k}: {c}/{k}" for k, c in zip(buckets, counts)))
    return {
        "strategy": strategy,
        "buckets": list(buckets),
        "mispred_cum": counts,
        "total_mispred": int(cum[-1]) if pop else 0,
        "n_cases": int(pop),
    }

def _drive_results_path():
    """Drive-Pfad der Ergebnisdatei auf Colab, sonst None."""
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/test_results.xlsx"
    return None

def test_logger(header, results):
    drive_path = _drive_results_path()
    # Basis-Workbook: Drive (akkumuliert ueber mehrere Colab-Runs) > lokal > neu
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
    ws.append(["'" + "=" * 80])  # separator line

    for path in ["test_results.xlsx"] + ([drive_path] if drive_path else []):
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            wb.save(path)
            print("Results saved to", path)
        except Exception as e:
            print("Failed to save", path, ":", e)

def _compute_sample_weights(df, state, prop_weight_mode, corrected_weights):
    """Per-case weights for distance-decayed propagation labels."""
    if prop_weight_mode == "uniform" or state["nearest_dist"] is None:
        return None
    covered_mask = state["covered_arr"]
    nearest_dist = state["nearest_dist"]
    t = state["T"]
    if prop_weight_mode == "linear_decay":
        factors = np.maximum(0.0, 1.0 - nearest_dist / t)
    elif prop_weight_mode == "gaussian_decay":
        sigma = t / 2.0
        factors = np.exp(-(nearest_dist ** 2) / (2 * sigma ** 2))
    else:
        factors = np.ones(len(df))
    sample_weights = np.ones(len(df))
    sample_weights[covered_mask] = corrected_weights * factors[covered_mask]
    # Direct oracle corrections always get full weight
    for pid in state["directly_corrected"]:
        idx = np.where(state["X_ids"] == pid)[0][0]
        sample_weights[idx] = corrected_weights
    return sample_weights

def retrain_catboost(df, l=10, corrected_weights=100, corrected_saved=True, strategy="entropy",
                     greedy_batching=False, greedy_T=None, compute_shap=False, skip_retrain_on_skip=False,
                     propagation_space=None, initial_shap=None,
                     prop_weight_mode="uniform", selection_mode="uncertainty",
                     early_stop=False, chunk_size=50, min_delta=0.001, stop_patience=2,
                     seed=42, run=None):
    if greedy_batching:
        state = new_state(greedy_T, propagation_space=propagation_space)
        # Bootstrap SHAP from initial model so first iteration can use SHAP spaces
        if initial_shap is not None:
            state["shap_vals"] = initial_shap
    else:
        state = {"corrected": []}

    novelty_cache = novelty_scores(df) if strategy == "novelty" else None

    results = []
    target_iters = {1, l//2 + 1, l}
    cat_importances = None
    cat_model = None
    precision = recall = 0.0
    tn = fp = fn = tp = 0
    trees_this = 0

    for i in range(l):
        print(f"\n=== Iteration {i+1} ===")
        t_iter = time.time()

        t_q = time.time()
        unc_rec = _uncertainty_distribution(df, strategy, novelty_cache)

        if greedy_batching:
            df, state, meta = greedy_iteration(df, strategy, state, selection_mode=selection_mode)
            if meta.get("type") == "no_candidates":
                print("No more candidates in pool.")
                break
            corrected_ids = list(state["directly_corrected"] | state["covered"])
            skip_retrain = skip_retrain_on_skip and meta.get("type") == "skip"
        else:
            uncertain_df = uncertainty_query(df, strategy, exclude_posting_ids=state["corrected"],
                                             novelty_scores=novelty_cache)
            meta = {}
            mask = df['posting_id'].isin(uncertain_df['posting_id'])
            meta["misspredicted"] = (df.loc[mask, 'label'] != df.loc[mask, 'pred_label']).sum()
            df.loc[mask, 'pred_label'] = df.loc[mask, 'label']
            if corrected_saved:
                state["corrected"].extend(uncertain_df['posting_id'].tolist())
            corrected_ids = state["corrected"]
            skip_retrain = False
        query_s = time.time() - t_q

        if skip_retrain:
            print(f"Skip-Retrain (keine Labeländerung) | "
                  f"Precision: {precision:.4f}, Recall: {recall:.4f}, Trees: {trees_this}"
                  f", dur {time.time()-t_iter:.1f}s (query {query_s:.1f}s, train 0.0s)")
            train_s = 0.0
            es = None
        else:
            sample_weights = _compute_sample_weights(df, state, prop_weight_mode, corrected_weights) if greedy_batching else None

            t_tr = time.time()
            df, cat_importances, precision, recall, cat_model, tn, fp, fn, tp, shap_vals, es = train_catboost(
                df, corrected_ids=corrected_ids, corrected_weights=corrected_weights,
                sample_weights=sample_weights, compute_shap=compute_shap,
                early_stop=early_stop, chunk_size=chunk_size, min_delta=min_delta,
                stop_patience=stop_patience, seed=seed
            )
            train_s = time.time() - t_tr
            trees_this = cat_model.tree_count_
            # Store SHAP in state so greedy propagation can use it
            if greedy_batching:
                state["shap_vals"] = shap_vals
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Trees (this iter): {trees_this}"
                  f", dur {time.time()-t_iter:.1f}s (query {query_s:.1f}s, train {train_s:.1f}s)")
            if greedy_batching:
                print(f"type={meta.get('type')}, centers={meta.get('n_centers')}, "
                      f"covered={meta.get('n_covered')}, flipped={meta.get('n_flipped')}, "
                      f"prop_dims={meta.get('prop_dims')}, "
                      f"M_density={meta.get('M_density', 0):.0f}")
            else:
                print(f"Misspredicted: {meta.get('misspredicted')}")

        if run is not None:
            try:
                auc_val = float(roc_auc_score(df["label"].astype(int), df["pred_score"]))
            except ValueError:
                auc_val = None
            if greedy_batching:
                greedy_meta = {
                    "type": meta.get("type"), "n_centers": meta.get("n_centers"),
                    "n_covered": meta.get("n_covered"), "cumulative_direct": meta.get("cumulative_direct"),
                    "n_flipped": meta.get("n_flipped"), "propagation_accuracy": meta.get("propagation_accuracy"),
                    "prop_dims": meta.get("prop_dims"), "M_density": meta.get("M_density"),
                    "T": state.get("T"), "selection_mode": selection_mode,
                }
                corrected_so_far = meta.get("cumulative_direct")
            else:
                greedy_meta = None
                corrected_so_far = len(state["corrected"])
            run.write("iter", run_id=run.run_id,
                      iteration=i + 1,
                      mode="skip" if skip_retrain else "retrain",
                      precision=precision, recall=recall,
                      trees_this=trees_this,
                      model_trees=cat_model.tree_count_ if cat_model is not None else None,
                      tn=tn, fp=fp, fn=fn, tp=tp, auc=auc_val,
                      misspredicted=meta.get("misspredicted") if not greedy_batching else None,
                      corrected_so_far=corrected_so_far,
                      query_s=query_s, train_s=train_s, dur_s=time.time()-t_iter,
                      uncertainty=unc_rec, greedy=greedy_meta,
                      early_stop=es)
            if es is not None:
                for c in es.get("chunk_curve", []):
                    run.write("chunk", run_id=run.run_id, iteration=i + 1, **c)

        if (i+1) in target_iters:
            base = (f"Iteration: {i+1}, Precision: {precision:.4f}, Recall: {recall:.4f}, Trees: {trees_this}, "
                    f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
            if skip_retrain:
                extra = ", type=skip (retrain skipped)"
            elif greedy_batching:
                extra = (f", type={meta.get('type')}, centers={meta.get('n_centers')}, "
                         f"covered={meta.get('n_covered')}, cum_direct={meta.get('cumulative_direct')}, "
                         f"flipped={meta.get('n_flipped')}, prop_acc={meta.get('propagation_accuracy'):.4f}, "
                         f"T={state['T']:.4f}, prop_dims={meta.get('prop_dims')}, "
                         f"weight_mode={prop_weight_mode}, sel_mode={selection_mode}")
            else:
                extra = f", Misspredicted: {meta.get('misspredicted')}"
            results.append(base + extra)

    return df, cat_importances, precision, recall, results

def incremental_catboost(df, l=10, corrected_weights=100, corrected_saved=True, strategy="entropy",
                         greedy_batching=False, greedy_T=None, compute_shap=False, skip_retrain_on_skip=False,
                         propagation_space=None, initial_shap=None,
                         prop_weight_mode="uniform", selection_mode="uncertainty",
                         early_stop=False, chunk_size=50, min_delta=0.001, stop_patience=2,
                         seed=42, run=None):
    """Like retrain_catboost, but continues the previous model via init_model
    instead of a full 500-tree retrain (warm start)."""
    if greedy_batching:
        state = new_state(greedy_T, propagation_space=propagation_space)
        # Bootstrap SHAP from initial model so first iteration can use SHAP spaces
        if initial_shap is not None:
            state["shap_vals"] = initial_shap
    else:
        state = {"corrected": []}

    novelty_cache = novelty_scores(df) if strategy == "novelty" else None

    results = []
    target_iters = {1, l//2 + 1, l}
    cat_importances = None
    cat_model = None
    precision = recall = 0.0
    tn = fp = fn = tp = 0
    prev_trees = 0
    trees_this = 0

    for i in range(l):
        print(f"\n=== Incremental Iteration {i+1} ===")
        t_iter = time.time()

        t_q = time.time()
        unc_rec = _uncertainty_distribution(df, strategy, novelty_cache)

        if greedy_batching:
            df, state, meta = greedy_iteration(df, strategy, state, selection_mode=selection_mode)
            if meta.get("type") == "no_candidates":
                print("No more candidates in pool.")
                break
            corrected_ids = list(state["directly_corrected"] | state["covered"])
            skip_retrain = skip_retrain_on_skip and meta.get("type") == "skip"
        else:
            uncertain_df = uncertainty_query(df, strategy, exclude_posting_ids=state["corrected"],
                                             novelty_scores=novelty_cache)
            meta = {}
            mask = df['posting_id'].isin(uncertain_df['posting_id'])
            meta["misspredicted"] = (df.loc[mask, 'label'] != df.loc[mask, 'pred_label']).sum()
            df.loc[mask, 'pred_label'] = df.loc[mask, 'label']
            if corrected_saved:
                state["corrected"].extend(uncertain_df['posting_id'].tolist())
            corrected_ids = state["corrected"]
            skip_retrain = False
        query_s = time.time() - t_q

        if skip_retrain:
            print(f"Skip-Retrain (keine Labeländerung) | "
                  f"Precision: {precision:.4f}, Recall: {recall:.4f}, Trees: {trees_this}"
                  f", dur {time.time()-t_iter:.1f}s (query {query_s:.1f}s, train 0.0s)")
            train_s = 0.0
            es = None
        else:
            sample_weights = _compute_sample_weights(df, state, prop_weight_mode, corrected_weights) if greedy_batching else None

            if cat_model is not None:
                prev_trees = cat_model.tree_count_
            t_tr = time.time()
            df, cat_importances, precision, recall, cat_model, tn, fp, fn, tp, shap_vals, es = train_catboost(
                df, incremental=True, inc_model=cat_model, full_data=df,
                corrected_ids=corrected_ids, corrected_weights=corrected_weights,
                sample_weights=sample_weights, compute_shap=compute_shap,
                early_stop=early_stop, chunk_size=chunk_size, min_delta=min_delta,
                stop_patience=stop_patience, seed=seed
            )
            train_s = time.time() - t_tr
            trees_this = cat_model.tree_count_ - prev_trees
            # Store SHAP in state so greedy propagation can use it
            if greedy_batching:
                state["shap_vals"] = shap_vals
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Trees (this iter): {trees_this}"
                  f", dur {time.time()-t_iter:.1f}s (query {query_s:.1f}s, train {train_s:.1f}s)")
            if greedy_batching:
                print(f"type={meta.get('type')}, centers={meta.get('n_centers')}, "
                      f"covered={meta.get('n_covered')}, flipped={meta.get('n_flipped')}, "
                      f"prop_dims={meta.get('prop_dims')}, "
                      f"M_density={meta.get('M_density', 0):.0f}")
            else:
                print(f"Misspredicted: {meta.get('misspredicted')}")

        if run is not None:
            try:
                auc_val = float(roc_auc_score(df["label"].astype(int), df["pred_score"]))
            except ValueError:
                auc_val = None
            if greedy_batching:
                greedy_meta = {
                    "type": meta.get("type"), "n_centers": meta.get("n_centers"),
                    "n_covered": meta.get("n_covered"), "cumulative_direct": meta.get("cumulative_direct"),
                    "n_flipped": meta.get("n_flipped"), "propagation_accuracy": meta.get("propagation_accuracy"),
                    "prop_dims": meta.get("prop_dims"), "M_density": meta.get("M_density"),
                    "T": state.get("T"), "selection_mode": selection_mode,
                }
                corrected_so_far = meta.get("cumulative_direct")
            else:
                greedy_meta = None
                corrected_so_far = len(state["corrected"])
            run.write("iter", run_id=run.run_id,
                      iteration=i + 1,
                      mode="skip" if skip_retrain else "incremental",
                      precision=precision, recall=recall,
                      trees_this=trees_this,
                      model_trees=cat_model.tree_count_ if cat_model is not None else None,
                      tn=tn, fp=fp, fn=fn, tp=tp, auc=auc_val,
                      misspredicted=meta.get("misspredicted") if not greedy_batching else None,
                      corrected_so_far=corrected_so_far,
                      query_s=query_s, train_s=train_s, dur_s=time.time()-t_iter,
                      uncertainty=unc_rec, greedy=greedy_meta,
                      early_stop=es)
            if es is not None:
                for c in es.get("chunk_curve", []):
                    run.write("chunk", run_id=run.run_id, iteration=i + 1, **c)

        if (i+1) in target_iters:
            base = (f"Iteration: {i+1}, Precision: {precision:.4f}, Recall: {recall:.4f}, Trees: {trees_this}, "
                    f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
            if skip_retrain:
                extra = ", type=skip (retrain skipped)"
            elif greedy_batching:
                extra = (f", type={meta.get('type')}, centers={meta.get('n_centers')}, "
                         f"covered={meta.get('n_covered')}, cum_direct={meta.get('cumulative_direct')}, "
                         f"flipped={meta.get('n_flipped')}, prop_acc={meta.get('propagation_accuracy'):.4f}, "
                         f"T={state['T']:.4f}, prop_dims={meta.get('prop_dims')}, "
                         f"weight_mode={prop_weight_mode}, sel_mode={selection_mode}")
            else:
                extra = f", Misspredicted: {meta.get('misspredicted')}"
            results.append(base + extra)

    return df, cat_importances, precision, recall, cat_model, results

def replace_posting(df, posting_id, replacement):
    df.loc[df['posting_id'] == posting_id, 'pred_label'] = replacement
    return df

def run_unsupervised(compute_shap=False, early_stop=False, chunk_size=50, min_delta=0.001, stop_patience=2, seed=42, run=None):
    df, labels = full_dataprep()
    df_if = run_if(df, labels, verbose=False)
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

def run_supervised(training_strat= 'retrain', l=10, corrected_weights=100, corrected_saved=True, strategy="entropy", return_full_data=False, greedy_batching=False, greedy_T=None, compute_shap=False, skip_retrain_on_skip=False, propagation_space=None, prop_weight_mode="uniform", selection_mode="uncertainty", early_stop=False, chunk_size=50, min_delta=0.001, stop_patience=2, seed=42):
    needs_shap = any(s.startswith("shap") for s in propagation_space) if propagation_space else False
    config = dict(timestamp=datetime.now().isoformat(timespec="seconds"),
                  training_strat=training_strat, l=l, corrected_weights=corrected_weights,
                  corrected_saved=corrected_saved, strategy=strategy,
                  greedy_batching=greedy_batching, greedy_T=greedy_T, compute_shap=needs_shap,
                  skip_retrain_on_skip=skip_retrain_on_skip, propagation_space=propagation_space,
                  prop_weight_mode=prop_weight_mode, selection_mode=selection_mode,
                  early_stop=early_stop, chunk_size=chunk_size, min_delta=min_delta,
                  stop_patience=stop_patience, seed=seed)
    run = logstore.open_run(config)
    t0 = time.time()
    try:
        df, cat_importances, precision, recall, cat_model, initial_shap = run_unsupervised(
            compute_shap=needs_shap, early_stop=early_stop, chunk_size=chunk_size,
            min_delta=min_delta, stop_patience=stop_patience, seed=seed, run=run)
        if training_strat == 'incremental':
            df, cat_importances, precision, recall, cat_model, results = incremental_catboost(
                df, l, corrected_weights, corrected_saved, strategy,
                greedy_batching=greedy_batching, greedy_T=greedy_T,
                compute_shap=needs_shap, skip_retrain_on_skip=skip_retrain_on_skip,
                propagation_space=propagation_space, initial_shap=initial_shap,
                prop_weight_mode=prop_weight_mode, selection_mode=selection_mode,
                early_stop=early_stop, chunk_size=chunk_size, min_delta=min_delta,
                stop_patience=stop_patience, seed=seed, run=run)
        else:
            df, cat_importances, precision, recall, results = retrain_catboost(
                df, l, corrected_weights, corrected_saved, strategy,
                greedy_batching=greedy_batching, greedy_T=greedy_T,
                compute_shap=needs_shap, skip_retrain_on_skip=skip_retrain_on_skip,
                propagation_space=propagation_space, initial_shap=initial_shap,
                prop_weight_mode=prop_weight_mode, selection_mode=selection_mode,
                early_stop=early_stop, chunk_size=chunk_size, min_delta=min_delta,
                stop_patience=stop_patience, seed=seed, run=run)
        run.close(wall_time_s=time.time() - t0, final_precision=precision, final_recall=recall)
        printed_run_id = run.run_id
        test_logger(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] | run_id= {printed_run_id} | training_strat= {training_strat}, l={l}, corrected_weights = {corrected_weights}, corrected_saved = {corrected_saved}, strategy = {strategy}, greedy_batching = {greedy_batching}, greedy_T = {greedy_T}, compute_shap = {compute_shap}, skip_retrain_on_skip = {skip_retrain_on_skip}, propagation_space = {propagation_space}, prop_weight_mode = {prop_weight_mode}, selection_mode = {selection_mode}, early_stop = {early_stop}, chunk_size = {chunk_size}, min_delta = {min_delta}, stop_patience = {stop_patience}, seed = {seed}", results)
    except Exception:
        run.close(wall_time_s=time.time() - t0)
        raise
    return df, cat_importances, precision, recall, cat_model

if __name__ == "__main__":

    
    #df, cat_importances, precision, recall, cat_model = run_supervised(training_strat='retrain', l=500, corrected_weights=100, corrected_saved=True, strategy="margin", return_full_data=False, greedy_batching=True, greedy_T=0.5, propagation_space=["shap_raw"], prop_weight_mode="uniform", selection_mode="uncertainty")
    
    #df, cat_importances, precision, recall, cat_model = run_supervised(training_strat='retrain', l=500, corrected_weights=100, corrected_saved=True, strategy="margin", return_full_data=False, greedy_batching=True, greedy_T=0.5, propagation_space=["shap_raw"], prop_weight_mode="linear_decay", selection_mode="uncertainty")
    
    #df, cat_importances, precision, recall, cat_model = run_supervised(training_strat='retrain', l=500, corrected_weights=100, corrected_saved=True, strategy="margin", return_full_data=False, greedy_batching=True, greedy_T=0.5, propagation_space=["shap_raw"], prop_weight_mode="uniform", selection_mode="uncertainty_density")
    
    #df, cat_importances, precision, recall, cat_model = run_supervised(training_strat='retrain', l=500, corrected_weights=100, corrected_saved=True, strategy="margin", return_full_data=False, greedy_batching=True, greedy_T=0.5, propagation_space=["shap_raw"], prop_weight_mode="linear_decay")
    #df, cat_importances, precision, recall, cat_model = run_supervised(training_strat='retrain', l=500, corrected_weights=100, corrected_saved=True, strategy="margin", return_full_data=False, greedy_batching=True, greedy_T=0.5, propagation_space=["shap_raw"], selection_mode="uncertainty_density")
    #df, cat_importances, precision, recall, cat_model = run_supervised(training_strat='retrain', l=500, corrected_weights=100, corrected_saved=True, strategy="margin", return_full_data=False, greedy_batching=True, greedy_T=0.5, propagation_space=["shap_raw"], prop_weight_mode="linear_decay", selection_mode="uncertainty_density")
    #df, cat_importances, precision, recall, cat_model = run_supervised(training_strat='incremental', l=100, corrected_weights=100, corrected_saved=True, strategy="margin", return_full_data=False, greedy_batching=True, greedy_T=0.5, propagation_space=["shap_raw"])
    
    # ========================================================================
    # TO TEST: Early-Stop (Chunked Training) - noch nicht getestete Varianten
    # ========================================================================
    # Fokus: full retrain. Staerkste Variante = greedy_batching + shap_raw.
    # Dazu eine non-greedy Variante als reiner Laufzeit-Vergleich.
    # chunk_size=50 und stop_patience=2 sind fix; nur min_delta variiert.
    # - STRICT:   min_delta=0.01   -> Training stoppt sehr schnell (~150-300 Baeume)
    # - MODERATE: min_delta=0.001  -> laeuft haeufig bis ans Cap von 500 Baeumen

    # --- greedy + shap_raw, Baseline OFF (Referenz/Wandzeit) ---
    df, cat_importances, precision, recall, cat_model = run_supervised(training_strat='retrain', l=30, corrected_weights=100, corrected_saved=True, strategy="margin", greedy_batching=True, greedy_T=0.5, propagation_space=["shap_raw"])

    # --- greedy + shap_raw, early_stop STRICT ---
    df, cat_importances, precision, recall, cat_model = run_supervised(training_strat='retrain', l=30, corrected_weights=100, corrected_saved=True, strategy="margin", greedy_batching=True, greedy_T=0.5, propagation_space=["shap_raw"], early_stop=True, chunk_size=50, min_delta=0.01, stop_patience=2)

    # --- greedy + shap_raw, early_stop MODERATE ---
    df, cat_importances, precision, recall, cat_model = run_supervised(training_strat='retrain', l=30, corrected_weights=100, corrected_saved=True, strategy="margin", greedy_batching=True, greedy_T=0.5, propagation_space=["shap_raw"], early_stop=True, chunk_size=50, min_delta=0.001, stop_patience=2)

    # --- non-greedy (uncertainty) Baseline OFF, reiner Laufzeit-Vergleich ---
    df, cat_importances, precision, recall, cat_model = run_supervised(training_strat='retrain', l=30, corrected_weights=100, corrected_saved=True, strategy="margin")

    # --- non-greedy, early_stop STRICT ---
    df, cat_importances, precision, recall, cat_model = run_supervised(training_strat='retrain', l=30, corrected_weights=100, corrected_saved=True, strategy="margin", early_stop=True, chunk_size=50, min_delta=0.01, stop_patience=2)

    # --- non-greedy, early_stop MODERATE ---
    df, cat_importances, precision, recall, cat_model = run_supervised(training_strat='retrain', l=30, corrected_weights=100, corrected_saved=True, strategy="margin", early_stop=True, chunk_size=50, min_delta=0.001, stop_patience=2)
    
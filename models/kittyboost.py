from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report

CORRECTED_WEIGHT = 100.0


def fit_split_model(
    frame,
    trusted_ids,
    trusted_targets,
    review_labels,
    *,
    extra_ids=None,
    extra_targets=None,
    class_weight_anomaly=None,
    feature_weights=None,
    seed=42,
):
    """Fit split-training targets using canonical experiment CatBoost settings."""
    extra_ids = extra_ids or set()
    extra_targets = extra_targets or {pid: 0 for pid in extra_ids}
    targets = dict(trusted_targets)
    targets.update(extra_targets)
    targets.update(review_labels)
    fit_ids = set(targets)
    fit = frame.loc[frame["posting_id"].isin(fit_ids)].copy()
    fit["pred_label"] = fit["posting_id"].map(targets).astype(int)
    y_fit = fit["pred_label"].to_numpy(dtype=int)
    positives = int(y_fit.sum())
    if positives == 0 or positives == len(y_fit):
        raise ValueError("Fit data must contain both target classes.")
    anomaly_weight = (
        float(class_weight_anomaly)
        if class_weight_anomaly is not None
        else max(5.0, len(y_fit) / positives)
    )
    sample_weight = np.ones(len(fit), dtype=float)
    sample_weight[fit["posting_id"].isin(review_labels).to_numpy()] = CORRECTED_WEIGHT
    model_kwargs = dict(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        random_seed=seed,
        verbose=False,
        class_weights={0: 1.0, 1: anomaly_weight},
        allow_writing_files=False,
    )
    if feature_weights is not None:
        n_features = feature_matrix(fit).shape[1]
        if len(feature_weights) != n_features:
            raise ValueError(
                f"Expected {n_features} CatBoost feature weights, got {len(feature_weights)}"
            )
        model_kwargs["feature_weights"] = feature_weights
    model = CatBoostClassifier(**model_kwargs)
    model.fit(Pool(feature_matrix(fit), label=y_fit, weight=sample_weight))
    return model


def feature_matrix(frame):
    return frame.drop(columns=["label", "posting_id", "pred_label", "pred_score"])


def compute_path_groups(model, df):
    """Return a group ID for rows that share the same leaf path across all trees.

    Rows on the same path take the same splits and receive the same prediction.
    The returned array follows the row order in ``df``.
    """
    X = df.drop(columns=["label", "posting_id", "pred_label", "pred_score"])
    leaf_idx = np.asarray(model.calc_leaf_indexes(Pool(X)))
    cont = np.ascontiguousarray(leaf_idx)
    key = cont.view(np.dtype((np.void, cont.dtype.itemsize * cont.shape[1]))).ravel()
    _, inverse = np.unique(key, return_inverse=True)
    return inverse


def train_catboost(df, verbose = False, incremental = False, inc_model = None, full_data = None, corrected_ids = None,corrected_weights=100, compute_shap=False, sample_weights=None, incremental_iters=20, early_stop=False, chunk_size=50, min_delta=0.001, stop_patience=2, seed=42, fit_ids=None, fit_targets=None):
    #fit on assigned labels and evaluate against stored labels
    full_frame = df
    if fit_ids is None:
        fit_frame = df
    else:
        if incremental:
            raise ValueError("fit_ids is supported only for full retraining")
        fit_mask = df["posting_id"].isin(fit_ids)
        fit_frame = df.loc[fit_mask].copy()
        if fit_targets is not None:
            fit_frame["pred_label"] = fit_frame["posting_id"].map(fit_targets).astype(int)
        if fit_frame.empty:
            raise ValueError("fit_ids selected no CatBoost training rows")

    y_train_target = fit_frame["pred_label"].astype(int)  #fit on assigned targets
    y_true = fit_frame["label"].astype(int)  #evaluate against stored labels
    if fit_ids is not None and y_train_target.nunique() < 2:
        raise ValueError("Fit data must contain both target classes.")

    X = fit_frame.drop(columns=["label", "posting_id", "pred_label", "pred_score"])

    weight_normal = 1.0
    #avoid division by zero when no anomaly targets exist
    anomaly_count = y_train_target.sum()
    if anomaly_count > 0:
        weight_anomaly = max(5, (len(y_train_target) / anomaly_count))
    else:
        weight_anomaly = 5.0  #use minimum anomaly weight when none are present

    its = 500
    if incremental:
        its = incremental_iters

    model_kwargs = dict(
        iterations=its,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        #task_type='GPU',  # flip manually for Colab
        random_seed=seed,
        verbose=False,
    )

    if incremental:
        #apply class imbalance through sample weights when warm-starting
        if sample_weights is not None:
            weights = np.asarray(sample_weights).copy()
            if fit_ids is not None and len(weights) == len(full_frame):
                weights = weights[fit_mask.to_numpy()]
        elif corrected_ids is not None:
            weights = np.ones(len(fit_frame))
            weights[fit_frame['posting_id'].isin(corrected_ids)] = corrected_weights
        else:
            weights = np.ones(len(fit_frame))
        weights[y_train_target.values == 1] *= weight_anomaly
        train_pool = Pool(X, y_train_target, weight=weights)
    else:
        #use explicit weights when corrected cases are present
        model_kwargs["class_weights"] = {0: weight_normal, 1: weight_anomaly}
        if corrected_ids is not None:
            if sample_weights is not None:
                weights = np.asarray(sample_weights).copy()
                if fit_ids is not None and len(weights) == len(full_frame):
                    weights = weights[fit_mask.to_numpy()]
            else:
                weights = np.ones(len(fit_frame))
                weights[fit_frame['posting_id'].isin(corrected_ids)] = corrected_weights
            train_pool = Pool(X, y_train_target, weight=weights)
        else:
            train_pool = Pool(X, y_train_target)

    reason = "cap"
    chunk_curve = []
    if early_stop:
        #stop when predictions stabilize across consecutive chunks
        rng = np.random.RandomState(seed)
        sample_idx = rng.choice(len(fit_frame), size=min(20000, len(fit_frame)), replace=False)
        X_all = fit_frame.drop(columns=["label", "posting_id", "pred_label", "pred_score"]).reset_index(drop=True)
        X_sample = X_all.iloc[sample_idx]
        model = None
        prev_pred = None
        stalls = 0
        total = 0
        while total < its:
            n = min(chunk_size, its - total)
            chunk_kwargs = dict(model_kwargs)
            chunk_kwargs["iterations"] = n
            chunk_model = CatBoostClassifier(**chunk_kwargs)
            chunk_model.fit(train_pool, init_model=model if model is not None else (inc_model if incremental else None))
            total += n
            pred = chunk_model.predict_proba(X_sample)[:, 1]
            if prev_pred is None:
                delta = float("inf")
            else:
                delta = float(np.mean(np.abs(pred - prev_pred)))
            prev_pred = pred
            if delta < min_delta:
                stalls += 1
            else:
                stalls = 0
            print(f"[chunk] trees={total}, delta={delta:.6f}, stalls={stalls}")
            chunk_curve.append({"trees": total, "delta": delta, "stalls": stalls})
            model = chunk_model
            if stalls >= stop_patience:
                reason = "plateau"
                print(f"Early stopping after {total} trees (mean |Δprediction| < {min_delta})")
                break
    else:
        model = CatBoostClassifier(**model_kwargs)
        model.fit(train_pool, init_model=inc_model if incremental else None)

    #evaluate on the same rows used for final predictions
    if incremental and full_data is not None:
        X_full = full_data.drop(columns=["label", "posting_id", "pred_label", "pred_score"])
        full_pool = Pool(X_full)
        preds = model.predict_proba(full_pool)[:, 1]
        preds_class = model.predict(full_pool)
        y_true_full = full_data["label"].astype(int)
        precision = precision_score(y_true_full, preds_class)
        recall = recall_score(y_true_full, preds_class)
        full_data['pred_score'] = preds
        full_data['pred_label'] = preds_class
        eval_y = y_true_full
    elif fit_ids is not None:
        X_full = full_frame.drop(columns=["label", "posting_id", "pred_label", "pred_score"])
        prediction_pool = Pool(X_full)
        preds = model.predict_proba(prediction_pool)[:, 1]
        preds_class = model.predict(prediction_pool)
        y_true_full = full_frame["label"].astype(int)
        precision = precision_score(y_true_full, preds_class)
        recall = recall_score(y_true_full, preds_class)
        eval_y = y_true_full
    else:
        preds = model.predict_proba(train_pool)[:, 1]
        preds_class = model.predict(train_pool)
        precision = precision_score(y_true, preds_class)
        recall = recall_score(y_true, preds_class)
        eval_y = y_true

    thresholds = np.linspace(0.01, 0.99, 200)
    f1_scores = [f1_score(eval_y, (preds > t).astype(int), zero_division=0) for t in thresholds]
    best_t = thresholds[np.argmax(f1_scores)]

    df_pred = (preds > best_t).astype(int)

    if compute_shap:
        shap_pool = prediction_pool if fit_ids is not None else train_pool
        shap_vals = model.get_feature_importance(shap_pool, type="ShapValues")
        shap_vals = shap_vals[:, :-1]
    else:
        shap_vals = None

    if verbose:
        print("=== CatBoost: trained on assigned labels, evaluated on stored labels ===")
        print("ROC-AUC:", roc_auc_score(eval_y, preds))
        print(classification_report(eval_y, preds_class, zero_division=0))
        print(f"\nBest F1 threshold: {best_t:.3f}")
        print("Best F1 score:", max(f1_scores))
        print("\n=== Metrics at best F1 threshold ===")
        print(classification_report(eval_y, df_pred, zero_division=0))
        if shap_vals is not None:
            print("\nSHAP:")
            print(shap_vals[:5])
    
    tn, fp, fn, tp = confusion_matrix(eval_y, df_pred).ravel()
    print("\n=== Confusion matrix at best F1 threshold ===")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP):  {tp}")
    #return full_data for incremental runs, otherwise return the input frame
    return_df = full_data if (incremental and full_data is not None) else full_frame
    return_df['pred_score'] = preds
    return_df['pred_label'] = preds_class
    early_stop_info = {
        "active": early_stop,
        "tree_count": model.tree_count_,
        "stop_reason": reason if early_stop else None,
        "chunk_curve": chunk_curve,
    }
    return return_df, model.get_feature_importance(), precision, recall, model, tn, fp, fn, tp, shap_vals, early_stop_info

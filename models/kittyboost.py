from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report


def train_catboost(df, verbose = False, incremental = False, inc_model = None, full_data = None, corrected_ids = None,corrected_weights=100, compute_shap=False, sample_weights=None, incremental_iters=20):
    # Trainiere auf generated_label, evaluiere auf true label
    y_train_target = df["pred_label"].astype(int)  # Trainiere auf Rules
    y_true = df["label"].astype(int)  # Evaluiere auf echte Labels

    X = df.drop(columns=["label", "posting_id", "pred_label", "pred_score"])

    weight_normal = 1.0
    # Avoid divide by zero when no anomalies exist in training data
    anomaly_count = y_train_target.sum()
    if anomaly_count > 0:
        weight_anomaly = max(5, (len(y_train_target) / anomaly_count))
    else:
        weight_anomaly = 5.0  # Default weight if no anomalies

    its = 500
    if incremental:
        its = incremental_iters

    model_kwargs = dict(
        iterations=its,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        #task_type='GPU',  # flip manually for Colab
        random_seed=42,
        verbose=False,
    )

    if incremental:
        # class_weights can't be combined with init_model -> fold imbalance into per-sample weights
        if sample_weights is not None:
            weights = sample_weights.copy()
        elif corrected_ids is not None:
            weights = np.ones(len(df))
            weights[df['posting_id'].isin(corrected_ids)] = corrected_weights
        else:
            weights = np.ones(len(df))
        weights[y_train_target.values == 1] *= weight_anomaly
        train_pool = Pool(X, y_train_target, weight=weights)
    else:
        #das wurde geändert wegen error
        model_kwargs["class_weights"] = {0: weight_normal, 1: weight_anomaly}
        if corrected_ids is not None:
            if sample_weights is not None:
                weights = sample_weights
            else:
                weights = np.ones(len(df))
                weights[df['posting_id'].isin(corrected_ids)] = corrected_weights
            train_pool = Pool(X, y_train_target, weight=weights)
        else:
            train_pool = Pool(X, y_train_target)

    model = CatBoostClassifier(**model_kwargs)
    model.fit(train_pool, init_model=inc_model if incremental else None)

    # eval metrics always on the same data the final predictions are computed on
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
        shap_vals = model.get_feature_importance(train_pool, type="ShapValues")
        shap_vals = shap_vals[:, :-1]
    else:
        shap_vals = None

    if verbose:
        print("=== CATBOOST (Trainiert auf generated_label, evaluiert auf true label) ===")
        print("AUC:", roc_auc_score(eval_y, preds))
        print(classification_report(eval_y, preds_class, zero_division=0))
        print(f"\nbest F1: {best_t:.3f}")
        print("F1 @ best threshold:", max(f1_scores))
        print("\n=== METRICS (optimierter Threshold) ===")
        print(classification_report(eval_y, df_pred, zero_division=0))
        if shap_vals is not None:
            print("\nSHAP:")
            print(shap_vals[:5])
    
    tn, fp, fn, tp = confusion_matrix(eval_y, df_pred).ravel()
    print("\n=== Confusion Matrix (optimierter Threshold) ===")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP):  {tp}")
    # Return full_data wenn incremental mit full_data, sonst df
    return_df = full_data if (incremental and full_data is not None) else df
    return_df['pred_score'] = preds
    return_df['pred_label'] = preds_class
    return return_df, model.get_feature_importance(), precision, recall, model, tn, fp, fn, tp, shap_vals

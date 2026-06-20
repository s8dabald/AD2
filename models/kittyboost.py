from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report


def train_catboost(df, verbose = False, incremental = False, inc_model = None, full_data = None, corrected_ids = None,corrected_weights=100):
    
    if corrected_ids is not None:
        weights = np.ones(len(df))
        weights[df['posting_id'].isin(corrected_ids)] = corrected_weights  # Erhöhe Gewicht für korrigierte Samples
    # Trainiere auf generated_label, evaluiere auf true label
    y_train_target = df["pred_label"].astype(int)  # Trainiere auf Rules
    y_true = df["label"].astype(int)  # Evaluiere auf echte Labels

    X = df.drop(columns=["label", "posting_id", "pred_label", "pred_score"])
    #print(X.columns)
    # FULL TRAINING - kein split
    if corrected_ids is not None:
        train_pool = Pool(X, y_train_target, weight=weights)
    else:
        train_pool = Pool(X, y_train_target)

    weight_normal = 1.0
    # Avoid divide by zero when no anomalies exist in training data
    anomaly_count = y_train_target.sum()
    if anomaly_count > 0:
        weight_anomaly = max(5, (len(y_train_target) / anomaly_count))
    else:
        weight_anomaly = 5.0  # Default weight if no anomalies

    its = 500
    if incremental:
        its = 20

    model = CatBoostClassifier(
        iterations=its,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        class_weights={0: weight_normal, 1: weight_anomaly} #das wurde geändert wegen error
    )

    model.fit(train_pool, init_model=inc_model if incremental else None)

    # Für incremental training mit full_data: predictions auf kompletten Datensatz
    if incremental and full_data is not None:
        X_full = full_data.drop(columns=["label", "posting_id", "pred_label", "pred_score"])
        full_pool = Pool(X_full)
        preds = model.predict_proba(full_pool)[:, 1]
        preds_class = model.predict(full_pool)
        y_true_full = full_data["label"].astype(int)
        precision = precision_score(y_true_full, preds_class)
        recall = recall_score(y_true_full, preds_class)
        # Auch den full_data updaten mit predictions
        full_data['pred_score'] = preds
        full_data['pred_label'] = preds_class
    else:
        preds = model.predict_proba(train_pool)[:, 1]
        preds_class = model.predict(train_pool)
        precision = precision_score(y_true, preds_class)
        recall = recall_score(y_true, preds_class)

    thresholds = np.linspace(0.01, 0.99, 200)
    f1_scores = [f1_score(y_true, (preds > t).astype(int), zero_division=0) for t in thresholds]
    best_t = thresholds[np.argmax(f1_scores)]


    df_pred = (preds > best_t).astype(int)

    shap_vals = model.get_feature_importance(train_pool, type="ShapValues")

    if verbose:
        print("=== CATBOOST (Trainiert auf generated_label, evaluiert auf true label) ===")
        print("AUC:", roc_auc_score(y_true, preds))
        print(classification_report(y_true, preds_class, zero_division=0))
        print(f"\nbest F1: {best_t:.3f}")
        print("F1 @ best threshold:", max(f1_scores))
        print("\n=== METRICS (optimierter Threshold) ===")
        print(classification_report(y_true, df_pred, zero_division=0))
        print("\nSHAP:")
        print(shap_vals[:5])
    
    tn, fp, fn, tp = confusion_matrix(y_true, df_pred).ravel()
    print("\n=== Confusion Matrix (optimierter Threshold) ===")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP):  {tp}")
    # Return full_data wenn incremental mit full_data, sonst df
    return_df = full_data if (incremental and full_data is not None) else df
    return_df['pred_score'] = preds
    return_df['pred_label'] = preds_class
    return return_df, model.get_feature_importance(), precision, recall, model, tn, fp, fn, tp

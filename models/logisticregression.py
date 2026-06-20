# Logistic Regression (RIVER / true online learning)
from river import linear_model, preprocessing, optim, metrics
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report

def run_river(X, y_gen):
    log_reg = (
        preprocessing.StandardScaler() |
        linear_model.LogisticRegression(
            optimizer=optim.SGD(0.01)
        )
    )
    for idx in X.index:
        x = X.loc[idx].to_dict()
        y = int(y_gen.loc[idx])
        log_reg.learn_one(x, y)

    preds_gen = []
    preds_class_gen = []

    for idx in X.index:
        x = X.loc[idx].to_dict()

        proba = log_reg.predict_proba_one(x).get(1, 0.0)
        pred = int(proba >= 0.5)

        preds_gen.append(proba)
        preds_class_gen.append(pred)

    preds_gen = np.array(preds_gen)
    preds_class_gen = np.array(preds_class_gen)
    return preds_gen, preds_class_gen

def run_lr_sklearn(X, y_gen):
    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        solver='lbfgs'
    )

    log_reg.fit(X, y_gen)
    preds_gen = log_reg.predict_proba(X)[:, 1]
    preds_class_gen = log_reg.predict(X)
    return preds_gen, preds_class_gen

def lr_evaluation(y, preds_gen, preds_class_gen):
    precision_gen = precision_score(y, preds_class_gen)
    f1_gen = f1_score(y, preds_class_gen)
    roc_auc_gen = roc_auc_score(y, preds_gen)
    recall_gen = recall_score(y, preds_class_gen)

    cm_gen = confusion_matrix(y, preds_class_gen)
    tn_gen, fp_gen, fn_gen, tp_gen = cm_gen.ravel()

    print(f"Precision: {precision_gen:.4f}")
    print(f"Recall:    {recall_gen:.4f}")
    print(f"F1-Score:  {f1_gen:.4f}")
    print(f"ROC-AUC:   {roc_auc_gen:.4f}")
    print(f"TP: {tp_gen}, FP: {fp_gen}, TN: {tn_gen}, FN: {fn_gen}")
    print("\nDetailed Report:")
    print(classification_report(y, preds_class_gen))

def run_logistic_regression(df,as_river = True, verbose=False, evaluate_on_true_label = True):
    """Trains a logistic regression model on the generated labels from IF and evaluates it on either the true labels or the IF predicted labels.
    Args:
        df (pd.DataFrame): DataFrame containing the features, true labels, and IF predicted labels.
        as_river (bool): If True, uses the RIVER library for online learning. If False, uses scikit-learn for batch learning.
        verbose (bool): If True, prints detailed evaluation metrics.
        evaluate_on_true_label (bool): If True, evaluates the model on the true labels. If False, evaluates on the IF predicted labels. Defaults to True.
    Returns:
        pd.DataFrame: DataFrame with added columns for logistic regression predictions.
    """
    X = df.drop(columns=["label", "posting_id", "if_label"])

    y_gen = df["if_label"].astype(int)
    y_true = df["label"].astype(int)

    if as_river:
        preds_gen, preds_class_gen = run_river(X, y_gen)
    else:
        preds_gen, preds_class_gen = run_lr_sklearn(X, y_gen)

    print("=== LR trained on generated_label, evaluated on IF prediction labels ===")
    if verbose:
        if evaluate_on_true_label:
            print("\n=== LR trained on generated_label, evaluated on true label ===")
            lr_evaluation(y_true, preds_gen, preds_class_gen)
        else:
            print("\n=== LR trained on generated_label, evaluated on predicted label ===")
            lr_evaluation(y_gen, preds_gen, preds_class_gen)
    df['lr_pred_label'] = preds_class_gen
    df['lr_pred_proba'] = preds_gen
    return df
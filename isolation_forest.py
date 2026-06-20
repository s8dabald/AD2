from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
#load final feature matrix
def run_if(df, labels, verbose=False):
    X_final = df.drop(columns= 'posting_id')

    #setup isolation forest model
    if_model = IsolationForest(
            n_estimators=1000,
            max_samples=400,
            random_state=42,
            n_jobs=-1,

        )

    #fit model on feature space
    if_model.fit(X_final)

    #compute anomaly scores
    scores = if_model.decision_function(X_final)
    #flip sign so higher means more anomalous
    anomaly_scores = -scores

    #choose cutoff based on percentile
    threshold = np.percentile(anomaly_scores, 95)
    #binary prediction from threshold
    pred_labels = (anomaly_scores >= threshold).astype(int)

    #evaluate against labels
    precision = precision_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    roc_auc = roc_auc_score(labels, pred_labels)

    tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()

    # print results
    if verbose:
        print("Isolation Forest Evaluation:")
        print("Precision:", precision)
        print("Recall   :", recall)
        print("F1       :", f1)
        print("ROC-AUC  :", roc_auc)
        print()
        print("TP:", tp)
        print("FP:", fp)
        print("TN:", tn)
        print("FN:", fn)
        print()
        print("Feature vector length:", X_final.shape[1])
        print("Threshold used:", threshold)

    df_if = df.copy()
    df_if['pred_score'] = anomaly_scores
    df_if['pred_label'] = pred_labels
    df_if = df_if.merge(labels.reset_index(), on='posting_id')

    return df_if
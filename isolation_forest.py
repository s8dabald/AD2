from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
#prepare feature matrix
def run_if(df, labels, verbose=False, seed=42):
    X_final = df.drop(columns= 'posting_id')

    #configure isolation forest
    if_model = IsolationForest(
            n_estimators=1000,
            max_samples=400,
            random_state=seed,
            n_jobs=-1,

        )

    #fit on features
    if_model.fit(X_final)

    #score anomalies
    scores = if_model.decision_function(X_final)
    #higher scores mean more anomalous
    anomaly_scores = -scores

    #use the 95th percentile cutoff
    threshold = np.percentile(anomaly_scores, 95)
    #convert scores to labels
    pred_labels = (anomaly_scores >= threshold).astype(int)

    #compare with stored labels
    precision = precision_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    roc_auc = roc_auc_score(labels, pred_labels)

    tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()

    #print metrics
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

    df_if = df.assign(pred_score=anomaly_scores, pred_label=pred_labels)
    df_if = df_if.merge(labels.reset_index(), on='posting_id')

    return df_if

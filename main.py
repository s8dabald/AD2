from dataprep import full_dataprep
import pandas as pd
from isolation_forest import run_if
from models.kittyboost import train_catboost
from querystep import uncertainty_query
from openpyxl import Workbook, load_workbook

def test_logger(header, results):
    try:
        wb = load_workbook("test_results.xlsx")
        ws = wb.active
    except:
        wb = Workbook()
        ws = wb.active
    
    ws.append([header])
    for result_str in results:
        ws.append([result_str])
    ws.append(["'" + "=" * 80])  # separator line
    
    wb.save("test_results.xlsx")
    print("Results saved to test_results.xlsx")

def retrain_catboost(df, l=10, corrected_weights=100, corrected_saved= True, strategy="entropy"):
    """Retrain CatBoost iteratively by correcting uncertain samples based on the specified strategy.
    Args:
        df (pd.DataFrame): Input DataFrame containing 'label', 'pred_label', and 'posting_id' columns.
        l (int): Number of iterations for retraining.
        corrected_weights (int): Weight to assign to corrected samples during training.
        corrected_saved (bool): Whether to save the posting_ids of corrected samples to exclude them in future iterations.
        strategy (str): Strategy for selecting uncertain samples ('entropy', 'margin', 'least_confidence')."""
    corrected = []
    results = []
    target_iters = {1, l//2 + 1, l}

    for i in range(l):
        print(f"\n=== Iteration {i+1} ===")

        # Get uncertain samples (DataFrame with posting_id column), excluding already corrected
        uncertain_df = uncertainty_query(df, strategy, exclude_posting_ids=corrected)
        
        # Vectorized: Replace pred_label with true label for all uncertain samples at once
        mask = df['posting_id'].isin(uncertain_df['posting_id'])
        misspredicted = (df.loc[mask, 'label'] != df.loc[mask, 'pred_label']).sum()
        df.loc[mask, 'pred_label'] = df.loc[mask, 'label']
        
        if corrected_saved:
            corrected.extend(uncertain_df['posting_id'].tolist())

        # Retrain CatBoost on updated labels
        df, cat_importances, precision, recall, cat_model, tn, fp, fn, tp = train_catboost(df, corrected_ids=corrected, corrected_weights=corrected_weights)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"Misspredicted: {misspredicted}")
        
        if (i+1) in target_iters:
            results.append(f"Iteration: {i+1}, Precision: {precision:.4f}, Recall: {recall:.4f}, TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    return df, cat_importances, precision, recall, results

def incremental_catboost(df, l=10, return_full_data=False):
    corrected = []
    cat_model = None
    results = []
    counter = 0
    
    for i in range(l):
        print(f"\n=== Incremental Iteration {i+1} ===")
        
        # Get uncertain samples
        uncertain_df = uncertainty_query(df, strategy="entropy", exclude_posting_ids=corrected)
        
        if uncertain_df.empty:
            print("No more uncertain samples.")
            break
        
        counter += 1
        
        # Vectorized: Replace labels for all uncertain samples at once
        mask = df['posting_id'].isin(uncertain_df['posting_id'])
        misspredicted = (df.loc[mask, 'label'] != df.loc[mask, 'pred_label']).sum()
        df.loc[mask, 'pred_label'] = df.loc[mask, 'label']
        corrected.extend(uncertain_df['posting_id'].tolist())
        
        # Get the updated uncertain_df from df
        uncertain_df_updated = df[df['posting_id'].isin(uncertain_df['posting_id'])].copy()
        
        # Choose training dataset based on return_full_data flag
        train_data = df if return_full_data else uncertain_df_updated
        
        # Train incrementally
        df, cat_importances, precision, recall, cat_model, tn, fp, fn, tp = train_catboost(
            train_data, incremental=True, inc_model=cat_model, full_data=df
        )
        
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"Misspredicted: {misspredicted}")
        
        if counter in {1, l//2 + 1, l}:
            results.append(f"Iteration: {counter}, Precision: {precision:.4f}, Recall: {recall:.4f}, TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    
    return df, cat_importances, precision, recall, cat_model, results

def replace_posting(df, posting_id, replacement):
    df.loc[df['posting_id'] == posting_id, 'pred_label'] = replacement
    return df

def run_unsupervised():
    df, labels = full_dataprep()
    df_if = run_if(df, labels, verbose=False)
    df_cat, cat_importances, precision, recall, cat_model,tn, fp, fn, tp = train_catboost(df_if, verbose=True)
    print(f"\nInitial Precision: {precision:.4f}, Initial Recall: {recall:.4f}")
    return df_cat, cat_importances, precision, recall, cat_model

def run_supervised(training_strat= 'retrain', l=10, corrected_weights=100, corrected_saved=True, strategy="entropy", return_full_data=False):
    df, cat_importances, precision, recall, cat_model = run_unsupervised()
    if training_strat == 'incremental':
        df, cat_importances, precision, recall, cat_model, results = incremental_catboost(df, l, return_full_data)
    else:
        df, cat_importances, precision, recall, results = retrain_catboost(df, l,corrected_weights, corrected_saved, strategy)
    test_logger(f"training_strat= {training_strat}, l={l}, corrected_weights = {corrected_weights}, corrected_saved = {corrected_saved}, strategy = {strategy}", results)
    return df, cat_importances, precision, recall, cat_model

if __name__ == "__main__":
    
    df, cat_importances, precision, recall, cat_model = run_supervised(training_strat='retrain', l=100, corrected_weights=100, corrected_saved=True, strategy="entropy", return_full_data=False)
    df, cat_importances, precision, recall, cat_model = run_supervised(training_strat='retrain', l=50, corrected_weights=100, corrected_saved=True, strategy="entropy", return_full_data=False)
    df, cat_importances, precision, recall, cat_model = run_supervised(training_strat='retrain', l=100, corrected_weights=100, corrected_saved=True, strategy="margin", return_full_data=False)
    df, cat_importances, precision, recall, cat_model = run_supervised(training_strat='retrain', l=100, corrected_weights=100, corrected_saved=True, strategy="novelty", return_full_data=False)
    
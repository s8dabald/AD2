import numpy as np
import pandas as pd

def margin_uncertainty(df, proba):
    uncertainty_margin = 1 - (2 * np.abs(proba - 0.5))
    df['margin_uncertainty'] = uncertainty_margin
    return df

def entropy_uncertainty(df, proba):
    eps = 1e-12
    p = np.clip(proba, eps, 1 - eps)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    uncertainty_entropy = entropy / np.log(2)
    df['entropy_uncertainty'] = uncertainty_entropy
    return df

def novelty_uncertainty(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    X = df.drop(columns=["label", "posting_id", "pred_label", "pred_score"] )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = 5
    knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    knn.fit(X_scaled)
    distances, indices = knn.kneighbors(X_scaled)

    # Remove self-neighbor
    novelty_raw = distances[:, 1:].mean(axis=1)
    novelty = (novelty_raw - novelty_raw.min()) / (novelty_raw.max() - novelty_raw.min() + 1e-12)
    df['novelty_uncertainty'] = novelty
    return df

def uncertainty_query(df, strategy = "margin", exclude_posting_ids=None):
    if exclude_posting_ids is None:
        exclude_posting_ids = []
    if strategy == "margin":
        df = margin_uncertainty(df, df['pred_score'])
        filtered = df[~df['posting_id'].isin(exclude_posting_ids)]
        filtered = filtered.sort_values("margin_uncertainty", ascending=False).head(10)
        return filtered.drop(columns=["margin_uncertainty"])
    elif strategy == "entropy": 
        df = entropy_uncertainty(df, df['pred_score'])
        filtered = df[~df['posting_id'].isin(exclude_posting_ids)]
        filtered = filtered.sort_values("entropy_uncertainty", ascending=False).head(10)
        return filtered.drop(columns=["entropy_uncertainty"])
    elif strategy == "novelty":
        df = novelty_uncertainty(df)
        filtered = df[~df['posting_id'].isin(exclude_posting_ids)]
        filtered = filtered.sort_values("novelty_uncertainty", ascending=False).head(10)
        return filtered.drop(columns=["novelty_uncertainty"])
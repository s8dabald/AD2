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

def novelty_scores(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    X = df.drop(columns=["label", "posting_id", "pred_label", "pred_score"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = 5
    knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    knn.fit(X_scaled)
    distances, indices = knn.kneighbors(X_scaled)

    #remove the case itself
    novelty_raw = distances[:, 1:].mean(axis=1)
    novelty = (novelty_raw - novelty_raw.min()) / (novelty_raw.max() - novelty_raw.min() + 1e-12)
    return novelty

def novelty_uncertainty(df):
    df['novelty_uncertainty'] = novelty_scores(df)
    return df

def uncertainty_query(df, strategy="margin", exclude_posting_ids=None,
                     novelty_scores=None, random_scores=None, random_state=42):
    if exclude_posting_ids is None:
        exclude_posting_ids = []
    if strategy == "random":
        if random_scores is None:
            random_scores = np.random.RandomState(random_state).random_sample(len(df))
        random_scores = np.asarray(random_scores, dtype=float)
        if len(random_scores) != len(df):
            raise ValueError("random_scores must have one value per dataframe row")
        eligible = np.flatnonzero(~df["posting_id"].isin(exclude_posting_ids).to_numpy())
        if not len(eligible):
            return df.iloc[0:0]
        chosen = eligible[int(np.argmax(random_scores[eligible]))]
        return df.iloc[[chosen]]
    elif strategy == "margin":
        df = margin_uncertainty(df, df['pred_score'])
        filtered = df[~df['posting_id'].isin(exclude_posting_ids)]
        filtered = filtered.sort_values("margin_uncertainty", ascending=False).head(1)
        return filtered.drop(columns=["margin_uncertainty"])
    elif strategy == "entropy": 
        df = entropy_uncertainty(df, df['pred_score'])
        filtered = df[~df['posting_id'].isin(exclude_posting_ids)]
        filtered = filtered.sort_values("entropy_uncertainty", ascending=False).head(1)
        return filtered.drop(columns=["entropy_uncertainty"])
    elif strategy == "novelty":
        if novelty_scores is None:
            df = novelty_uncertainty(df)
        else:
            df["novelty_uncertainty"] = novelty_scores
        filtered = df[~df['posting_id'].isin(exclude_posting_ids)]
        filtered = filtered.sort_values("novelty_uncertainty", ascending=False).head(1)
        return filtered.drop(columns=["novelty_uncertainty"])


def select_batch(frame, excluded_ids, reviewed_ids, priority, tie_break, batch_size):
    """Select deterministic highest-priority unreviewed batch, preserving canonical tie breaks."""
    ids = frame["posting_id"].astype(str)
    eligible = ~ids.isin(set(excluded_ids) | set(reviewed_ids)).to_numpy()
    rows = np.flatnonzero(eligible)
    priority = np.asarray(priority, dtype=float)
    tie_break = np.asarray(tie_break, dtype=float)
    order = rows[np.lexsort((tie_break[rows], -priority[rows]))]
    chosen = order[:batch_size]
    if len(chosen) != batch_size:
        raise ValueError(f"Expected {batch_size} reviews, selected {len(chosen)}")
    return ids.iloc[chosen].tolist()

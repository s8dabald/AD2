import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from querystep import margin_uncertainty, entropy_uncertainty, novelty_uncertainty


def get_feature_cols(df):
    return df.drop(columns=["label", "posting_id", "pred_label", "pred_score"]).columns.tolist()


def estimate_threshold(df, feature_cols, k=10, sample_size=5000, random_state=42):
    """T_init: Median der mittleren k-NN-Distanzen im gescalten Feature-Space."""
    X = df[feature_cols].values
    if sample_size is not None and len(X) > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X = X[idx]
    X_scaled = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric="euclidean", n_jobs=-1)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    knn_dists = distances[:, 1:].mean(axis=1)
    return float(np.median(knn_dists))


def global_partition(df, centers, feature_cols, T):
    """Globales Voronoi: jeder Case -> nächstgelegenes Center. Gecovert, wenn innerhalb T.
    Returns: nearest (int array, Center-Index je Row), covered (bool array)."""
    X_scaled = StandardScaler().fit_transform(df[feature_cols].values)
    ids = df["posting_id"].values
    center_ids = list(centers.keys())
    center_idx = [np.where(ids == c)[0][0] for c in center_ids]
    C = X_scaled[center_idx]

    n = len(df)
    k = len(center_ids)
    dists = np.empty((n, k))
    for j in range(k):
        diff = X_scaled - C[j]
        dists[:, j] = np.sqrt((diff ** 2).sum(axis=1))

    nearest = dists.argmin(axis=1)
    nearest_dist = dists[np.arange(n), nearest]
    covered = nearest_dist <= T
    return nearest, covered


def _uncertainty_scores(df, strategy):
    if strategy == "margin":
        df = margin_uncertainty(df, df["pred_score"])
        return df, "margin_uncertainty"
    elif strategy == "entropy":
        df = entropy_uncertainty(df, df["pred_score"])
        return df, "entropy_uncertainty"
    elif strategy == "novelty":
        df = novelty_uncertainty(df)
        return df, "novelty_uncertainty"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def greedy_iteration(df, strategy, state):
    """Eine greedy Iteration: Selektion des most uncertain Case M, Oracle-Review,
    ggf. neues Center (Split/neues Territorium), globaler Re-Partition, Propagation.

    state: dict mit
        centers            : {posting_id -> label}
        directly_corrected : set
        covered            : set (monoton wachsend)
        region_of          : {posting_id -> center posting_id}
        T                  : Cluster-Radius (None -> wird geschätzt)

    Returns: (df_updated, state_updated, meta)
    """
    meta = {}

    if state["T"] is None:
        feature_cols = get_feature_cols(df)
        state["T"] = estimate_threshold(df, feature_cols)

    df, score_col = _uncertainty_scores(df, strategy)
    pool_mask = ~df["posting_id"].isin(state["directly_corrected"])
    pool = df[pool_mask]
    if pool.empty:
        df = df.drop(columns=[score_col])
        meta["type"] = "no_candidates"
        return df, state, meta

    m_pos = pool[score_col].idxmax()
    M = df.loc[m_pos, "posting_id"]
    M_label = int(df.loc[m_pos, "label"])
    df = df.drop(columns=[score_col])

    if M not in state["covered"]:
        state["centers"][M] = M_label
        state["directly_corrected"].add(M)
        meta["type"] = "new_center"
    else:
        cur_center = state["region_of"][M]
        cur_label = state["centers"][cur_center]
        state["directly_corrected"].add(M)
        if cur_label == M_label:
            meta["type"] = "skip"
        else:
            state["centers"][M] = M_label
            meta["type"] = "split"
            meta["split_from"] = cur_center

    feature_cols = get_feature_cols(df)
    nearest, covered = global_partition(df, state["centers"], feature_cols, state["T"])

    ids = df["posting_id"].values
    center_ids = list(state["centers"].keys())
    center_labels = np.array([state["centers"][c] for c in center_ids])

    prop_labels = center_labels[nearest].astype(df["pred_label"].dtype)
    prev_pred = df["pred_label"].values.copy()
    covered_pos = np.where(covered)[0]
    if len(covered_pos):
        df.loc[df.index[covered_pos], "pred_label"] = prop_labels[covered_pos]

    state["covered"] = set(ids[covered])
    state["region_of"] = {ids[i]: center_ids[nearest[i]] for i in range(len(ids)) if covered[i]}

    changed = df["pred_label"].values != prev_pred
    meta["M"] = M
    meta["M_label"] = M_label
    meta["n_centers"] = len(state["centers"])
    meta["n_covered"] = len(state["covered"])
    meta["cumulative_direct"] = len(state["directly_corrected"])
    meta["n_flipped"] = int(changed.sum())
    if len(covered_pos):
        acc = (df.loc[df.index[covered_pos], "label"].values == prop_labels[covered_pos]).mean()
        meta["propagation_accuracy"] = float(acc)
    else:
        meta["propagation_accuracy"] = 1.0

    return df, state, meta

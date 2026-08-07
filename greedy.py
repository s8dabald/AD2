import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from querystep import margin_uncertainty, entropy_uncertainty, novelty_uncertainty


def get_feature_cols(df):
    return df.drop(columns=["label", "posting_id", "pred_label", "pred_score"]).columns.tolist()


def new_state(T=None):
    """State für den Greedy-Loop inkl. Proximity-Caches.

    caches:
        scaler       : StandardScaler, gefittet auf dem vollen Feature-Space (1x)
        X_scaled     : skaliertes Feature-Array (1x, statisch)
        X_ids        : posting_id-Array (1x, statisch)
        dist_cache   : {center_id -> Distanzvektor aller Cases zu diesem Center}
        nearest      : argmin-Partition aus dem letzten Re-Partition (Reuse bei skip)
        covered_arr  : covered-Bool-Array aus dem letzten Re-Partition
        novelty_scores : statische Novelty-Scores (1x)
    """
    return {
        "centers": {},
        "directly_corrected": set(),
        "covered": set(),
        "region_of": {},
        "T": T,
        "scaler": None,
        "X_scaled": None,
        "X_ids": None,
        "dist_cache": {},
        "nearest": None,
        "covered_arr": None,
        "novelty_scores": None,
    }


def estimate_threshold(df, feature_cols, scaler=None, k=10, sample_size=5000, random_state=42):
    """T_init: Median der mittleren k-NN-Distanzen im gescalten Feature-Space.

    Nutzt den (vollen) scaler, falls übergeben, sonst eigenen Fit.
    Deterministisch via RandomState(seed) und stabilem Feature-Space.
    """
    X = df[feature_cols].values
    if sample_size is not None and len(X) > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X = X[idx]
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric="euclidean", n_jobs=-1)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    knn_dists = distances[:, 1:].mean(axis=1)
    return float(np.median(knn_dists))


def _ensure_features(df, state, feature_cols):
    if state["scaler"] is None:
        state["scaler"] = StandardScaler().fit(df[feature_cols].values)
    if state["X_scaled"] is None:
        state["X_scaled"] = state["scaler"].transform(df[feature_cols].values)
    if state["X_ids"] is None:
        state["X_ids"] = df["posting_id"].values


def center_distances(state, center_id):
    """Distanzvektor aller Cases zum Center (via cached X_scaled)."""
    center_idx = np.where(state["X_ids"] == center_id)[0][0]
    diff = state["X_scaled"] - state["X_scaled"][center_idx]
    return np.sqrt((diff ** 2).sum(axis=1))


def _partition_from_cache(df, state):
    """Voronoi-Partition über die Distanz-Caches der aktuellen Center."""
    center_ids = list(state["centers"].keys())
    dist_matrix = np.column_stack([state["dist_cache"][c] for c in center_ids])
    nearest = dist_matrix.argmin(axis=1)
    nearest_dist = dist_matrix[np.arange(len(df)), nearest]
    covered = nearest_dist <= state["T"]
    state["nearest"] = nearest
    state["covered_arr"] = covered
    return nearest, covered


def _uncertainty_scores(df, strategy, state):
    if strategy == "margin":
        df = margin_uncertainty(df, df["pred_score"])
        return df, "margin_uncertainty"
    elif strategy == "entropy":
        df = entropy_uncertainty(df, df["pred_score"])
        return df, "entropy_uncertainty"
    elif strategy == "novelty":
        if state["novelty_scores"] is None:
            df = novelty_uncertainty(df)
            state["novelty_scores"] = df["novelty_uncertainty"].values.copy()
        else:
            df["novelty_uncertainty"] = state["novelty_scores"]
        return df, "novelty_uncertainty"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def greedy_iteration(df, strategy, state):
    """Eine greedy Iteration: Selektion des most uncertain Case M, Oracle-Review,
    ggf. neues Center (Split/neues Territorium), globaler Re-Partition, Propagation.

    state: dict via new_state(T); Caches werden befüllt und bei skip- Iterationen
    wiederverwendet (Re-Partition nur bei new_center/split).

    Returns: (df_updated, state_updated, meta)
    """
    meta = {}
    feature_cols = get_feature_cols(df)

    _ensure_features(df, state, feature_cols)
    if state["T"] is None:
        state["T"] = estimate_threshold(df, feature_cols, scaler=state["scaler"])

    df, score_col = _uncertainty_scores(df, strategy, state)
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

    if meta["type"] in ("new_center", "split"):
        state["dist_cache"][M] = center_distances(state, M)
        nearest, covered = _partition_from_cache(df, state)
    else:
        if state["nearest"] is None:
            nearest, covered = _partition_from_cache(df, state)
        else:
            nearest, covered = state["nearest"], state["covered_arr"]

    ids = state["X_ids"]
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

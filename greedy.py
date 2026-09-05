import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from querystep import margin_uncertainty, entropy_uncertainty, novelty_uncertainty

# Flag-based features: rule checks (promptly, weekend, nwh, top_n, high_cash).
# These are excluded when using features_raw.
FLAG_PREFIXES = ("promptly_", "weekend_", "nwh_", "top_n_", "high_cash_")


def get_feature_cols(df):
    """All model feature columns (everything except label, posting_id, pred_label, pred_score)."""
    return df.drop(columns=["label", "posting_id", "pred_label", "pred_score"]).columns.tolist()


def _get_feature_groups(df):
    """Split feature columns into raw (values) and flags (rule checks).
    features_all = features_raw + features_flags = all 119 features."""
    all_features = get_feature_cols(df)
    raw = [f for f in all_features if not any(f.startswith(p) for p in FLAG_PREFIXES)]
    flags = [f for f in all_features if any(f.startswith(p) for p in FLAG_PREFIXES)]
    return {"features_all": all_features, "features_raw": raw, "features_flags": flags}


def new_state(T=None, propagation_space=None):
    """State for the greedy loop, including propagation caches.

    Caches:
        T                : cluster radius (None = auto-estimate)
        centers          : {posting_id -> label}
        directly_corrected : set of reviewed posting_ids
        covered          : set (monotonically growing)
        region_of        : {posting_id -> center posting_id}
        X_ids            : posting_id array (1x, static)
        dist_cache       : {center_id -> distance vector for all cases}
        nearest          : argmin partition from last re-partition (reuse on skip)
        covered_arr      : covered bool array from last re-partition
        novelty_scores   : novelty scores (1x)
        propagation_space : list of space keys, e.g. ["features_all"] or ["shap_all"]
        prop_scaled      : scaled feature matrix for features_all (1x)
        prop_raw_scaled  : scaled feature matrix for features_raw (1x)
        prop_flags_scaled : scaled feature matrix for features_flags (1x)
        shap_vals        : SHAP array (N, n_features), set once after each retrain
        shap_raw_indices : column indices for shap_raw (1x)
        shap_flag_indices : column indices for shap_flags (1x)
    """
    return {
        "centers": {},
        "directly_corrected": set(),
        "covered": set(),
        "region_of": {},
        "T": T,
        "X_ids": None,
        "dist_cache": {},
        "nearest": None,
        "covered_arr": None,
        "novelty_scores": None,
        # propagation
        "propagation_space": propagation_space or ["features_all"],
        "prop_scaled": None,
        "prop_raw_scaled": None,
        "prop_flags_scaled": None,
        "shap_vals": None,
        "shap_raw_indices": None,
        "shap_flag_indices": None,
        # matrix cache
        "prop_matrix": None,
        "prop_matrix_shap_id": None,
        # selection / weighting
        "nearest_dist": None,
        "neighbor_density": None,
    }


def _ensure_propagation(df, state):
    """Scale feature groups and compute SHAP indices. Everything is cached (1x)."""
    groups = _get_feature_groups(df)
    ps = state["propagation_space"]
    n = len(df)

    # Scale feature groups on first call, then reuse
    if "features_all" in ps and state["prop_scaled"] is None:
        state["prop_scaled"] = StandardScaler().fit_transform(df[groups["features_all"]].values)
    if "features_raw" in ps and state["prop_raw_scaled"] is None:
        state["prop_raw_scaled"] = StandardScaler().fit_transform(df[groups["features_raw"]].values)
    if "features_flags" in ps and state["prop_flags_scaled"] is None:
        state["prop_flags_scaled"] = StandardScaler().fit_transform(df[groups["features_flags"]].values)

    if state["X_ids"] is None:
        state["X_ids"] = df["posting_id"].values

    # Compute SHAP column indices once (mapping from feature_cols order to SHAP columns)
    if state["shap_raw_indices"] is None and any(s.startswith("shap") for s in ps):
        all_features = groups["features_all"]
        state["shap_raw_indices"] = [i for i, f in enumerate(all_features)
                                     if not any(f.startswith(p) for p in FLAG_PREFIXES)]
        state["shap_flag_indices"] = [i for i, f in enumerate(all_features)
                                      if any(f.startswith(p) for p in FLAG_PREFIXES)]


def _build_propagation_matrix(state):
    """Build the combined matrix for distance computation based on propagation_space.
    Feature parts are already scaled, SHAP parts are used as-is (comparable scale)."""
    if state["prop_matrix"] is not None and state["prop_matrix_shap_id"] == id(state["shap_vals"]):
        return state["prop_matrix"]

    parts = []
    ps = state["propagation_space"]

    if "features_all" in ps and state["prop_scaled"] is not None:
        parts.append(state["prop_scaled"])
    if "features_raw" in ps and state["prop_raw_scaled"] is not None:
        parts.append(state["prop_raw_scaled"])
    if "features_flags" in ps and state["prop_flags_scaled"] is not None:
        parts.append(state["prop_flags_scaled"])

    if state["shap_vals"] is not None:
        if "shap_all" in ps:
            parts.append(state["shap_vals"])
        if "shap_raw" in ps and state["shap_raw_indices"] is not None:
            parts.append(state["shap_vals"][:, state["shap_raw_indices"]])
        if "shap_flags" in ps and state["shap_flag_indices"] is not None:
            parts.append(state["shap_vals"][:, state["shap_flag_indices"]])

    if not parts:
        raise ValueError("propagation_space produced empty matrix — check your config")
    matrix = np.hstack(parts)
    state["prop_matrix"] = matrix
    state["prop_matrix_shap_id"] = id(state["shap_vals"])
    return matrix


def _prop_dims(state):
    """Number of dimensions in the current propagation space (no matrix allocation)."""
    ps = state["propagation_space"]
    dims = 0
    if "features_all" in ps and state["prop_scaled"] is not None:
        dims += state["prop_scaled"].shape[1]
    if "features_raw" in ps and state["prop_raw_scaled"] is not None:
        dims += state["prop_raw_scaled"].shape[1]
    if "features_flags" in ps and state["prop_flags_scaled"] is not None:
        dims += state["prop_flags_scaled"].shape[1]
    if state["shap_vals"] is not None:
        if "shap_all" in ps:
            dims += state["shap_vals"].shape[1]
        if "shap_raw" in ps and state["shap_raw_indices"] is not None:
            dims += len(state["shap_raw_indices"])
        if "shap_flags" in ps and state["shap_flag_indices"] is not None:
            dims += len(state["shap_flag_indices"])
    return dims


def _compute_neighbor_density(state):
    """Anzahl Nachbarn innerhalb Radius T für jeden Fall in Propagierungs-Raum.
    Wird nach jedem SHAP-Update neu berechnet (Matrix ändert sich)."""
    from sklearn.neighbors import radius_neighbors_graph
    mat = _build_propagation_matrix(state)
    T = state["T"]
    graph = radius_neighbors_graph(mat, radius=T, mode='connectivity', include_self=False)
    state["neighbor_density"] = np.array(graph.sum(axis=1)).flatten().astype(float)


def estimate_threshold(state, k=10, sample_size=5000, random_state=42):
    """Estimate cluster radius T as median of mean k-NN distances in propagation space.
    Deterministic via RandomState(seed)."""
    mat = _build_propagation_matrix(state)
    if sample_size is not None and len(mat) > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(mat), size=sample_size, replace=False)
        mat = mat[idx]
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(mat)), metric="euclidean", n_jobs=-1)
    nn.fit(mat)
    distances, _ = nn.kneighbors(mat)
    knn_dists = distances[:, 1:].mean(axis=1)
    return float(np.median(knn_dists))


def center_distances(state, center_id):
    """Distance vector from all cases to a center in propagation space."""
    center_idx = np.where(state["X_ids"] == center_id)[0][0]
    mat = _build_propagation_matrix(state)
    diff = mat - mat[center_idx]
    return np.sqrt((diff ** 2).sum(axis=1))


def _partition_from_cache(state):
    """Voronoi partition using cached center distances."""
    center_ids = list(state["centers"].keys())
    dist_matrix = np.column_stack([state["dist_cache"][c] for c in center_ids])
    nearest = dist_matrix.argmin(axis=1)
    nearest_dist = dist_matrix[np.arange(len(state["X_ids"])), nearest]
    covered = nearest_dist <= state["T"]
    state["nearest"] = nearest
    state["covered_arr"] = covered
    state["nearest_dist"] = nearest_dist
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


def greedy_iteration(df, strategy, state, selection_mode="uncertainty"):
    """One greedy iteration: select most uncertain case M, oracle review,
    new center / split / skip, global re-partition, label propagation.

    Returns: (df_updated, state_updated, meta)
    """
    meta = {}

    _ensure_propagation(df, state)
    if state["T"] is None:
        state["T"] = estimate_threshold(state)

    df, score_col = _uncertainty_scores(df, strategy, state)
    pool_mask = ~df["posting_id"].isin(state["directly_corrected"])
    pool = df[pool_mask]
    if pool.empty:
        df = df.drop(columns=[score_col])
        meta["type"] = "no_candidates"
        return df, state, meta

    if selection_mode == "uncertainty_density":
        _compute_neighbor_density(state)
        pool_idx = pool.index.values
        uncertainty = pool[score_col].values
        density = state["neighbor_density"][pool_idx]
        density_max = density.max()
        density_norm = density / density_max if density_max > 0 else density
        combined = uncertainty * density_norm
        m_pos = pool.index[np.argmax(combined)]
    else:
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

    # Re-partition only when centers change; reuse on skip
    if meta["type"] in ("new_center", "split"):
        state["dist_cache"][M] = center_distances(state, M)
        nearest, covered = _partition_from_cache(state)
    else:
        if state["nearest"] is None:
            nearest, covered = _partition_from_cache(state)
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
    meta["prop_dims"] = _prop_dims(state)
    meta["selection_mode"] = selection_mode
    M_idx = np.where(state["X_ids"] == M)[0][0]
    meta["M_density"] = float(state["neighbor_density"][M_idx]) if state["neighbor_density"] is not None else 0.0
    if len(covered_pos):
        acc = (df.loc[df.index[covered_pos], "label"].values == prop_labels[covered_pos]).mean()
        meta["propagation_accuracy"] = float(acc)
    else:
        meta["propagation_accuracy"] = 1.0

    return df, state, meta

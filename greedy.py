import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
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


def new_state(T=None, propagation_space=None, seed=42):
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
        "seed": seed,
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
    # cluster_baskets
    "pipeline_baskets": [],
    "case_to_basket": {},
    "candidate_case_to_basket": {},
    "case_source": {},
    "review_phase_count": 0,
    "review_phase_size": 25,
    "phase": "review",
    "conflict_confirmations": [],
    "basket_counter": 0,
    "shap_scaler": None,
    "shap_std_cache": None,
    "shap_std_id": None,
        "topk_pos": None,
        "topk_k": None,
        "topk_fallback": False,
        "path_groups": None,
        "_ideal_map": None,
    "_ideal_names": None,
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
        state["T"] = estimate_threshold(state, random_state=state.get("seed", 42))

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


def path_group_iteration(df, strategy, state, selection_mode="uncertainty"):
    """One iteration of path-group propagation: review the most uncertain case and
    propagate its label to all cases that share its leaf path through all trees.

    Same seed logic as the other propagation modes (most uncertain, not yet
    directly corrected). The seed is the only directly corrected case (gets
    corrected_weights); propagated group members stay at weight 1. Already
    reviewed cases are never overwritten.

    Needs state["path_groups"] (Series posting_id -> group id) and no SHAP.

    Returns: (df_updated, state_updated, meta)
    """
    meta = {}

    if state.get("path_groups") is None:
        meta["type"] = "no_candidates"
        return df, state, meta

    df, score_col = _uncertainty_scores(df, strategy, state)
    pool_mask = ~df["posting_id"].isin(state["directly_corrected"])
    pool = df[pool_mask]
    if pool.empty:
        df = df.drop(columns=[score_col])
        meta["type"] = "no_candidates"
        return df, state, meta

    if selection_mode == "uncertainty_density":
        if state["T"] is None:
            state["T"] = estimate_threshold(state, random_state=state.get("seed", 42))
        if state["neighbor_density"] is None:
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
    M_pred_score = float(df.loc[m_pos, "pred_score"])
    M_pred_label = int(df.loc[m_pos, "pred_label"])
    df = df.drop(columns=[score_col])

    grp = df["posting_id"].map(state["path_groups"])
    group_id = grp.loc[m_pos]
    in_group = grp == group_id
    reviewed = df["posting_id"].isin(state["directly_corrected"])
    target = in_group & ~reviewed

    prev_pred = df["pred_label"].values.copy()
    df.loc[target, "pred_label"] = M_label
    state["directly_corrected"].add(M)

    labels = df.loc[target, "label"].astype(int).to_numpy()
    n_target = int(len(labels))
    n_mixed = int((labels != M_label).sum())

    meta["type"] = "path_group"
    meta["M"] = M
    meta["M_label"] = M_label
    meta["M_pred_score"] = M_pred_score
    meta["M_pred_label"] = M_pred_label
    meta["group_size"] = int(in_group.sum())
    meta["n_covered"] = n_target
    meta["n_flipped"] = int((df["pred_label"].values != prev_pred).sum())
    meta["propagation_accuracy"] = float((labels == M_label).mean()) if n_target else 1.0
    meta["n_mixed"] = n_mixed
    meta["cumulative_direct"] = len(state["directly_corrected"])
    meta["selection_mode"] = selection_mode
    return df, state, meta


def _adaptive_topk(row, pred_score):
    """Positions (Index in row/sel_cols) des entscheidungsrelevanten SHAP- prefixes.

    Logit-Baseline = logit(pred_score) - Summe(SHAP); kumuliert vorzeichenbehaftet,
    das letzte Ueberqueren der Logit-0 bestimmt den k-Hat. Reine SHAP/Score-Logik
    (kein Label). Gibt (k, order[:k]) zurueck; k=0 heisst Baseline entscheidet
    bereits -> kein Top-k (Fallback: voller Raum).
    """
    p = min(max(pred_score, 1e-7), 1.0 - 1e-7)
    logit = float(np.log(p / (1.0 - p)))
    base = logit - float(row.sum())
    order = np.argsort(-np.abs(row))
    cum = base + np.cumsum(row[order])
    tol = 1e-9 * max(1.0, abs(logit))
    side = np.where(cum >= tol, 1, -1)
    final = side[-1] if len(side) else 0
    dev = np.where(side != final)[0]
    if len(dev):
        k = int(dev[-1]) + 2  # Prefix bis einschliesslich des Features, das die Endseite fixiert
    else:
        k = 0  # Entscheidung steht schon bei der Baseline: keine Familie
    k = int(min(k, len(row)))
    return k, order[:k]


def _seed_and_shap_box(df, state, strategy, selection_mode, prop_delta,
                       case_delta_mode=False):
    """Shared seed selection + adaptive SHAP decision-space box (candidate family).

    Used by shap_box (reviewer_rule_propagate) and reviewer_box. Returns (df, info);
    info is None when the review pool is empty (no candidates). Does NOT apply any
    feature-space (value) filter.

    case_delta_mode=True (shap_box_u): box half-width is per CANDIDATE, derived
    from the candidate's uncertainty RANK u_rank_i in [0, 1]:
        delta_i = 0.5 * clip((u_rank_i - 0.5) / 0.5, 0, 1)
    The bottom 50% get delta=0 (only exact SHAP twins); the top 50% ramp from
    0 to 0.5. Only error seeds expand, only same-prediction candidates are
    admitted, and directly reviewed cases are never overwritten.
    """
    _ensure_propagation(df, state)
    ps = state["propagation_space"]
    if "shap_raw" in ps and state["shap_raw_indices"] is not None:
        sel_cols = state["shap_raw_indices"]
    elif "shap_all" in ps and state["shap_vals"] is not None:
        sel_cols = list(range(state["shap_vals"].shape[1]))
    elif "shap_flags" in ps and state["shap_flag_indices"] is not None:
        sel_cols = state["shap_flag_indices"]
    else:
        raise ValueError("shap_box propagation_mode requires a shap propagation_space (z.B. ['shap_raw'])")
    if state["shap_vals"] is None:
        raise ValueError("shap_box needs shap_vals (compute_shap=True / initial_shap via run_unsupervised)")
    S = state["shap_vals"]

    df, score_col = _uncertainty_scores(df, strategy, state)
    pool_mask = ~df["posting_id"].isin(state["directly_corrected"])
    pool = df[pool_mask]
    if pool.empty:
        df = df.drop(columns=[score_col])
        return df, None

    if selection_mode == "uncertainty_density":
        if state["T"] is None:
            state["T"] = estimate_threshold(state, random_state=state.get("seed", 42))
        if state["neighbor_density"] is None:
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
    M_pred_score = float(df.loc[m_pos, "pred_score"])
    M_pred_label = int(df.loc[m_pos, "pred_label"])
    u_all = df[score_col].values.astype(float)
    u_rank_all = (df[score_col].rank(method="average", pct=True).values.astype(float)
                  if case_delta_mode else None)
    u_seed = float(df.loc[m_pos, score_col])
    df = df.drop(columns=[score_col])

    # ---- Decision Space: adaptive entscheidende SHAP-Features (echte Werte).
    # Baseline = Logit - Summe der SHAP-Beitraege im Zielfeatures-Raum; daraus
    # kumulieren (vorzeichenbehaftet) und das LETZTE Ueberqueren der Logit-0
    # bestimmen. Danach kann der echte Rest das Vorzeichen nicht mehr kippen.
    M_idx = np.where(state["X_ids"] == M)[0][0]
    row = S[M_idx, sel_cols]
    k, topk = _adaptive_topk(row, M_pred_score)
    if case_delta_mode:
        rank_scaled = np.clip((u_rank_all - 0.5) / 0.5, 0.0, 1.0)
        delta_all = 0.5 * rank_scaled
    else:
        delta_all = np.full(len(df), min(prop_delta, 0.99))
    delta_seed = float(delta_all[M_idx])
    u_rank_seed = float(u_rank_all[M_idx]) if case_delta_mode else None
    d = delta_seed

    mask = np.zeros(len(df), dtype=bool)
    family_size = 1
    col_idx = []
    bands = []
    if k >= 1:
        col_idx = [sel_cols[t] for t in topk]
        bands = []
        for j in col_idx:
            v = float(S[M_idx, j])
            if v == 0.0:
                lo, hi = -1e-9, 1e-9
            else:
                lo, hi = sorted((v * (1.0 - d), v * (1.0 + d)))
            bands.append((lo, hi))

        mask = np.ones(len(df), dtype=bool)
        for j in col_idx:
            v = float(S[M_idx, j])
            if v == 0.0:
                mask &= np.abs(S[:, j]) <= 1e-9
                continue
            lo = v * (1.0 - delta_all)
            hi = v * (1.0 + delta_all)
            mask &= (S[:, j] >= np.minimum(lo, hi)) & (S[:, j] <= np.maximum(lo, hi))
        if not mask.any():
            mask[M_idx] = True  # Fallback: mindestens der Seed selbst
        family_size = int(mask.sum())
    else:
        # k==0: Entscheidung steht bereits an der Baseline -> keine Familie, nur Seed
        mask[M_idx] = True

    if case_delta_mode:
        mask &= df["pred_label"].values == M_pred_label
        reviewed = df["posting_id"].isin(state["directly_corrected"]).values
        reviewed[M_idx] = False
        mask &= ~reviewed
        if M_label == M_pred_label:
            mask[:] = False
        mask[M_idx] = True
        family_size = int(mask.sum())

    return df, {
        "S": S, "sel_cols": sel_cols, "mask": mask, "family_size": family_size,
        "col_idx": col_idx, "bands": bands, "k": k, "M": M, "M_label": M_label,
        "M_pred_score": M_pred_score, "M_pred_label": M_pred_label,
        "M_idx": M_idx, "m_pos": m_pos, "u_all": u_all, "u_seed": u_seed,
        "u_rank_all": u_rank_all, "u_rank_seed": u_rank_seed,
        "delta_all": delta_all, "delta_seed": delta_seed,
        "case_delta_mode": case_delta_mode,
    }


def reviewer_rule_propagate(df, strategy, state, selection_mode="uncertainty",
                            prop_k=2, prop_delta=0.25, feature_space="off",
                            case_delta_mode=False):
    """Prüfer-Regel-Propagation (shap_box, adaptive Feature-Selektion): Statt
    Kugel/Voronoi baut der simulierte Prüfer aus der SHAP-Erklärung des reviewed
    Seeds eine Regel in zwei Ebenen.

    Decision Space: Adaptive Auswahl der SHAP-Features, die den Fall entschieden
    haben (echte, vorzeichenbehaftete Werte kumuliert ab Baseline; Prefix bis zum
    LETZTEN Überqueren der Logit-0 — danach kann der echte Rest das Urteil nicht
    mehr kippen). Kein festes k, keine Betragssummen. Um diese Features werden
    vorzeichen-erhaltende Bänder [v*(1-delta), v*(1+delta)] gezogen.

    Feature Space (Keyword `feature_space`):
      - "off": kein Wettere Filter (Verhalten wie bisheriges shap_box).
      - "value_box": alle Original-Features (features_raw) müssen innerhalb
        ±prop_delta um die Seed-Werte liegen.
      - "perfect_knowledge": Werte-Bänder aus den Family-Mitgliedern, deren
        echtes Label == Prüfer-Urteil über den Seed (min-max pro Original-Feature).
        Simuliert den Prüfer mit perfekter Kenntnis des gültigen Wertebereichs.

    case_delta_mode=True (shap_box_u): Die Bandbreite ist nicht fix, sondern
    pro KANDIDAT aus dessen Unsicherheits-RANG u_rank_i in [0,1] abgeleitet:
    delta_i = 0.5 * clip((u_rank_i - 0.5) / 0.5, 0, 1). Die unteren 50% werden
    nie verbreitert (delta=0, nur exakte SHAP-Zwillinge); die oberen 50% rampen
    von 0 bis 0.5. Zusaetzlich expandieren nur Error-Seeds, nur Kandidaten mit
    gleicher Vorhersage, und direkt reviewte Cases werden nie ueberschrieben.
    Nur mit feature_space="off" kombinierbar.

    Propagiert wird das Seed-Label auf die finalen Cases (last-writer-wins, ausser
    bei case_delta_mode: direkt reviewte Cases bleiben geschuetzt). Ist die Region
    leer, wird nur der Seed selbst korrigiert. Prüfer-Urteil/True Labels fließen
    NUR bei feature_space="perfect_knowledge" ein — nie in die SHAP-Auswahl.
    Braucht greedy_batching=True mit shap propagation_space.
    """
    meta = {}

    if case_delta_mode and feature_space != "off":
        raise ValueError("shap_box_u (case_delta_mode) requires feature_space='off'")

    df, box = _seed_and_shap_box(df, state, strategy, selection_mode, prop_delta,
                                 case_delta_mode=case_delta_mode)
    if box is None:
        meta["type"] = "no_candidates"
        return df, state, meta

    S = box["S"]
    sel_cols = box["sel_cols"]
    mask = box["mask"]
    family_size = box["family_size"]
    col_idx = box["col_idx"]
    bands = box["bands"]
    k = box["k"]
    M = box["M"]
    M_label = box["M_label"]
    M_pred_score = box["M_pred_score"]
    M_pred_label = box["M_pred_label"]
    M_idx = box["M_idx"]
    m_pos = box["m_pos"]
    delta_seed = box["delta_seed"]
    u_seed = box["u_seed"]
    u_rank_seed = box.get("u_rank_seed")
    d = delta_seed if case_delta_mode else min(prop_delta, 0.99)
    value_bands = None

    if k >= 1:
        # ---- Feature Space: alle Original-Features (features_raw), ohne Flags.
        if feature_space in ("value_box", "perfect_knowledge"):
            raw = _get_feature_groups(df)["features_raw"]
            if feature_space == "value_box":
                value_bands = []
                for f in raw:
                    v = float(df.loc[m_pos, f])
                    if v == 0.0:
                        lo, hi = -1e-9, 1e-9
                    else:
                        lo, hi = sorted((v * (1.0 - d), v * (1.0 + d)))
                    value_bands.append((f, lo, hi))
                    val = df[f].values
                    mask &= (val >= lo) & (val <= hi)
            else:  # perfect_knowledge: Bänder aus Family-Mitgliedern mit Seed-Urteil
                labels = df["label"].values.astype(int)
                sub = mask & (labels == M_label)
                if sub.any():
                    value_bands = []
                    for f in raw:
                        vals = df.loc[df.index[sub], f].values
                        lo, hi = float(vals.min()), float(vals.max())
                        value_bands.append((f, lo, hi))
                        val = df[f].values
                        mask &= (val >= lo - 1e-12) & (val <= hi + 1e-12)
        if not mask.any():
            mask = np.zeros(len(df), dtype=bool)
            mask[M_idx] = True  # Fallback: nur der Seed

    prev_pred = df["pred_label"].values.copy()
    covered_idxs = np.where(mask)[0]
    if len(covered_idxs):
        df.loc[df.index[covered_idxs], "pred_label"] = M_label

    ids = state["X_ids"]
    state["covered"] |= set(ids[mask])
    state["directly_corrected"].add(M)
    state["covered_arr"] = mask
    state["nearest_dist"] = None

    changed = df["pred_label"].values != prev_pred
    true_labels = df["label"].values.astype(int)
    new_pred = df["pred_label"].values
    n_changed_correct = int((changed & (new_pred == true_labels)).sum())
    n_changed_wrong = int((changed & (new_pred != true_labels)).sum())
    all_features = get_feature_cols(df)
    delta_covered = None
    u_covered = None
    if case_delta_mode and len(covered_idxs):
        dc = box["delta_all"][covered_idxs]
        uc = box["u_all"][covered_idxs]
        delta_covered = {
            "min": float(np.min(dc)),
            "median": float(np.median(dc)),
            "max": float(np.max(dc)),
        }
        u_covered = {
            "min": float(np.min(uc)),
            "median": float(np.median(uc)),
            "max": float(np.max(uc)),
        }
    meta["M"] = M
    meta["M_label"] = M_label
    meta["M_pred_score"] = M_pred_score
    meta["M_pred_label"] = M_pred_label
    meta["M_uncertainty"] = float(u_seed)
    meta["delta_seed"] = float(delta_seed)
    meta["delta_covered"] = delta_covered
    meta["covered_u"] = u_covered
    meta["case_delta_mode"] = case_delta_mode
    if case_delta_mode:
        meta["M_uncertainty_rank"] = None if u_rank_seed is None else float(u_rank_seed)
        meta["n_changed_correct"] = n_changed_correct
        meta["n_changed_wrong"] = n_changed_wrong
    meta["type"] = "skip" if not bool(changed.any()) else "rule"
    meta["rule"] = {
        "features": [all_features[j] for j in col_idx],
        "bands": [[lo, hi] for lo, hi in bands],
        "k": k,
        "delta": float(delta_seed) if case_delta_mode else prop_delta,
        "delta_mode": "uncertainty_rank" if case_delta_mode else "fixed",
        "delta_covered": delta_covered,
        "feature_space": feature_space,
        "family_size": family_size,
    }
    meta["value_bands"] = value_bands
    meta["n_covered"] = len(covered_idxs)
    meta["cumulative_direct"] = len(state["directly_corrected"])
    meta["n_flipped"] = int(changed.sum())
    meta["prop_dims"] = len(sel_cols)
    meta["selection_mode"] = selection_mode
    meta["M_density"] = float(state["neighbor_density"][M_idx]) if state["neighbor_density"] is not None else 0.0
    if len(covered_idxs):
        acc = (df.loc[df.index[covered_idxs], "label"].values == M_label).mean()
        meta["propagation_accuracy"] = float(acc)
    else:
        meta["propagation_accuracy"] = 1.0

    return df, state, meta


# --- reviewer_box helpers ---

_ORIGINAL_NUM_CONST = ["date_difference"]
_ORIGINAL_CAT_CONST = ["tax_rate", "user"]
_ORIGINAL_CAT_EXPANDED = ["gl_account_name", "cd_flag", "promptly", "weekend",
                          "nwh", "top_n", "high_cash"]
_ORIGINAL_NUM_EXPANDED = ["amount"]


def _map_columns_to_original_features(df):
    """Map model feature columns back to original (pre-OHE) features.

    Returns {model_col -> (original, kind, slot)} with kind in {"num","cat"}.
    Numeric expanded columns keep the slot (amount_1..3); categorical constants
    have no slot (tax_rate_<cat>, user_<cat>); date_difference is numeric const.
    """
    consts = _ORIGINAL_NUM_CONST + _ORIGINAL_CAT_CONST
    expanded = _ORIGINAL_NUM_EXPANDED + _ORIGINAL_CAT_EXPANDED
    mapping = {}
    for col in get_feature_cols(df):
        if col in consts:
            kind = "num" if col in _ORIGINAL_NUM_CONST else "cat"
            mapping[col] = (col, kind, None)
            continue
        matched = None
        for orig in sorted(expanded, key=len, reverse=True):
            if col.startswith(orig + "_"):
                matched = orig
                break
        if matched is None:
            continue
        rest = col[len(matched) + 1:]
        parts = rest.split("_", 1)
        if not parts[0].isdigit():
            continue
        slot = int(parts[0])
        kind = "cat" if matched in _ORIGINAL_CAT_EXPANDED else "num"
        mapping[col] = (matched, kind, slot)
    return mapping


def _build_feature_units(df, col_idx, all_features):
    """Group SHAP-box model columns into original feature units (for the rule)."""
    mapping = _map_columns_to_original_features(df)
    units = {}
    for j in col_idx:
        info = mapping.get(all_features[j])
        if info is None:
            continue
        orig, kind, slot = info
        units.setdefault((orig, kind, slot), set()).add(all_features[j])
    return {key: sorted(cols) for key, cols in units.items()}


def _select_representatives(S, mask, M_idx, col_idx, n, strategy, seed, exclude_idx=None):
    """Label-free, diversity-driven selection of representative family members."""
    cand = np.where(mask)[0]
    cand = cand[cand != M_idx]
    if exclude_idx is not None and len(exclude_idx):
        cand = cand[~np.isin(cand, exclude_idx)]
    if n <= 0 or len(cand) == 0:
        return []
    if len(col_idx) == 0:
        return [int(i) for i in sorted(cand)[:n]]
    sub = S[:, col_idx]
    if len(cand) <= n:
        d = np.linalg.norm(sub[cand] - sub[M_idx], axis=1)
        order = np.lexsort((cand, -d))
        return [int(cand[p]) for p in order]
    if strategy == "random":
        rng = np.random.RandomState(seed)
        return [int(i) for i in rng.choice(cand, size=n, replace=False)]
    if strategy == "kmeans_medoid":
        km = KMeans(n_clusters=n, n_init=10, random_state=seed)
        labels = km.fit_predict(sub[cand])
        reps = []
        for c in range(n):
            members = cand[labels == c]
            if len(members) == 0:
                continue
            ctr = km.cluster_centers_[c]
            dist = np.linalg.norm(sub[members] - ctr, axis=1)
            reps.append(int(members[int(np.argmin(dist))]))
        return reps
    # farthest_point (default): greedy k-center from the seed
    d_seed = np.linalg.norm(sub[cand] - sub[M_idx], axis=1)
    chosen = [int(cand[int(np.lexsort((cand, -d_seed))[0])])]
    min_d = np.linalg.norm(sub[cand] - sub[chosen[0]], axis=1)
    while len(chosen) < n:
        dd = min_d.copy()
        for c in chosen:
            dd[cand == c] = -1.0
        if (dd < 0).all():
            break
        nxt = int(cand[int(np.lexsort((cand, -dd))[0])])
        chosen.append(nxt)
        min_d = np.minimum(min_d, np.linalg.norm(sub[cand] - sub[nxt], axis=1))
    return chosen


def _active_category(df, cols, case_idx):
    vals = df[cols].values[case_idx]
    nz = np.where(vals == 1)[0]
    return cols[int(nz[0])] if len(nz) else None


def _derive_reviewer_value_rule(df, seed_idx, accepted, rejected, units, prop_delta):
    """Derive a conservative feature-space rule from reviewed representatives.

    Returns a list of constraints (strict conjunction) or None when no
    discriminating feature could be established from the reviewed cases only.
    Numeric units: interval [min_A - pad, max_A + pad] (pad from prop_delta),
    kept only if some rejected value falls outside. Categorical units: excluded
    categories = rejected active cats not present in any accepted case, kept
    only if non-empty.
    """
    accepted = list(dict.fromkeys(list(accepted) + [seed_idx]))
    rejected = list(dict.fromkeys(list(rejected)))
    if len(rejected) == 0:
        return None
    rule = []
    for (orig, kind, slot), cols in units.items():
        if kind == "num":
            col = cols[0]
            a = df[col].values[accepted]
            r = df[col].values[rejected]
            min_a, max_a = float(a.min()), float(a.max())
            span = max_a - min_a
            if span > 0:
                pad = prop_delta * span
            else:
                base = max(abs(min_a), abs(max_a))
                pad = prop_delta * base if base > 0 else prop_delta
            lo, hi = min_a - pad, max_a + pad
            if not np.any((r < lo) | (r > hi)):
                continue
            rule.append({"kind": "num", "orig": orig, "slot": slot,
                         "col": col, "lo": float(lo), "hi": float(hi)})
        else:
            allowed = {_active_category(df, cols, i) for i in accepted}
            allowed.discard(None)
            excluded = set()
            for i in rejected:
                act = _active_category(df, cols, i)
                if act is not None and act not in allowed:
                    excluded.add(act)
            if not excluded:
                continue
            rule.append({"kind": "cat", "orig": orig, "slot": slot,
                         "exclude": sorted(excluded)})
    return rule if rule else None


def _apply_value_rule(df, mask, rule):
    m = mask.copy()
    for r in rule:
        if r["kind"] == "num":
            vals = df[r["col"]].values
            m &= (vals >= r["lo"]) & (vals <= r["hi"])
        else:
            for c in r["exclude"]:
                m &= (df[c].values != 1)
    return m


def reviewer_box_iteration(df, strategy, state, selection_mode="uncertainty",
                           n_representatives=3, rep_selection="farthest_point",
                           fallback="family", prop_delta=0.25):
    """reviewer_box: SHAP-Box-Kandidatenfamilie + reviewer-informierter Feature-Space-Filter.

    Seed und Kandidatenfamilie identisch zu shap_box. Zusätzlich werden
    n_representatives weitere Fälle aus der Familie simuliert geprüft (label-frei
    nach Diversität im SHAP-Raum gewählt); daraus wird eine sparsame
    Feature-Space-Regel abgeleitet (nur trennende Original-Features), die die
    Familie einschränkt. Keine Regel ableitbar -> Fallback ("family" = volle
    SHAP-Box-Familie, "seed_only" = nur der Seed). Reviewte Reps erhalten
    direkt_corrected (HITL-Gewichtung), behalten aber ihr Prüfer-Label und
    werden nicht mit M_label propagiert.
    """
    meta = {}
    df, box = _seed_and_shap_box(df, state, strategy, selection_mode, prop_delta)
    if box is None:
        meta["type"] = "no_candidates"
        return df, state, meta

    S = box["S"]
    sel_cols = box["sel_cols"]
    mask = box["mask"]
    family_size = box["family_size"]
    col_idx = box["col_idx"]
    bands = box["bands"]
    k = box["k"]
    M = box["M"]
    M_label = box["M_label"]
    M_pred_score = box["M_pred_score"]
    M_pred_label = box["M_pred_label"]
    M_idx = box["M_idx"]

    ids = state["X_ids"]
    all_features = get_feature_cols(df)
    units = _build_feature_units(df, col_idx, all_features)

    # 1) Repräsentanten wählen (label-frei, ohne bereits geprüfte Fälle) und
    #    simulierten Prüfer anwenden (Urteil = echtes Label der Reps).
    reviewed_set = set(str(p) for p in state.get("directly_corrected", set()))
    exclude_idx = np.array([i for i in range(len(ids)) if str(ids[i]) in reviewed_set],
                           dtype=int)
    reps = _select_representatives(S, mask, M_idx, col_idx, n_representatives,
                                   rep_selection, state.get("seed", 42), exclude_idx)
    accepted = []
    rejected = []
    rep_labels = {}
    for ridx in reps:
        r_label = int(df.iloc[ridx]["label"])
        rep_labels[int(ridx)] = r_label
        df.loc[df.index[ridx], "pred_label"] = r_label
        state["directly_corrected"].add(ids[ridx])
        if r_label == M_label:
            accepted.append(int(ridx))
        else:
            rejected.append(int(ridx))

    # 2) Feature-Space-Regel aus den reviewten Reps ableiten
    rule = _derive_reviewer_value_rule(df, M_idx, accepted, rejected, units, prop_delta)
    fallback_used = False
    if rule is None:
        fallback_used = True
        if fallback == "seed_only":
            mask = np.zeros(len(df), dtype=bool)
            mask[M_idx] = True
        # "family": SHAP-Box-Familie unverändert propagieren
    else:
        mask = _apply_value_rule(df, mask, rule)
        if not mask.any():
            mask[M_idx] = True

    # 3) Propagation: nur Familie OHNE reviewte Reps (deren Label ist bereits korrekt)
    prev_pred = df["pred_label"].values.copy()
    rep_set = set(rep_labels.keys())
    prop_idxs = np.array([i for i in np.where(mask)[0] if i not in rep_set], dtype=int)
    if len(prop_idxs):
        df.loc[df.index[prop_idxs], "pred_label"] = M_label

    state["covered"] |= set(ids[mask])
    state["directly_corrected"].add(M)
    state["covered_arr"] = mask
    state["nearest_dist"] = None

    changed = df["pred_label"].values != prev_pred
    meta["M"] = M
    meta["M_label"] = M_label
    meta["M_pred_score"] = M_pred_score
    meta["M_pred_label"] = M_pred_label
    meta["type"] = "skip" if not bool(changed.any()) else "rule"
    meta["fallback_used"] = fallback_used
    meta["n_representatives"] = int(len(reps))
    meta["n_accepted"] = int(len(accepted))
    meta["n_rejected"] = int(len(rejected))
    meta["reviewed_representatives"] = [str(ids[r]) for r in reps]
    meta["n_rule_features"] = int(len(rule)) if rule else 0
    meta["reviewer_rule"] = rule
    meta["rule"] = {
        "features": [all_features[j] for j in col_idx],
        "bands": [[lo, hi] for lo, hi in bands],
        "k": k,
        "delta": prop_delta,
        "feature_space": "reviewer",
        "family_size": family_size,
    }
    meta["n_covered"] = int(len(prop_idxs))
    meta["cumulative_direct"] = len(state["directly_corrected"])
    meta["n_flipped"] = int(changed.sum())
    meta["prop_dims"] = len(sel_cols)
    meta["selection_mode"] = selection_mode
    meta["M_density"] = float(state["neighbor_density"][M_idx]) if state["neighbor_density"] is not None else 0.0
    if len(prop_idxs):
        acc = (df.loc[df.index[prop_idxs], "label"].values == M_label).mean()
        meta["propagation_accuracy"] = float(acc)
    else:
        meta["propagation_accuracy"] = 1.0

    return df, state, meta


# --- Cluster Baskets helpers ---

# --- Cluster Baskets helpers ---

_IDEAL_MAP = None
_IDEAL_NAMES = None
_IDEAL_CENTROIDS = None


def _load_ideal():
    global _IDEAL_MAP, _IDEAL_NAMES, _IDEAL_CENTROIDS
    if _IDEAL_MAP is not None:
        return _IDEAL_MAP, _IDEAL_NAMES
    ideal_path = "probe_out/ideal_membership.csv"
    if os.path.exists(ideal_path):
        try:
            dfm = pd.read_csv(ideal_path)
            _IDEAL_MAP = {str(r["posting_id"]): (int(r["cluster"]), int(r["label"]))
                          for _, r in dfm.iterrows()}
        except Exception:
            _IDEAL_MAP = {}
    else:
        _IDEAL_MAP = {}
    names_path = "probe_out/ideal_baskets_centroids_std.csv"
    if os.path.exists(names_path):
        try:
            dfn = pd.read_csv(names_path)
            if "basket_id" not in dfn.columns or "basket_name" not in dfn.columns:
                raise ValueError(
                    f"{names_path} must contain basket_id and basket_name columns"
                )
            if dfn["basket_id"].duplicated().any():
                duplicates = sorted(dfn.loc[dfn["basket_id"].duplicated(), "basket_id"].unique())
                raise ValueError(f"{names_path} contains duplicate basket IDs: {duplicates}")
            missing_names = sorted(
                dfn.loc[
                    dfn["basket_name"].isna()
                    | dfn["basket_name"].astype(str).str.strip().eq(""),
                    "basket_id",
                ].astype(int).tolist()
            )
            if missing_names:
                raise ValueError(
                    f"{names_path} contains missing basket names for IDs: {missing_names}"
                )
            _IDEAL_NAMES = {int(r["basket_id"]): (str(r.get("basket_name", "")) if pd.notna(r.get("basket_name", "")) else "",
                                                  int(r["label"]) if "label" in r else 0)
                            for _, r in dfn.iterrows()}
            shap_cols = [c for c in dfn.columns if c.startswith("shap_std_")]
            _IDEAL_CENTROIDS = {
                int(r["basket_id"]): np.asarray(r[shap_cols].values, dtype=float)
                for _, r in dfn.iterrows()
            }
        except ValueError:
            raise
        except Exception:
            _IDEAL_NAMES = {}
            _IDEAL_CENTROIDS = {}
    else:
        _IDEAL_NAMES = {}
        _IDEAL_CENTROIDS = {}
    return _IDEAL_MAP, _IDEAL_NAMES


def _oracle_basket_for(pid):
    m, n = _load_ideal()
    k = str(pid)
    if k not in m:
        return None, None, ""
    bid, lbl = m[k]
    name = ""
    if n and bid in n:
        name = n[bid][0] or ""
    return bid, lbl, name


def _reviewer_basket_for(pid, fallback_bid, fallback_label, fallback_name=""):
    """Return the simulated reviewer's selected basket for a case.

    The current offline experiment uses the Oracle mapping as its deterministic
    reviewer decision. Keeping that lookup behind this helper makes the
    selected basket explicit and leaves one seam for a future UI/API decision.
    """
    selected_bid, selected_label, selected_name = _oracle_basket_for(pid)
    if selected_bid is None:
        return fallback_bid, fallback_label, fallback_name
    return selected_bid, selected_label, selected_name


def _oracle_centroid_for(basket_id, sel_cols):
    _load_ideal()
    centroid = _IDEAL_CENTROIDS.get(basket_id) if _IDEAL_CENTROIDS else None
    if centroid is None:
        return None
    # The exported Oracle centroids are already in the selected SHAP space
    # (e.g. 52 shap_raw dimensions), while sel_cols refers to full-model
    # feature positions. Do not apply those full-space indices a second time.
    if len(centroid) == len(sel_cols):
        return centroid
    if max(sel_cols, default=-1) >= len(centroid):
        return None
    return centroid[sel_cols]


def _ensure_shap_std(state, S_raw, ids):
    if S_raw is None:
        return None
    if state["shap_std_cache"] is not None and state["shap_std_id"] == id(state["shap_vals"]):
        return state["shap_std_cache"]
    if state["shap_scaler"] is None:
        sc = StandardScaler()
        state["shap_scaler"] = sc.fit(S_raw)
    S_std = state["shap_scaler"].transform(S_raw)
    state["shap_std_cache"] = S_std
    state["shap_std_id"] = id(state["shap_vals"])
    return S_std


def _normalize_distance_scores(distances):
    """Normalize finite distances while keeping unavailable distances neutral."""
    distances = np.asarray(distances, dtype=float)
    normalized = np.zeros_like(distances)
    finite = np.isfinite(distances)
    if not finite.any():
        return normalized
    max_distance = distances[finite].max()
    if max_distance > 0:
        normalized[finite] = distances[finite] / max_distance
    return normalized


def _get_or_create_pipeline_basket(state, ideal_bid, ideal_label, ideal_name):
    for b in state["pipeline_baskets"]:
        if b["ideal_bid"] == ideal_bid:
            return b
    b = {
        "id": state["basket_counter"],
        "ideal_bid": ideal_bid,
        "label": ideal_label,
        "name": ideal_name,
        "centroid_std": None,
        "reviewed_ids": set(),
    }
    state["basket_counter"] += 1
    state["pipeline_baskets"].append(b)
    return b


def _update_basket_centroids(state, S_std, ids):
    if S_std is None:
        return
    id_to_idx = {str(pid): i for i, pid in enumerate(ids)}
    for b in state["pipeline_baskets"]:
        if len(b["reviewed_ids"]) == 0:
            continue
        idxs = [id_to_idx[str(pid)] for pid in b["reviewed_ids"] if str(pid) in id_to_idx]
        if idxs:
            b["centroid_std"] = S_std[idxs].mean(axis=0)


def _ensure_basket_centroids(state, S_std, ids):
    """Keep all active basket centroids in the current SHAP dimensionality."""
    if S_std is None:
        return
    dimension = S_std.shape[1]
    id_to_idx = {str(pid): i for i, pid in enumerate(ids)}
    for b in state["pipeline_baskets"]:
        centroid = b.get("centroid_std")
        if isinstance(centroid, np.ndarray) and centroid.shape == (dimension,):
            continue
        idxs = [
            id_to_idx[str(pid)] for pid in b.get("reviewed_ids", set())
            if str(pid) in id_to_idx
        ]
        b["centroid_std"] = S_std[idxs].mean(axis=0) if idxs else np.zeros(dimension)


def _assign_unreviewed(state, S_std, ids, positions=None):
    if S_std is None or len(state["pipeline_baskets"]) == 0:
        return
    _ensure_basket_centroids(state, S_std, ids)
    # ensure centroids exist (fallback to zero if empty)
    for b in state["pipeline_baskets"]:
        if b["centroid_std"] is None:
            b["centroid_std"] = np.zeros(S_std.shape[1])
    # build centroids matrix
    cents = np.stack([b["centroid_std"] for b in state["pipeline_baskets"]], axis=0)
    b_ids = [b["id"] for b in state["pipeline_baskets"]]
    if positions is not None and len(positions):
        S_use = S_std[:, positions]
        cents_use = cents[:, positions]
    else:
        S_use = S_std
        cents_use = cents
    for i, pid in enumerate(ids):
        src = state["case_source"].get(str(pid))
        if src == "reviewed":
            continue
        d = ((S_use[i] - cents_use) ** 2).sum(axis=1)
        j = int(np.argmin(d))
        state["case_to_basket"][str(pid)] = b_ids[j]
        state["case_source"][str(pid)] = "propagated"


def cluster_baskets_iteration(df, strategy, state, selection_mode="uncertainty", basket_alpha=0.5,
                              label_consistent=False, decision_provider=None, adaptive_topk=False):
    meta = {}

    _ensure_propagation(df, state)
    ps = state["propagation_space"]
    if "shap_raw" in ps and state["shap_raw_indices"] is not None:
        sel_cols = state["shap_raw_indices"]
    elif "shap_all" in ps and state["shap_vals"] is not None:
        sel_cols = list(range(state["shap_vals"].shape[1]))
    elif "shap_flags" in ps and state["shap_flag_indices"] is not None:
        sel_cols = state["shap_flag_indices"]
    else:
        raise ValueError("cluster_baskets requires shap propagation_space")
    if state["shap_vals"] is None:
        raise ValueError("cluster_baskets needs shap_vals")

    S_raw = state["shap_vals"][:, sel_cols]
    ids = state["X_ids"]
    S_std = _ensure_shap_std(state, S_raw, ids)
    _ensure_basket_centroids(state, S_std, ids)

    df, score_col = _uncertainty_scores(df, strategy, state)
    pool_mask = ~df["posting_id"].isin(state["directly_corrected"]) if False else True  # reviewed excluded from selection? reviewed are directly_corrected
    pool_mask = ~df["posting_id"].astype(str).isin([str(p) for p in state.get("directly_corrected", set())]) if False else ~np.isin(df["posting_id"], list(state.get("directly_corrected", set())))
    # easier: exclude reviewed
    reviewed_set = set(str(pid) for pid in state.get("directly_corrected", set()))
    pool = df[~df["posting_id"].astype(str).isin(reviewed_set)]

    if pool.empty:
        df = df.drop(columns=[score_col])
        meta["type"] = "no_candidates"
        return df, state, meta

    # compute basket_fit: dist to assigned basket (adaptive_topk: vorheriger Seed-topk-Raum)
    if S_std is not None and len(state["pipeline_baskets"]) > 0:
        id_to_idx = {pid: i for i, pid in enumerate(ids)}
        cents = np.stack([b["centroid_std"] for b in state["pipeline_baskets"]], axis=0)
        b_ids_list = [b["id"] for b in state["pipeline_baskets"]]
        # map id->basket
        id_to_b = state["case_to_basket"]
        d_to_b = np.full(len(ids), np.inf)
        pos = state.get("topk_pos") if adaptive_topk else None
        for i, pid in enumerate(ids):
            bid = id_to_b.get(str(pid))
            if bid is not None:
                j = b_ids_list.index(bid) if bid in b_ids_list else 0
                if pos is not None:
                    diff = S_std[i, pos] - cents[j, pos]
                else:
                    diff = S_std[i] - cents[j]
                d_to_b[i] = (diff ** 2).sum() ** 0.5
    else:
        d_to_b = np.zeros(len(ids))

    pool_idx = pool.index.values
    u = pool[score_col].values
    bf = d_to_b[pool.index.values]
    bf_n = _normalize_distance_scores(bf)
    combined = basket_alpha * u + (1.0 - basket_alpha) * bf_n
    m_pos = pool.index[np.argmax(combined)]
    M = df.loc[m_pos, "posting_id"]
    M_str = str(M)

    if adaptive_topk:
        M_idx = int(np.where(ids == M)[0][0])
        k_top, pos_top = _adaptive_topk(S_raw[M_idx], float(df.loc[m_pos, "pred_score"]))
        state["topk_k"] = int(k_top)
        state["topk_pos"] = pos_top if k_top >= 1 else None
        state["topk_fallback"] = k_top < 1

    # determine basket from ideal
    ideal_bid, ideal_label, ideal_name = _oracle_basket_for(M)
    if ideal_bid is None:
        ideal_bid, ideal_label, ideal_name = 0, int(df.loc[m_pos, "label"]), ""

    if decision_provider is not None:
        selected = decision_provider({
            "posting_id": M_str,
            "proposed_basket": ideal_bid,
            "proposed_label": ideal_label,
            "proposed_name": ideal_name,
            "kind": "review",
        })
        if selected is None:
            df = df.drop(columns=[score_col])
            return df, state, {
                "type": "review_required",
                "phase": "review",
                "posting_id": M_str,
                "proposed_basket": ideal_bid,
                "proposed_label": ideal_label,
                "proposed_name": ideal_name,
            }
        ideal_bid, ideal_label, ideal_name = selected

    pb = _get_or_create_pipeline_basket(state, ideal_bid, ideal_label, ideal_name)
    # mark reviewed
    pb["reviewed_ids"].add(M_str)
    state["directly_corrected"].add(M)
    state["case_to_basket"][M_str] = pb["id"]
    state["case_source"][M_str] = "reviewed"

    _update_basket_centroids(state, S_std, ids)
    bid_to_label = {b["id"]: b["label"] for b in state["pipeline_baskets"]}
    _assign_unreviewed(state, S_std, ids,
                       positions=state.get("topk_pos") if adaptive_topk else None)
    prev_pred = df["pred_label"].values.copy()
    if label_consistent:
        state["candidate_case_to_basket"] = dict(state["case_to_basket"])
        case_labels = dict(zip(
            df["posting_id"].astype(str),
            df["pred_label"].astype(int),
        ))
        allowed = {
            pid: bid for pid, bid in state["case_to_basket"].items()
            if case_labels[pid] == bid_to_label[bid]
        }
        for pid in list(state["case_to_basket"]):
            if pid not in allowed and pid != M_str:
                state["case_to_basket"].pop(pid, None)
                state["case_source"].pop(pid, None)

    # propagate labels
    assigned_bids = np.asarray(
        [state["case_to_basket"].get(str(pid), -1) for pid in ids],
        dtype=object,
    )
    assigned_mask = assigned_bids != -1
    if assigned_mask.any():
        assigned_labels = np.asarray([
            bid_to_label[bid] for bid in assigned_bids[assigned_mask]
        ], dtype=df["pred_label"].dtype)
        pred_labels = df["pred_label"].to_numpy(copy=True)
        pred_labels[assigned_mask] = assigned_labels
        df["pred_label"] = pred_labels

    changed = df["pred_label"].values != prev_pred
    state["covered"] = set(ids)  # all assigned conceptually
    meta["M"] = M
    meta["type"] = "basket_add" if len(pb["reviewed_ids"]) == 1 else "basket_update"
    meta["n_baskets"] = len(state["pipeline_baskets"])
    meta["n_reviewed"] = len(state["directly_corrected"])
    meta["n_propagated"] = sum(1 for s in state["case_source"].values() if s == "propagated")
    meta["n_flipped"] = int(changed.sum())
    meta["baskets"] = [
        {
            "id": b["id"],
            "oracle_id": b["ideal_bid"],
            "name": b["name"],
            "label": b["label"],
            "total": sum(1 for bid in state["case_to_basket"].values() if bid == b["id"]),
            "reviewed": len(b["reviewed_ids"]),
            "propagated": sum(
                1 for pid, bid in state["case_to_basket"].items()
                if bid == b["id"] and state["case_source"].get(pid) == "propagated"
            ),
        }
        for b in state["pipeline_baskets"]
    ]
    if adaptive_topk:
        all_features = get_feature_cols(df)
        pos = state.get("topk_pos")
        meta["adaptive_topk"] = True
        meta["topk_k"] = state.get("topk_k")
        meta["topk_features"] = [all_features[sel_cols[t]] for t in pos] if pos is not None else []
        meta["topk_fallback"] = bool(state.get("topk_fallback"))
    df = df.drop(columns=[score_col])
    return df, state, meta


def label_consistent_basket_iteration(df, strategy, state, selection_mode="uncertainty",
                                      basket_alpha=0.5, decision_provider=None, adaptive_topk=False):
    if state["phase"] == "review":
        df, state, meta = cluster_baskets_iteration(
            df, strategy, state, selection_mode=selection_mode,
            basket_alpha=basket_alpha, label_consistent=True,
            decision_provider=decision_provider, adaptive_topk=adaptive_topk)
        if meta.get("type") == "review_required":
            meta["phase"] = "review"
            meta["force_retrain"] = False
            return df, state, meta
        state["review_phase_count"] += 1
        meta["phase"] = "review"
        meta["force_retrain"] = state["review_phase_count"] >= state["review_phase_size"]
        if meta["force_retrain"]:
            state["phase"] = "conflict"
        return df, state, meta

    _ensure_propagation(df, state)
    sel_cols = state["shap_raw_indices"]
    S_std = _ensure_shap_std(state, state["shap_vals"][:, sel_cols], state["X_ids"])
    ids = state["X_ids"]
    _ensure_basket_centroids(state, S_std, ids)
    prev_pred = df["pred_label"].values.copy()

    topk_pos = state.get("topk_pos") if adaptive_topk else None

    if not state["conflict_confirmations"]:
        cents = np.stack([b["centroid_std"] for b in state["pipeline_baskets"]], axis=0)
        basket_ids = [b["id"] for b in state["pipeline_baskets"]]
        if topk_pos is not None:
            S_use = S_std[:, topk_pos]
            cents_use = cents[:, topk_pos]
        else:
            S_use = S_std
            cents_use = cents
        nearest = np.argmin(
            ((S_use[:, None, :] - cents_use[None, :, :]) ** 2).sum(axis=2),
            axis=1,
        )
        state["candidate_case_to_basket"] = {
            str(pid): basket_ids[int(nearest[i])] for i, pid in enumerate(ids)
        }
        reviewed = {str(pid) for pid in state["directly_corrected"]}
        for b in state["pipeline_baskets"]:
            conflict = [
                i for i, pid in enumerate(ids)
                if state["candidate_case_to_basket"].get(str(pid)) == b["id"]
                and str(pid) not in reviewed
                and int(df.iloc[i]["pred_label"]) != b["label"]
            ]
            if not conflict:
                continue
            if len(conflict) == 1:
                chosen = conflict
            else:
                if topk_pos is not None:
                    S_conf = S_std[conflict][:, topk_pos]
                    cen_conf = b["centroid_std"][topk_pos]
                else:
                    S_conf = S_std[conflict]
                    cen_conf = b["centroid_std"]
                km = KMeans(n_clusters=2, n_init=10, random_state=state.get("seed", 42))
                sub_labels = km.fit_predict(S_conf)
                sub_centers = km.cluster_centers_
                target = int(np.argmin(np.linalg.norm(
                    sub_centers - cen_conf, axis=1)))
                chosen = [idx for idx, sub in zip(conflict, sub_labels) if sub == target]
            if topk_pos is not None:
                rep_vecs = S_std[chosen][:, topk_pos]
                rep_cen = b["centroid_std"][topk_pos]
            else:
                rep_vecs = S_std[chosen]
                rep_cen = b["centroid_std"]
            representative = chosen[int(np.argmin(np.linalg.norm(
                rep_vecs - rep_cen, axis=1)))]
            state["conflict_confirmations"].append({
                "basket_id": b["id"],
                "case_ids": [str(ids[idx]) for idx in chosen],
                "representative": str(ids[representative]),
            })

    confirmation = state["conflict_confirmations"][0] if state["conflict_confirmations"] else None
    if confirmation is not None:
        representative = confirmation["representative"]
        proposed_bid = confirmation["basket_id"]
        proposed_basket = next(
            b for b in state["pipeline_baskets"] if b["id"] == proposed_bid
        )
        if decision_provider is not None:
            selected = decision_provider({
                "posting_id": representative,
                "proposed_basket": proposed_bid,
                "proposed_label": proposed_basket["label"],
                "proposed_name": proposed_basket["name"],
                "cluster_size": len(confirmation["case_ids"]),
                "case_ids": confirmation["case_ids"],
                "kind": "propagation",
            })
            if selected is None:
                return df, state, {
                    "type": "review_required",
                    "phase": "conflict",
                    "posting_id": representative,
                    "proposed_basket": proposed_bid,
                    "proposed_label": proposed_basket["label"],
                    "proposed_name": proposed_basket["name"],
                    "cluster_size": len(confirmation["case_ids"]),
                    "case_ids": confirmation["case_ids"],
                }
            selected_bid, selected_label, selected_name = selected
            state["conflict_confirmations"].pop(0)
        else:
            state["conflict_confirmations"].pop(0)
        target_idx = int(np.where(ids.astype(str) == representative)[0][0])
        if decision_provider is None:
            selected_bid, selected_label, selected_name = _reviewer_basket_for(
                representative,
                fallback_bid=proposed_bid,
                fallback_label=proposed_basket["label"],
                fallback_name=proposed_basket["name"],
            )
        reviewed_basket = _get_or_create_pipeline_basket(
            state, selected_bid, selected_label, selected_name)
        reviewed_basket["reviewed_ids"].add(representative)
        state["directly_corrected"].add(ids[target_idx])
        state["case_to_basket"][representative] = reviewed_basket["id"]
        state["case_source"][representative] = "reviewed"
        df.loc[df.index[target_idx], "pred_label"] = reviewed_basket["label"]

        propagation_applied = selected_bid == proposed_bid
        if propagation_applied:
            target_indices = [
                int(np.where(ids.astype(str) == pid)[0][0])
                for pid in confirmation["case_ids"]
            ]
            for idx in target_indices:
                pid = str(ids[idx])
                state["case_to_basket"][pid] = proposed_bid
                state["case_source"][pid] = "reviewed" if pid == representative else "propagated"
                df.loc[df.index[idx], "pred_label"] = next(
                    b["label"] for b in state["pipeline_baskets"] if b["id"] == proposed_bid)

        if state["conflict_confirmations"]:
            state["phase"] = "conflict"
        else:
            state["phase"] = "review"
            state["review_phase_count"] = 0
        state["covered"] = set(state["case_to_basket"])
        meta = {
            "type": "conflict_confirmed" if propagation_applied else "conflict_rejected",
            "phase": "conflict",
            "force_retrain": not state["conflict_confirmations"],
            "representative": representative,
            "proposed_basket": proposed_bid,
            "selected_basket": selected_bid,
            "target_basket": proposed_bid,
            "confirmed": propagation_applied,
            "propagation_applied": propagation_applied,
            "n_baskets": len(state["pipeline_baskets"]),
            "n_reviewed": len(state["directly_corrected"]),
            "n_flipped": int((df["pred_label"].values != prev_pred).sum()),
            "baskets": [],
            "topk_k": state.get("topk_k"),
            "topk_fallback": bool(state.get("topk_fallback")),
        }
    else:
        state["phase"] = "review"
        state["review_phase_count"] = 0
        meta = {
            "type": "conflict_subclusters",
            "phase": "conflict",
            "force_retrain": True,
            "n_baskets": len(state["pipeline_baskets"]),
            "n_reviewed": len(state["directly_corrected"]),
            "n_flipped": 0,
            "baskets": [],
            "topk_k": state.get("topk_k"),
            "topk_fallback": bool(state.get("topk_fallback")),
        }
    for b in state["pipeline_baskets"]:
        total = sum(1 for bid in state["case_to_basket"].values() if bid == b["id"])
        meta["baskets"].append({
            "id": b["id"], "oracle_id": b["ideal_bid"], "name": b["name"],
            "label": b["label"], "total": total,
            "reviewed": len(b["reviewed_ids"]), "propagated": total - len(b["reviewed_ids"]),
        })
    return df, state, meta


def cluster_baskets_oracle_iteration(df, strategy, state, selection_mode="uncertainty", basket_alpha=0.5):
    """Cluster-basket propagation using fixed Oracle centroids from probe output."""
    _ensure_propagation(df, state)
    ps = state["propagation_space"]
    if "shap_raw" in ps and state["shap_raw_indices"] is not None:
        sel_cols = state["shap_raw_indices"]
    elif "shap_all" in ps and state["shap_vals"] is not None:
        sel_cols = list(range(state["shap_vals"].shape[1]))
    elif "shap_flags" in ps and state["shap_flag_indices"] is not None:
        sel_cols = state["shap_flag_indices"]
    else:
        raise ValueError("cluster_baskets_oracle requires shap propagation_space")
    if state["shap_vals"] is None:
        raise ValueError("cluster_baskets_oracle needs shap_vals")

    S_std = _ensure_shap_std(state, state["shap_vals"][:, sel_cols], state["X_ids"])
    df, score_col = _uncertainty_scores(df, strategy, state)
    reviewed_set = {str(pid) for pid in state.get("directly_corrected", set())}
    pool = df[~df["posting_id"].astype(str).isin(reviewed_set)]
    if pool.empty:
        df = df.drop(columns=[score_col])
        return df, state, {"type": "no_candidates"}

    if state["pipeline_baskets"]:
        cents = np.stack([b["centroid_std"] for b in state["pipeline_baskets"]], axis=0)
        b_ids = [b["id"] for b in state["pipeline_baskets"]]
        assigned_dist = np.full(len(df), np.inf)
        for i, pid in enumerate(state["X_ids"]):
            bid = state["case_to_basket"].get(str(pid))
            if bid in b_ids:
                assigned_dist[i] = np.linalg.norm(S_std[i] - cents[b_ids.index(bid)])
    else:
        assigned_dist = np.zeros(len(df))

    pool_idx = pool.index.values
    uncertainty = pool[score_col].values
    dist = assigned_dist[pool_idx]
    dist_n = _normalize_distance_scores(dist)
    combined = basket_alpha * uncertainty + (1.0 - basket_alpha) * dist_n
    m_pos = pool.index[np.argmax(combined)]
    M = df.loc[m_pos, "posting_id"]
    ideal_bid, ideal_label, ideal_name = _oracle_basket_for(M)
    if ideal_bid is None:
        ideal_bid, ideal_label, ideal_name = 0, int(df.loc[m_pos, "label"]), ""
    centroid = _oracle_centroid_for(ideal_bid, sel_cols)
    if centroid is None or len(centroid) != S_std.shape[1]:
        raise ValueError(
            f"No compatible Oracle centroid for basket {ideal_bid} and propagation space"
        )

    pb = _get_or_create_pipeline_basket(state, ideal_bid, ideal_label, ideal_name)
    pb["centroid_std"] = centroid
    M_str = str(M)
    pb["reviewed_ids"].add(M_str)
    state["directly_corrected"].add(M)
    state["case_to_basket"][M_str] = pb["id"]
    state["case_source"][M_str] = "reviewed"
    _assign_unreviewed(state, S_std, state["X_ids"])

    prev_pred = df["pred_label"].values.copy()
    bid_to_label = {b["id"]: b["label"] for b in state["pipeline_baskets"]}
    for i, pid in enumerate(state["X_ids"]):
        bid = state["case_to_basket"].get(str(pid))
        if bid in bid_to_label:
            df.loc[df.index[i], "pred_label"] = bid_to_label[bid]

    changed = df["pred_label"].values != prev_pred
    state["covered"] = set(state["X_ids"])
    meta = {
        "M": M,
        "type": "basket_add" if len(pb["reviewed_ids"]) == 1 else "basket_update",
        "n_baskets": len(state["pipeline_baskets"]),
        "n_reviewed": len(state["directly_corrected"]),
        "n_propagated": sum(1 for s in state["case_source"].values() if s == "propagated"),
        "n_flipped": int(changed.sum()),
        "baskets": [],
    }
    for b in state["pipeline_baskets"]:
        meta["baskets"].append({
            "id": b["id"],
            "oracle_id": b["ideal_bid"],
            "name": b["name"],
            "label": b["label"],
            "total": sum(1 for bid in state["case_to_basket"].values() if bid == b["id"]),
            "reviewed": len(b["reviewed_ids"]),
            "propagated": sum(
                1 for pid, bid in state["case_to_basket"].items()
                if bid == b["id"] and state["case_source"].get(pid) == "propagated"
            ),
        })
    df = df.drop(columns=[score_col])
    return df, state, meta


def cluster_baskets_direct_iteration(df, state):
    """Recluster current SHAPs like the ideal-basket probe and map clusters."""
    _ensure_propagation(df, state)
    ps = state["propagation_space"]
    if "shap_raw" in ps and state["shap_raw_indices"] is not None:
        sel_cols = state["shap_raw_indices"]
    elif "shap_all" in ps and state["shap_vals"] is not None:
        sel_cols = list(range(state["shap_vals"].shape[1]))
    elif "shap_flags" in ps and state["shap_flag_indices"] is not None:
        sel_cols = state["shap_flag_indices"]
    else:
        raise ValueError("cluster_baskets_direct requires shap propagation_space")
    if state["shap_vals"] is None:
        raise ValueError("cluster_baskets_direct needs shap_vals")

    S_raw = state["shap_vals"][:, sel_cols]
    # Match shap_cluster_probe.py: fit the scaler on the current SHAP matrix
    # for every clustering pass, then cluster the scaled embedding.
    S_std = StandardScaler().fit_transform(S_raw)
    _load_ideal()
    if not _IDEAL_CENTROIDS:
        raise ValueError("cluster_baskets_direct needs Oracle centroids")

    n_clusters = len(_IDEAL_CENTROIDS)
    if len(S_std) < n_clusters:
        raise ValueError(
            f"cluster_baskets_direct needs at least {n_clusters} cases, got {len(S_std)}"
        )
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=state.get("seed", 42))
    cluster_ids = km.fit_predict(S_std)
    oracle_ids = sorted(_IDEAL_CENTROIDS)
    oracle_centroids = np.stack([
        _oracle_centroid_for(bid, sel_cols) for bid in oracle_ids
    ])
    if oracle_centroids.shape[1] != km.cluster_centers_.shape[1]:
        raise ValueError("cluster_baskets_direct has incompatible centroid dimensions")

    distances = (
        (km.cluster_centers_[:, None, :] - oracle_centroids[None, :, :]) ** 2
    ).sum(axis=2)
    cluster_rows, oracle_cols = linear_sum_assignment(distances)
    cluster_to_oracle = {
        int(cluster_id): oracle_ids[int(oracle_col)]
        for cluster_id, oracle_col in zip(cluster_rows, oracle_cols)
    }

    state["pipeline_baskets"] = []
    state["case_to_basket"] = {}
    state["case_source"] = {}
    for bid in oracle_ids:
        centroid = _oracle_centroid_for(bid, sel_cols)
        name, label = _IDEAL_NAMES.get(bid, ("", 0))
        state["pipeline_baskets"].append({
            "id": bid,
            "ideal_bid": bid,
            "label": label,
            "name": name,
            "centroid_std": centroid,
            "reviewed_ids": set(),
        })
    ids = state["X_ids"]
    prev_pred = df["pred_label"].values.copy()
    bid_to_label = {b["id"]: b["label"] for b in state["pipeline_baskets"]}
    for i, pid in enumerate(ids):
        bid = cluster_to_oracle[int(cluster_ids[i])]
        state["case_to_basket"][str(pid)] = bid
        state["case_source"][str(pid)] = "propagated"
        df.loc[df.index[i], "pred_label"] = bid_to_label[bid]
    state["covered"] = set(ids)

    changed = df["pred_label"].values != prev_pred
    meta = {
        "type": "direct_rematch",
        "n_baskets": len(state["pipeline_baskets"]),
        "n_reviewed": 0,
        "n_propagated": len(ids),
        "n_flipped": int(changed.sum()),
        "baskets": [],
    }
    for b in state["pipeline_baskets"]:
        total = sum(1 for bid in state["case_to_basket"].values() if bid == b["id"])
        meta["baskets"].append({
            "id": b["id"],
            "oracle_id": b["ideal_bid"],
            "name": b["name"],
            "label": b["label"],
            "total": total,
            "reviewed": 0,
            "propagated": total,
        })
    return df, state, meta
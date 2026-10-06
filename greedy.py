import os
import numpy as np
import pandas as pd
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
    # cluster_baskets
    "pipeline_baskets": [],
    "case_to_basket": {},
    "case_source": {},
    "basket_counter": 0,
    "shap_scaler": None,
    "shap_std_cache": None,
    "shap_std_id": None,
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


def reviewer_rule_propagate(df, strategy, state, selection_mode="uncertainty",
                            prop_k=2, prop_delta=0.25, feature_space="off"):
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

    Propagiert wird das Seed-Label auf die finalen Cases (last-writer-wins). Ist
    die Region leer, wird nur der Seed selbst korrigiert. Prüfer-Urteil/True Labels
    fließen NUR bei feature_space="perfect_knowledge" ein — nie in die SHAP-Auswahl.
    Braucht greedy_batching=True mit shap propagation_space.
    """
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
        raise ValueError("shap_box propagation_mode requires a shap propagation_space (z.B. ['shap_raw'])")
    if state["shap_vals"] is None:
        raise ValueError("shap_box needs shap_vals (compute_shap=True / initial_shap via run_unsupervised)")
    S = state["shap_vals"]

    df, score_col = _uncertainty_scores(df, strategy, state)
    pool_mask = ~df["posting_id"].isin(state["directly_corrected"])
    pool = df[pool_mask]
    if pool.empty:
        df = df.drop(columns=[score_col])
        meta["type"] = "no_candidates"
        return df, state, meta

    if selection_mode == "uncertainty_density":
        if state["T"] is None:
            state["T"] = estimate_threshold(state)
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

    # ---- Decision Space: adaptive entscheidende SHAP-Features (echte Werte).
    # Baseline = Logit - Summe der SHAP-Beiträge im Zielfeatures-Raum; daraus
    # kumulieren (vorzeichenbehaftet) und das LETZTE Überqueren der Logit-0
    # bestimmen. Danach kann der echte Rest das Vorzeichen nicht mehr kippen.
    M_idx = np.where(state["X_ids"] == M)[0][0]
    row = S[M_idx, sel_cols]
    p = min(max(M_pred_score, 1e-7), 1.0 - 1e-7)
    logit = float(np.log(p / (1.0 - p)))
    base = logit - float(row.sum())
    order = np.argsort(-np.abs(row))
    cum = base + np.cumsum(row[order])
    tol = 1e-9 * max(1.0, abs(logit))
    side = np.where(cum >= tol, 1, -1)
    final = side[-1] if len(side) else 0
    dev = np.where(side != final)[0]
    if len(dev):
        k = int(dev[-1]) + 2  # Prefix bis einschließlich des Features, das die Endseite fixiert
    else:
        k = 0  # Entscheidung steht schon bei der Baseline: keine Familie
    k = int(min(k, len(sel_cols)))
    d = min(prop_delta, 0.99)

    mask = np.zeros(len(df), dtype=bool)
    family_size = 1
    col_idx = []
    bands = []
    value_bands = None
    if k >= 1:
        topk = order[:k]
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
        for j, (lo, hi) in zip(col_idx, bands):
            col = S[:, j]
            mask &= (col >= lo) & (col <= hi)
        if not mask.any():
            mask[M_idx] = True  # Fallback: mindestens der Seed selbst
        family_size = int(mask.sum())

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
    else:
        # k==0: Entscheidung steht bereits an der Baseline -> keine Familie, nur Seed
        mask[M_idx] = True

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
    all_features = get_feature_cols(df)
    meta["M"] = M
    meta["M_label"] = M_label
    meta["M_pred_score"] = M_pred_score
    meta["M_pred_label"] = M_pred_label
    meta["type"] = "skip" if not bool(changed.any()) else "rule"
    meta["rule"] = {
        "features": [all_features[j] for j in col_idx],
        "bands": [[lo, hi] for lo, hi in bands],
        "k": k,
        "delta": prop_delta,
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


# --- Cluster Baskets helpers ---

# --- Cluster Baskets helpers ---

_IDEAL_MAP = None
_IDEAL_NAMES = None


def _load_ideal():
    global _IDEAL_MAP, _IDEAL_NAMES
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
            _IDEAL_NAMES = {int(r["basket_id"]): (str(r.get("basket_name", "")) if pd.notna(r.get("basket_name", "")) else "",
                                                  int(r["label"]) if "label" in r else 0)
                            for _, r in dfn.iterrows()}
        except Exception:
            _IDEAL_NAMES = {}
    else:
        _IDEAL_NAMES = {}
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
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    for b in state["pipeline_baskets"]:
        if len(b["reviewed_ids"]) == 0:
            continue
        idxs = [id_to_idx[pid] for pid in b["reviewed_ids"] if pid in id_to_idx]
        if idxs:
            b["centroid_std"] = S_std[idxs].mean(axis=0)


def _assign_unreviewed(state, S_std, ids):
    if S_std is None or len(state["pipeline_baskets"]) == 0:
        return
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    # ensure centroids exist (fallback to zero if empty)
    for b in state["pipeline_baskets"]:
        if b["centroid_std"] is None:
            b["centroid_std"] = np.zeros(S_std.shape[1])
    # build centroids matrix
    cents = np.stack([b["centroid_std"] for b in state["pipeline_baskets"]], axis=0)
    b_ids = [b["id"] for b in state["pipeline_baskets"]]
    for i, pid in enumerate(ids):
        src = state["case_source"].get(str(pid))
        if src == "reviewed":
            continue
        d = ((S_std[i] - cents) ** 2).sum(axis=1)
        j = int(np.argmin(d))
        state["case_to_basket"][str(pid)] = b_ids[j]
        state["case_source"][str(pid)] = "propagated"


def cluster_baskets_iteration(df, strategy, state, selection_mode="uncertainty", basket_alpha=0.5):
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

    # compute basket_fit: dist to assigned basket
    if S_std is not None and len(state["pipeline_baskets"]) > 0:
        id_to_idx = {pid: i for i, pid in enumerate(ids)}
        cents = np.stack([b["centroid_std"] for b in state["pipeline_baskets"]], axis=0)
        b_ids_list = [b["id"] for b in state["pipeline_baskets"]]
        # map id->basket
        id_to_b = state["case_to_basket"]
        d_to_b = np.full(len(ids), np.inf)
        for i, pid in enumerate(ids):
            bid = id_to_b.get(str(pid))
            if bid is not None:
                j = b_ids_list.index(bid) if bid in b_ids_list else 0
                d_to_b[i] = ((S_std[i] - cents[j]) ** 2).sum() ** 0.5
    else:
        d_to_b = np.zeros(len(ids))

    pool_idx = pool.index.values
    u = pool[score_col].values
    bf = d_to_b[pool.index.values]
    bf_max = bf.max() if bf.size else 1.0
    bf_n = bf / bf_max if bf_max > 0 else bf
    combined = basket_alpha * u + (1.0 - basket_alpha) * bf_n
    m_pos = pool.index[np.argmax(combined)]
    M = df.loc[m_pos, "posting_id"]
    M_str = str(M)

    # determine basket from ideal
    ideal_bid, ideal_label, ideal_name = _oracle_basket_for(M)
    if ideal_bid is None:
        ideal_bid, ideal_label, ideal_name = 0, int(df.loc[m_pos, "label"]), ""

    pb = _get_or_create_pipeline_basket(state, ideal_bid, ideal_label, ideal_name)
    # mark reviewed
    pb["reviewed_ids"].add(M_str)
    state["directly_corrected"].add(M)
    state["case_to_basket"][M_str] = pb["id"]
    state["case_source"][M_str] = "reviewed"

    _update_basket_centroids(state, S_std, ids)
    _assign_unreviewed(state, S_std, ids)

    # propagate labels
    prev_pred = df["pred_label"].values.copy()
    bid_to_label = {b["id"]: b["label"] for b in state["pipeline_baskets"]}
    for i, pid in enumerate(ids):
        bid = state["case_to_basket"].get(str(pid))
        if bid is not None and bid in bid_to_label:
            df.loc[df.index[i], "pred_label"] = bid_to_label[bid]

    changed = df["pred_label"].values != prev_pred
    state["covered"] = set(ids)  # all assigned conceptually
    meta["M"] = M
    meta["type"] = "basket_add" if len(pb["reviewed_ids"]) == 1 else "basket_update"
    meta["n_baskets"] = len(state["pipeline_baskets"])
    meta["n_reviewed"] = len(state["directly_corrected"])
    meta["n_propagated"] = sum(1 for s in state["case_source"].values() if s == "propagated")
    meta["n_flipped"] = int(changed.sum())

    df = df.drop(columns=[score_col])
    return df, state, meta
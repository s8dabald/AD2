"""Research diagnostic: fit CatBoost on stored labels, then cluster SHAP values.

Steps:
  1. Prepare the same features as main.py and run Isolation Forest.
  2. Use stored labels as CatBoost targets instead of Isolation Forest predictions.
  3. Fit CatBoost with the main pipeline's settings and calculate SHAP values.
  4. Select raw SHAP features using FLAG_PREFIXES.
  5. Build each embedding, run K-means, and report cluster metrics and profiles.

Writes artifacts only for the selected method and scaling.
Run with: python -m legacy.shap_cluster_probe
"""

import html as htmllib
import os
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, homogeneity_completeness_v_measure,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

from dataprep import full_dataprep
from isolation_forest import run_if
from models.kittyboost import train_catboost
from propagation import FLAG_PREFIXES, get_feature_cols

CONFIG = dict(
    space="raw",              #raw excludes flags; all keeps every feature
    methods=["none", "lda_within", "whiten", "lda_only", "shuffled"],
    scalings=["std", "none"],
    n_clusters=50,
    n_init=10,
    lda_extra=10,             #extra within-class PCA axes
    w_weight=1.0,             #weight on the fisher direction
    eig_floor=1e-4,           #eigenvalue floor for stable whitening
    seed=42,
    sil_sample=5000,          #sample size for internal metrics
    k2_reference=True,        #compare with unsupervised k=2 using ari
    max_rows=None,            #limit rows for quick tests
    top_features=5,           #top SHAP features per cluster
    max_clusters_shown=20,    #maximum cluster profiles to print
    #write artifacts only for selected runs
    save_for=("none", "std"),  #write cluster assignments, SHAP values, and html
    save_dir=os.path.join(os.path.dirname(__file__), "probe_out"),
    save_clusters="clusters_oracle.csv",
    save_matrix="shap_matrix.npz",
    save_baskets="baskets.html",
    def_features=6,           #features used to describe each basket
    def_min_share=0.80,       #minimum basket share for a defining feature
    def_min_lift=1.5,         #basket share divided by global share
    sample_rows=10,           #sample cases per basket in html
    show_all_baskets=True,    #include baskets without anomalies
)


#embeddings

def within_class_cov(X, y):
    """Return pooled covariance using only within-class variation."""
    classes = np.unique(y)
    S = np.zeros((X.shape[1], X.shape[1]))
    for c in classes:
        D = X[y == c] - X[y == c].mean(axis=0)
        S += D.T @ D
    return S / max(1, len(X) - len(classes))


def _classes_center(X, y):
    return np.array([X[y == c].mean(axis=0) for c in np.unique(y)])


def fisher_direction(X, y, eig_floor):
    """Return the unit-length Fisher/LDA direction."""
    S_W = within_class_cov(X, y)
    ridge = eig_floor * np.trace(S_W) / max(1, X.shape[1])
    mus = _classes_center(X, y)
    d = mus[-1] - mus[0]
    w = np.linalg.solve(S_W + ridge * np.eye(X.shape[1]), d)
    n = np.linalg.norm(w)
    return w / n if n > 0 else w


def _std_cols(E):
    return E / np.maximum(E.std(axis=0), 1e-12)


def build_embedding(X, y, method, lda_extra, w_weight, eig_floor, seed):
    """Build an embedding and return a short description for the report."""
    if method == "none":
        return X, "no labels; original SHAP geometry"

    w = fisher_direction(X, y, eig_floor)
    proj = X @ w

    if method == "lda_only":
        return _std_cols(proj.reshape(-1, 1)) * w_weight, \
            f"Fisher direction, weight={w_weight}"

    if method == "lda_within":
        R = X - np.outer(proj, w)              #project out the fisher direction
        Rc = R.copy()                          #center residuals within each class
        for c in np.unique(y):
            Rc[y == c] -= R[y == c].mean(axis=0)
        n_extra = min(lda_extra, max(1, X.shape[1] - 1))
        V = PCA(n_components=n_extra, random_state=seed).fit(Rc).components_
        E = np.column_stack([proj] + [R @ v for v in V])
        E = _std_cols(E)
        E[:, 0] *= w_weight
        return E, f"Fisher direction (weight={w_weight}) plus {n_extra} within-class PCA axes"

    if method == "whiten":
        S_W = within_class_cov(X, y)
        ridge = eig_floor * np.trace(S_W) / max(1, X.shape[1])
        eig, V = np.linalg.eigh(S_W + ridge * np.eye(X.shape[1]))
        inv_sqrt = (V * (1.0 / np.sqrt(np.maximum(eig, eig_floor * eig.max())))) @ V.T
        E = (X - X.mean(axis=0)) @ inv_sqrt
        return _std_cols(E), "within-class whitening (WCCN), all dimensions retained"

    raise ValueError(f"Unknown embedding method: {method}")


#metrics

def purity(y, c):
    tot = sum(np.bincount(y[c == cl], minlength=2).max() for cl in np.unique(c))
    return float(tot) / len(y)


def internal_metrics(X_ref, c, sil_sample, seed):
    """Score cluster quality in the original SHAP space without labels."""
    rng = np.random.default_rng(seed)
    n = len(X_ref)
    idx = rng.choice(n, size=min(sil_sample, n), replace=False) if sil_sample < n else np.arange(n)
    Xs, cs = X_ref[idx], c[idx]
    out = {"n_scored": int(len(idx)), "n_clusters_scored": int(len(np.unique(cs)))}
    if out["n_clusters_scored"] < 2:
        return {**out, "silhouette": float("nan"), "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan")}
    out["silhouette"] = float(silhouette_score(Xs, cs))
    out["davies_bouldin"] = float(davies_bouldin_score(Xs, cs))
    out["calinski_harabasz"] = float(calinski_harabasz_score(Xs, cs))
    return out


def external_metrics(y, c, seed):
    """Compare clusters with stored labels; label-informed embeddings score optimistically."""
    h, comp, v = homogeneity_completeness_v_measure(y, c)
    y_rand = np.random.default_rng(seed).permutation(y)
    return {
        "ari": float(adjusted_rand_score(y, c)),
        "nmi": float(normalized_mutual_info_score(y, c)),
        "homogeneity": float(h), "completeness": float(comp), "v_measure": float(v),
        "purity": purity(y, c),
        "ari_random_baseline": float(adjusted_rand_score(y_rand, c)),
        "nmi_random_baseline": float(normalized_mutual_info_score(y_rand, c)),
    }


#printing

def hdr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_cluster_profiles(X, c, y, feature_names, top, max_shown):
    """Print cluster sizes, anomaly counts, and leading mean-SHAP features."""
    clusters = []
    for cl in np.unique(c):
        m = c == cl
        mean_shap = X[m].mean(axis=0)
        lead = np.argsort(-np.abs(mean_shap))[:top]
        clusters.append(dict(cl=int(cl), n=int(m.sum()), n_anom=int(y[m].sum()),
                             share=float(y[m].mean()),
                             feats=", ".join(f"{feature_names[i]}={mean_shap[i]:+.2f}" for i in lead)))
    with_anom = sorted([d for d in clusters if d["n_anom"] > 0], key=lambda d: (-d["n_anom"], -d["n"]))
    rest = sorted([d for d in clusters if d["n_anom"] == 0], key=lambda d: -d["n"])

    print(f"Clusters: {len(clusters)} | with anomalies: {len(with_anom)} | without: {len(rest)}")
    print(f"Anomalies in clusters with exactly one anomaly: "
          f"{sum(1 for d in clusters if d['n_anom'] == 1)}")
    print(f"\nClusters with anomalies (top {max_shown}):")
    for d in with_anom[:max_shown]:
        print(f"  c{d['cl']:>3}  n={d['n']:>6}  anom={d['n_anom']:>4} ({d['share']:.1%})  {d['feats']}")
    if len(with_anom) > max_shown:
        print(f"  ... {len(with_anom) - max_shown} more clusters with anomalies")
    print(f"\nLargest clusters without anomalies (top 5):")
    for d in rest[:5]:
        print(f"  c{d['cl']:>3}  n={d['n']:>6}  anom=0                    {d['feats']}")


def print_size_hist(c, bins=(0, 1, 10, 100, 1000, 10000)):
    sizes = np.array([(c == cl).sum() for cl in np.unique(c)])
    edges = list(bins)
    print("Cluster size histogram:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        print(f"  [{lo:>6}, {hi:>6}) : {int(((sizes >= lo) & (sizes < hi)).sum()):>3} Cluster")
    if sizes.max() >= edges[-1]:
        print(f"  [{edges[-1]:>6},      inf) : {int((sizes >= edges[-1]).sum()):>3} Cluster")


#basket definitions
#report features that distinguish each basket from the global profile

def feature_stats(F):
    """Summarize each feature's type, common values, median, and quantiles."""
    stats = []
    for j in range(F.shape[1]):
        col = F[:, j]
        uniq = np.unique(col)
        if uniq.size <= 2 and set(uniq.tolist()) <= {0.0, 1.0}:
            share1 = float((col == 1).mean())
            stats.append(dict(kind="binary", share1=share1, uniq=uniq))
        elif uniq.size <= 12:
            vals, cnts = np.unique(col, return_counts=True)
            stats.append(dict(kind="cat", vals=vals, shares=cnts / len(col)))
        else:
            stats.append(dict(kind="num", median=float(np.median(col)),
                              q10=float(np.quantile(col, 0.10)),
                              q90=float(np.quantile(col, 0.90))))
    return stats


def basket_definition(F, stats, names, mask, top_n, min_share, min_lift):
    """Return the basket's strongest defining features, ranked by lift."""
    found = []
    for j, st in enumerate(stats):
        col = F[mask, j]
        if st["kind"] == "binary":
            s1 = float((col == 1).mean())
            g1 = st["share1"]
            if s1 >= min_share and g1 > 0:
                found.append((s1 / g1, names[j], "present", s1, g1))
            elif s1 <= 0.05 and g1 >= 0.2:
                found.append(((1 - s1) / max(1e-9, 1 - g1), names[j], "absent", s1, g1))
        elif st["kind"] == "cat":
            vals, cnts = np.unique(col, return_counts=True)
            i = int(np.argmax(cnts))
            share = float(cnts[i]) / len(col)
            gshare = float(st["shares"][i])
            if share >= min_share and gshare > 0 and share / gshare >= min_lift:
                found.append((share / gshare, names[j], f"={vals[i]:g}", share, gshare))
        else:
            med = float(np.median(col))
            if med < st["q10"] or med > st["q90"]:
                q25, q75 = np.quantile(col, [0.25, 0.75])
                found.append((abs(med - st["median"]) / max(1e-9, st["q90"] - st["q10"]),
                              names[j], f"median {med:.2f} [IQR {q25:.2f}-{q75:.2f}]",
                              med, st["median"]))
    found.sort(key=lambda t: -t[0])
    return found[:top_n]


def _def_line(item):
    lift, name, kind, share, gshare = item
    if kind == "present":
        return f"{name} — {share:.0%} in basket (global {gshare:.2%}, lift {lift:.1f}x)"
    if kind == "absent":
        return f"{name} — only {share:.0%} in basket (global {gshare:.0%})"
    return f"{name} — {kind} (global {gshare:.2f}, Lift {lift:.1f}x)"


def build_baskets(F, Xs, c, y, names, feat_names, posting_id, raw_amounts, cfg):
    """Build basket summaries with definitions, SHAP profiles, and sample cases."""
    stats = feature_stats(F)
    col_of = {nm: j for j, nm in enumerate(feat_names)}
    out = []
    for cl in np.unique(c):
        m = c == cl
        idx = np.where(m)[0]
        mean_shap = Xs[m].mean(axis=0)
        lead = np.argsort(-np.abs(mean_shap))[:cfg["top_features"]]
        defs = basket_definition(F, stats, feat_names, m, cfg["def_features"],
                                  cfg["def_min_share"], cfg["def_min_lift"])
        def_cols = [(d[1], col_of[d[1]]) for d in defs if d[1] in col_of]

        one = idx[y[idx] == 1]
        zero = idx[y[idx] == 0]
        take_one = min(cfg["sample_rows"] // 2, len(one))
        take_zero = min(cfg["sample_rows"] - take_one, len(zero))
        samples = np.concatenate([one[:take_one], zero[:take_zero]])

        out.append(dict(
            cl=int(cl), n=int(m.sum()), n_anom=int(y[m].sum()), share=float(y[m].mean()),
            defs=defs,
            shap=[f"{names[i]} {mean_shap[i]:+.2f}" for i in lead],
            samples=[dict(
                posting_id=str(posting_id[i]),
                label=int(y[i]),
                amounts=raw_amounts.get(str(posting_id[i]), []) if raw_amounts else [],
                raw={nm: round(float(F[i, j]), 3) for nm, j in def_cols},
                shap=[f"{names[j]} {Xs[i, j]:+.2f}" for j in np.argsort(-np.abs(Xs[i]))[:3]],
            ) for i in samples],
        ))
    out.sort(key=lambda b: (-b["n_anom"], -b["n"]))
    return out


def load_raw_amounts(posting_ids):
    """Load original amounts for the requested sample postings."""
    try:
        raw = pd.read_csv("journal_entries.csv", sep=";", usecols=["posting_id", "amount"])
    except Exception as exc:                                  # pragma: no cover
        print(f"[note] Could not load original amounts ({exc}); showing cases without amounts")
        return {}
    wanted = set(posting_ids.tolist())
    grp = raw[raw["posting_id"].isin(wanted)].groupby("posting_id")["amount"]
    return {str(k): [round(float(v), 2) for v in vals] for k, vals in grp}


def export_baskets(cfg, F_all, X_all, Xs, c, y, y_all, rows, names, feat_cols,
                   posting_id, method, scaling):
    """Save cluster assignments, SHAP values, and basket cards as HTML."""
    outdir = cfg["save_dir"]
    os.makedirs(outdir, exist_ok=True)
    tag = f"{method}_{scaling}"
    sel = rows if rows is not None else np.arange(len(X_all))

    pd.DataFrame({"posting_id": posting_id[sel], "label": y, "cluster": c,
                  "row": sel, "method": method, "scaling": scaling}) \
        .to_csv(os.path.join(outdir, cfg["save_clusters"]), index=False)

    np.savez_compressed(os.path.join(outdir, cfg["save_matrix"]),
                        shap=X_all.astype(np.float32),
                        label=y_all.astype(np.int8),
                        posting_id=posting_id.astype(str),
                        feature_names=np.array(feat_cols, dtype=object),
                        method=method, scaling=scaling)

    raw_amounts = load_raw_amounts(posting_id[sel])
    baskets = build_baskets(F_all[sel], Xs, c, y, names, feat_cols,
                            posting_id[sel], raw_amounts, cfg)

    pure = [b for b in baskets if b["n_anom"] == b["n"] and b["n"] > 0]
    stray = [b for b in baskets if 0 < b["n_anom"] < b["n"]]

    hdr(f"Basket profiles: {method} / {scaling} — could a reviewer describe these groups?")
    print(f"Baskets: {len(baskets)} total | {len(pure)} pure anomaly | "
          f"{len(stray)} mixed | "
          f"{sum(1 for b in baskets if b['n_anom'] == 0)} pure normal")
    for b in pure:
        print(f"\n  Basket {b['cl']}: {b['n']} cases, {b['n_anom']} anomalies "
              f"({b['share']:.0%}) - anom=1")
        for d in b["defs"]:
            print(f"      {_def_line(d)}")
        print(f"      SHAP: {'; '.join(b['shap'][:3])}")
    cov = sum(b["n_anom"] for b in pure)
    print(f"\n  -> {cov}/{int(y_all.sum())} anomalies ({cov / y_all.sum():.1%}) in "
          f"{len(pure)} labelreinen Baskets")
    n_stray = int(y_all.sum()) - cov
    print(f"  -> {n_stray} anomalies in mixed baskets: "
          + ", ".join(f"c{b['cl']}={b['n_anom']}/{b['n']}" for b in stray))

    meta = (f"CatBoost trained on stored labels | space={cfg['space']} | "
            f"method={method} | scaling={scaling} | k={cfg['n_clusters']} | seed={cfg['seed']} | "
            f"cases={len(y_all)}, anomalies={int(y_all.sum())}")
    write_baskets_html(os.path.join(outdir, cfg["save_baskets"]), baskets, cfg, meta)
    print(f"\n  Artefakte -> {outdir}/")
    print(f"    {cfg['save_clusters']} ({len(sel)} rows, posting_id + basket)")
    print(f"    {cfg['save_matrix']} (SHAP {X_all.shape}, float32)")
    print(f"    {cfg['save_baskets']} ({len(baskets)} baskets, definitions + sample cases)")
    return baskets


def write_baskets_html(path, baskets, cfg, meta):
    """Write a self-contained HTML report with one expandable card per basket."""
    css = """
    body{font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1a1a1a}
    h1{font-size:20px} h2{font-size:15px;margin:0 0 10px}
    .meta{color:#555;margin-bottom:18px}
    table{border-collapse:collapse;font-size:12px;margin:6px 0 12px}
    th,td{border:1px solid #ddd;padding:3px 7px;text-align:left;vertical-align:top}
    th{background:#f4f4f4}
    details{border:1px solid #ddd;border-radius:4px;margin:8px 0;padding:8px 12px}
    summary{cursor:pointer;font-weight:600}
    .anom{background:#fff4f4}.anom summary{color:#a11}
    .pure{border-left:5px solid #a11}
    ul{margin:4px 0 10px;padding-left:20px}
    .tag{display:inline-block;background:#eee;border-radius:3px;padding:1px 6px;margin-right:6px;font-size:11px}
    .l0{color:#a11;font-weight:600}
    table.sum{font-size:12px} table.sum th{position:sticky;top:0}
    """
    parts = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
             "<title>SHAP cluster baskets</title>",
             f"<style>{css}</style></head><body>",
             "<h1>SHAP cluster baskets</h1>",
             f"<div class='meta'>{htmllib.escape(meta)}</div>",
             "<h2>Summary</h2><table class='sum'><tr><th>Basket</th><th>Cases</th>"
             "<th>Anomalies</th><th>Share</th><th>Pure</th><th>Definition</th></tr>"]
    shown = [b for b in baskets if b["n_anom"] > 0 or cfg["show_all_baskets"]]
    for b in shown:
        pure = "yes" if b["n_anom"] == b["n"] else ""
        top = "; ".join(d[1] for d in b["defs"][:3])
        cls = "anom" if b["n_anom"] > 0 else ""
        parts.append(f"<tr class='{cls}'><td><b>{b['cl']}</b></td><td>{b['n']:,}</td>"
                     f"<td>{b['n_anom']}</td><td>{b['share']:.1%}</td><td>{pure}</td>"
                     f"<td>{htmllib.escape(top)}</td></tr>")
    parts.append("</table><h2>Basket details</h2>")
    for b in shown:
        cls = "anom pure" if b["n_anom"] == b["n"] else ("anom" if b["n_anom"] else "")
        parts.append(f"<details class='{cls}'><summary>Basket {b['cl']} — {b['n']:,} cases, "
                     f"{b['n_anom']} anomalies ({b['share']:.1%})</summary>")
        parts.append("<b>Definition</b><ul>")
        for d in b["defs"]:
            parts.append(f"<li>{htmllib.escape(_def_line(d))}</li>")
        parts.append("</ul><b>Mean SHAP profile</b><ul>")
        for s in b["shap"]:
            parts.append(f"<li>{htmllib.escape(s)}</li>")
        parts.append("</ul><b>Sample cases</b>")
        defcols = [d[1] for d in b["defs"]]
        parts.append("<table><tr><th>Posting ID</th><th>Label</th><th>Original amounts</th>"
                     + "".join(f"<th>{htmllib.escape(cn)}</th>" for cn in defcols)
                     + "<th>Top-3 |SHAP|</th></tr>")
        for s in b["samples"]:
            lbl = f"<span class='l0'>{s['label']}</span>" if s["label"] else str(s["label"])
            parts.append(f"<tr><td>{htmllib.escape(s['posting_id'])}</td><td>{lbl}</td>"
                         f"<td>{htmllib.escape(', '.join(f'{a:,.0f}' for a in s['amounts']))}</td>"
                         + "".join(f"<td>{s['raw'].get(cn, '')}</td>" for cn in defcols)
                         + f"<td>{htmllib.escape('; '.join(s['shap']))}</td></tr>")
        parts.append("</table></details>")
    parts.append("</body></html>")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


#pipeline

def supervised_shap_run(cfg):
    hdr("Supervised reference: CatBoost trained on stored labels")
    t0 = time.time()

    df, labels = full_dataprep()
    df_if = run_if(df, labels, verbose=False)
    y = df_if["label"].astype(int).values
    print(f"Cases: {len(df_if)} | anomalies: {int(y.sum())} ({y.mean():.2%})")
    print(f"Features: {len(get_feature_cols(df_if))}")

    n_if = int(df_if["pred_label"].sum())
    df_if["pred_label"] = df_if["label"].astype(int)
    print(f"Training target: stored labels (initial Isolation Forest anomalies: {n_if})")

    hdr("CatBoost reference: trained and evaluated in-sample")
    out = train_catboost(df_if, verbose=True, compute_shap=True, seed=cfg["seed"])
    df_cat, _imp, precision, recall, _model, tn, fp, fn, tp, shap_vals, _es = out
    print(f"\nIn-sample reference: precision={precision:.4f}, recall={recall:.4f} "
          f"(TN={tn} FP={fp} FN={fn} TP={tp})")
    print("Note: in-sample scores are an upper bound, not a generalization result; "
          "do not compare them directly with active-learning P@500.")
    print(f"CatBoost + SHAP runtime: {time.time() - t0:.1f}s")

    feat_cols = get_feature_cols(df_cat)
    if cfg["space"] == "raw":
        idx = [i for i, f in enumerate(feat_cols)
               if not any(f.startswith(p) for p in FLAG_PREFIXES)]
    elif cfg["space"] == "all":
        idx = list(range(len(feat_cols)))
    else:
        raise ValueError(f"Unknown SHAP space: {cfg['space']}")
    X = np.asarray(shap_vals[:, idx], dtype=float)
    names = [feat_cols[i] for i in idx]
    posting_id = np.asarray(df_cat["posting_id"])
    print(f"\nSHAP matrix: {X.shape[0]} x {X.shape[1]} (space={cfg['space']})")
    print("Top 10 SHAP features overall: "
          + ", ".join(f"{names[i]}={X[:, i].std():.3f}" for i in np.argsort(-X.std(axis=0))[:10]))

    X_all = X
    y_all = y
    F_all = np.asarray(df_cat[feat_cols].values, dtype=float)
    print(f"Feature matrix for basket definitions: {F_all.shape}")

    rows = None
    if cfg["max_rows"] and cfg["max_rows"] < len(X):
        rng = np.random.default_rng(cfg["seed"])
        per_class = cfg["max_rows"] // 2
        parts = []
        for cl in (0, 1):
            pool = np.where(y == cl)[0]
            parts.append(rng.choice(pool, min(per_class, len(pool)), replace=False))
        keep = np.sort(np.concatenate(parts))
        X, y, rows = X[keep], y[keep], keep
        print(f"Quick-test sample: {len(X)} stratified rows")

    if cfg["k2_reference"]:
        hdr("Reference: unsupervised k=2 on SHAP values (no labels)")
        Xs = StandardScaler().fit_transform(X) if cfg["scalings"][0] == "std" else X
        for k in (2, cfg["n_clusters"]):
            ck = KMeans(n_clusters=k, n_init=cfg["n_init"],
                        random_state=cfg["seed"]).fit_predict(Xs)
            print(f"  unsupervised k={k:<4} ARI={adjusted_rand_score(y, ck):.4f}  "
                  f"NMI={normalized_mutual_info_score(y, ck):.4f}")
        print("Question: how much class structure is present in SHAP space without labels?")

    results = []
    for scaling in cfg["scalings"]:
        Xs = StandardScaler().fit_transform(X) if scaling == "std" else X.copy()
        for method in cfg["methods"]:
            label_for_embed = y
            embed_method = method
            note = ""
            if method == "shuffled":  #control: build embedding with shuffled labels
                label_for_embed = np.random.default_rng(cfg["seed"] + 1).permutation(y)
                embed_method = "lda_within"
                note = "control: embedding uses shuffled labels, scored against stored labels"
            hdr(f"Method: {method} | scaling={scaling} | k={cfg['n_clusters']}")
            t = time.time()
            E, desc = build_embedding(Xs, label_for_embed, embed_method, cfg["lda_extra"],
                                      cfg["w_weight"], cfg["eig_floor"], cfg["seed"])
            km = KMeans(n_clusters=cfg["n_clusters"], n_init=cfg["n_init"],
                        random_state=cfg["seed"])
            c = km.fit_predict(E)
            inertia = float(km.inertia_)
            print(f"Embedding: {E.shape[1]} dimensions ({desc}{'; ' + note if note else ''})")
            print(f"K-means: {cfg['n_clusters']} clusters, inertia={inertia:.1f}, "
                  f"{time.time() - t:.1f}s")

            ext = external_metrics(y, c, cfg["seed"])
            print("\n-- Internal metrics (original SHAP space) --")
            for k, v in internal_metrics(Xs, c, cfg["sil_sample"], cfg["seed"]).items():
                print(f"   {k:<20} {v:.4f}" if isinstance(v, float) else f"   {k:<20} {v}")
            print("-- External metrics against stored labels --")
            for k, v in ext.items():
                print(f"   {k:<24} {v:.4f}")
            print("-- Cluster profiles (mean SHAP vector) --")
            print_cluster_profiles(Xs, c, y, names, cfg["top_features"], cfg["max_clusters_shown"])
            print_size_hist(c)
            print(f"Total: {time.time() - t:.1f}s")

            results.append(dict(method=method, scaling=scaling, inertia=inertia, **ext))

            if cfg["save_for"] == (method, scaling):
                export_baskets(cfg, F_all, X_all, Xs, c, y, y_all, rows, names, feat_cols,
                               posting_id, method, scaling)

    hdr("Summary: external metrics")
    print(f"{'method':<14}{'scaling':<9}{'ARI':>9}{'NMI':>9}{'v_meas':>9}{'purity':>9}{'ARI(rand)':>12}")
    for r in results:
        print(f"{r['method']:<14}{r['scaling']:<9}{r['ari']:>9.4f}{r['nmi']:>9.4f}"
              f"{r['v_measure']:>9.4f}{r['purity']:>9.4f}{r['ari_random_baseline']:>12.4f}")

    hdr("INTERPRETATION")
    print("Internal metrics use the original SHAP space and measure geometric coherence.")
    print("External metrics are optimistic for label-informed embeddings by construction.")
    print("The shuffled-label control estimates the method's baseline; the difference "
          "from the true-label run reflects label information.")
    print("Unsupervised k=2 is the label-free reference. ARI near zero means SHAP geometry "
          "does not separate the two classes on its own.")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    supervised_shap_run(CONFIG)

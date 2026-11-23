"""Research diagnostic for tree-path groups, score bins, and classification errors.

Steps:
  1. Run the standard data, Isolation Forest, and CatBoost initialization.
  2. Compare error counts across uncertainty-ranked subsets.
  3. Group cases by leaf path and summarize errors per group.
  4. Bin scores, cluster SHAP patterns, and report errors per bin and cluster.

Writes a Markdown report without changing existing files.
Run with: python -m legacy.tree_path_probe
"""

import os
import time

import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from dataprep import full_dataprep
from isolation_forest import run_if
from models.kittyboost import train_catboost
from propagation import get_feature_cols

CONFIG = dict(
    seed=42,
    verbose_catboost=True,
    n_score_bins=100,
    shap_k=3,
    top_features=5,
    max_path_groups_shown=30,
    report_dir=os.path.join(os.path.dirname(__file__), "probe_out"),
    report_md="tree_path_probe.md",
    save_paths_csv=True,
    save_shap_csv=True,
)


class Reporter:
    """Print each line and collect it for the Markdown report."""

    def __init__(self, md_path):
        self.md_path = md_path
        self.lines = []

    def __call__(self, s=""):
        s = str(s)
        print(s)
        self.lines.append(s)

    def heading(self, title, level=2):
        self("")
        self("#" * level + " " + title)
        self("")

    def table(self, header, rows):
        self("| " + " | ".join(str(h) for h in header) + " |")
        self("|" + "|".join(["---"] * len(header)) + "|")
        for r in rows:
            self("| " + " | ".join(str(x) for x in r) + " |")

    def save(self):
        os.makedirs(os.path.dirname(self.md_path) or ".", exist_ok=True)
        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


def cm_counts(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def cm_row(y_true, y_pred):
    tn, fp, fn, tp = cm_counts(y_true, y_pred)
    n = tn + fp + fn + tp
    err = (fp + fn) / n if n else 0.0
    return n, tn, fp, fn, tp, err


def path_group_confusion(inverse, n_groups, y_true, y_pred, mask):
    def bincount(sel):
        return np.bincount(inverse[mask & sel], minlength=n_groups)

    tn = bincount((y_true == 0) & (y_pred == 0))
    fp = bincount((y_true == 0) & (y_pred == 1))
    fn = bincount((y_true == 1) & (y_pred == 0))
    tp = bincount((y_true == 1) & (y_pred == 1))
    n = bincount(np.ones(len(y_true), dtype=bool))
    return n, tn, fp, fn, tp


def main(cfg):
    t0 = time.time()
    rep = Reporter(os.path.join(cfg["report_dir"], cfg["report_md"]))
    rep("# Tree-path groups, score bins, and classification errors")
    rep("")
    rep(f"Config: seed={cfg['seed']}, n_score_bins={cfg['n_score_bins']}, "
        f"shap_k={cfg['shap_k']}, top_features={cfg['top_features']}")

    rep.heading("Initial model")
    df, labels = full_dataprep()
    df_if = run_if(df, labels, verbose=False, seed=cfg["seed"])
    df_cat, _, precision, recall, model, _, _, _, _, shap_vals, _ = train_catboost(
        df_if, verbose=cfg["verbose_catboost"], compute_shap=True,
        early_stop=False, seed=cfg["seed"])

    X = df_cat.drop(columns=["label", "posting_id", "pred_label", "pred_score"])
    feat_names = list(get_feature_cols(df_cat))
    y_true = df_cat["label"].astype(int).to_numpy()
    y_pred = df_cat["pred_label"].astype(int).to_numpy()
    score = df_cat["pred_score"].to_numpy(dtype=float)
    auc = float(roc_auc_score(y_true, score))

    rep(f"Cases={len(df_cat)} | trees={model.tree_count_} | ROC-AUC={auc:.4f} | "
        f"P={precision:.4f} | R={recall:.4f}")
    rep(f"SHAP matrix: {shap_vals.shape} | features: {len(feat_names)}")

    rep.heading("Errors across margin-uncertainty subsets")
    u = 1.0 - 2.0 * np.abs(score - 0.5)
    cut75 = float(np.percentile(u, 75))
    cut50 = float(np.median(u))
    low75 = u <= cut75
    low50 = u <= cut50
    rep(f"Median Uncertainty = {cut50:.4f} | p75 = {cut75:.4f}")
    rep(f"Lowest-uncertainty 50%: {int(low50.sum())} kept, {int((~low50).sum())} excluded | "
        f"Lowest-uncertainty 75%: {int(low75.sum())} kept, {int((~low75).sum())} excluded")
    rep("")
    rows = []
    for name, mask in (("All cases", np.ones(len(y_true), dtype=bool)),
                       ("Lowest-uncertainty 75%", low75),
                       ("Lowest-uncertainty 50%", low50)):
        n_, tn_, fp_, fn_, tp_, err = cm_row(y_true[mask], y_pred[mask])
        rows.append([name, n_, tn_, fp_, fn_, tp_, f"{err:.4f}"])
    rep.table(["Subset", "n", "TN", "FP", "FN", "TP", "Error rate"], rows)

    rep.heading(f"Leaf-path groups across all {model.tree_count_} trees")
    leaf_idx = np.asarray(model.calc_leaf_indexes(Pool(X)))
    n_trees = leaf_idx.shape[1]
    rep(f"Leaf-index matrix: {leaf_idx.shape[0]} cases × {n_trees} trees (dtype {leaf_idx.dtype})")
    cont = np.ascontiguousarray(leaf_idx)
    key = cont.view(np.dtype((np.void, cont.dtype.itemsize * n_trees))).ravel()
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    n_groups = len(counts)
    rep(f"Groups sharing a path across all {n_trees} trees: {n_groups} | "
        f"singletons: {int((counts == 1).sum())} ({(counts == 1).mean():.2%}) | "
        f"largest group: {int(counts.max())}")
    rep("Group sizes: " + ", ".join(
        f"p{q}={int(np.percentile(counts, q))}" for q in (50, 90, 99)) +
        f", max={int(counts.max())}")

    for set_name, mask in (("Lowest-uncertainty 75%", low75), ("Lowest-uncertainty 50%", low50)):
        n_g, tn_g, fp_g, fn_g, tp_g = path_group_confusion(
            inverse, n_groups, y_true, y_pred, mask)
        err_g = fp_g + fn_g
        errs_low = int(err_g.sum())
        singleton_low = (counts == 1) & (n_g > 0)
        errs_single = int(err_g[singleton_low].sum())
        errs_big = errs_low - errs_single
        rep("")
        rep(f"### {set_name}")
        rep("")
        rep(f"Errors: {errs_low} | in singleton groups: {errs_single} "
            f"({errs_single / max(1, errs_low):.2%}) | in groups with at least 2 cases: {errs_big} "
            f"({errs_big / max(1, errs_low):.2%})")
        rep("")
        rep(f"Groups with most errors ({set_name}, top {cfg['max_path_groups_shown']}):")
        rep("")
        order = np.argsort(-err_g)[: int(cfg["max_path_groups_shown"])]
        rows = []
        for g in order:
            if err_g[g] == 0:
                break
            rows.append([int(g), int(counts[g]), int(n_g[g]), int(tn_g[g]), int(fp_g[g]),
                         int(fn_g[g]), int(tp_g[g]), f"{err_g[g] / max(1, n_g[g]):.3f}"])
        rep.table(["Group", "Total", "Subset", "TN", "FP", "FN", "TP",
                   "Error rate"], rows)

    if cfg["save_paths_csv"]:
        path_csv = os.path.join(cfg["report_dir"], "tree_paths_groups.csv")
        pd.DataFrame({
            "posting_id": df_cat["posting_id"].astype(str),
            "path_group": inverse,
            "path_group_size": counts[inverse],
            "pred_score": score,
            "margin_uncertainty": u,
            "lower_50": low50,
            "lower_75": low75,
            "label": y_true,
            "pred_label": y_pred,
        }).to_csv(path_csv, index=False)
        rep("")
        rep(f"CSV: {path_csv}")

    rep.heading(f"Score bins ({cfg['n_score_bins']}) and SHAP clusters")
    nb = int(cfg["n_score_bins"])
    bin_id = np.clip((score * nb).astype(int), 0, nb - 1)
    cluster_all = np.full(len(df_cat), -1, dtype=int)
    bin_rows = []
    detail_rows = []
    for b in range(nb):
        m = bin_id == b
        lo, hi = b / nb, (b + 1) / nb
        bin_label = f"[{lo:.2f},{hi:.2f})"
        n_, tn_, fp_, fn_, tp_, err = cm_row(y_true[m], y_pred[m])
        k = min(int(cfg["shap_k"]), n_)
        if k >= 2:
            Sm = StandardScaler().fit_transform(shap_vals[m])
            cl = KMeans(n_clusters=k, n_init=10, random_state=cfg["seed"]).fit_predict(Sm)
        else:
            cl = np.zeros(n_, dtype=int)
        cluster_all[m] = cl
        bin_rows.append([bin_label, n_, tn_, fp_, fn_, tp_,
                         f"{err:.3f}" if n_ else "-"])
        if n_ == 0 or (fp_ + fn_) == 0:
            continue
        for cid in np.unique(cl):
            cm = cl == cid
            cn, ctn, cfp, cfn, ctp, cerr = cm_row(y_true[m][cm], y_pred[m][cm])
            mean_abs = np.abs(shap_vals[m][cm]).mean(axis=0)
            lead = np.argsort(-mean_abs)[: int(cfg["top_features"])]
            feats = ", ".join(f"{feat_names[i]}={mean_abs[i]:.3f}" for i in lead)
            detail_rows.append([bin_label, int(cid), cn, ctn, cfp, cfn, ctp,
                                f"{cerr:.3f}", feats])
    rep.table(["Bin", "n", "TN", "FP", "FN", "TP", "Error rate"], bin_rows)
    rep("")
    rep(f"SHAP clusters for bins with errors (k={cfg['shap_k']}, capped by bin size):")
    rep("")
    if detail_rows:
        rep.table(["Bin", "Cluster", "n", "TN", "FP", "FN", "TP", "Error rate",
                    "Top |SHAP| features"], detail_rows)
    else:
        rep("No bins contain errors.")

    if cfg["save_shap_csv"]:
        shap_csv = os.path.join(cfg["report_dir"], "shap_cluster_assignments.csv")
        pd.DataFrame({
            "posting_id": df_cat["posting_id"].astype(str),
            "score_bin": bin_id,
            "shap_cluster": cluster_all,
            "pred_score": score,
            "margin_uncertainty": u,
            "lower_50": low50,
            "lower_75": low75,
            "label": y_true,
            "pred_label": y_pred,
        }).to_csv(shap_csv, index=False)
        rep("")
        rep(f"CSV: {shap_csv}")

    rep.heading("Runtime")
    rep(f"Total: {time.time() - t0:.1f}s")
    rep.save()
    print(f"\nReport saved to {rep.md_path}")


if __name__ == "__main__":
    main(CONFIG)

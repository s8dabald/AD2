"""Standalone plotting of pipeline runs from JSONL logs (kind=iter records).

Not part of the pipeline. Edit RUN_PATHS below, then: python plot_run.py

Single run  -> plots/<run_id>.png with Precision+Recall and Mean-Uncertainty.
Multiple runs -> one overlay comparison figure (accuracy per run).
"""
import json
import os

#set run log paths here
RUN_PATHS = [
    "runs/2026-10-22_22-34-07_retrain_g_shap_raw_l500.jsonl",
]
SHOW = False  #set true to open an interactive window


def load_iters(path):
    iters = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("kind") == "iter":
                iters.append(rec)
    iters.sort(key=lambda r: int(r.get("iteration", 0)))
    return iters


def _series(iters, getter):
    xs, ys = [], []
    for r in iters:
        v = getter(r)
        if v is None:
            continue
        xs.append(int(r.get("iteration", 0)))
        ys.append(float(v))
    return xs, ys


def _mean_uncertainty(r):
    unc = r.get("uncertainty")
    if not isinstance(unc, dict):
        return None
    v = unc.get("mean_uncertainty")
    return float(v) if v is not None else None


def _shade_retrain(ax, iters):
    retrain_iters = [int(r.get("iteration", 0)) for r in iters if r.get("mode") == "retrain"]
    for it in retrain_iters:
        ax.axvspan(it - 0.5, it + 0.5, color="0.85", zorder=0, linewidth=0)


def _label_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def plot_single(plt, iters, run_id, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(f"{run_id}\n({path})", fontsize=10)

    x, y = _series(iters, lambda r: r.get("precision"))
    ax1.plot(x, y, label="Precision")
    x, y = _series(iters, lambda r: r.get("recall"))
    ax1.plot(x, y, label="Recall")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.02)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    _shade_retrain(ax1, iters)

    x, y = _series(iters, _mean_uncertainty)
    if x:
        ax2.plot(x, y, color="tab:red", label="Mean Uncertainty")
        ax2.set_ylabel("Mean Uncertainty")
        ax2.legend(loc="upper right")
    else:
        ax2.text(0.5, 0.5, "Mean uncertainty is not recorded in this log",
                 ha="center", va="center", transform=ax2.transAxes, color="0.5")
        ax2.set_yticks([])
    ax2.set_xlabel("Review iteration")
    ax2.grid(True, alpha=0.3)
    _shade_retrain(ax2, iters)

    fig.tight_layout()
    return fig


def plot_overlay(plt, runs):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("Precision and recall by run", fontsize=11)
    for label, iters in runs:
        x, y = _series(iters, lambda r: r.get("precision"))
        ax1.plot(x, y, label=f"{label} (P)")
        x, y = _series(iters, lambda r: r.get("recall"))
        ax2.plot(x, y, label=f"{label} (R)")
    ax1.set_ylabel("Precision")
    ax2.set_ylabel("Recall")
    ax2.set_xlabel("Review iteration")
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def main():
    import matplotlib
    if not SHOW:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)

    if len(RUN_PATHS) == 1:
        path = RUN_PATHS[0]
        iters = load_iters(path)
        if not iters:
            raise SystemExit(f"{path}: no iteration records found")
        run_id = iters[0].get("run_id") or _label_from_path(path)
        fig = plot_single(plt, iters, run_id, path)
        out = os.path.join("plots", f"{run_id}.png")
    else:
        runs = []
        for path in RUN_PATHS:
            iters = load_iters(path)
            if not iters:
                print(f"Skipping {path}: no iteration records found")
                continue
            runs.append((_label_from_path(path), iters))
        if not runs:
            raise SystemExit("No iteration records found in the selected files")
        fig = plot_overlay(plt, runs)
        out = os.path.join("plots", "overlay.png")

    fig.savefig(out, dpi=150)
    print(f"Saved plot to {out}")
    if SHOW:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

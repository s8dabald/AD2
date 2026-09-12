import os
import json
import glob
from datetime import datetime

import pandas as pd

RUNS_DIR = "runs"
DRIVE_RUNS_DIR = "/content/drive/MyDrive/runs_AD2"


def _drive_runs_dir():
    if os.path.isdir("/content/drive/MyDrive"):
        return DRIVE_RUNS_DIR
    return None


def _run_tag(config):
    parts = [
        config.get("training_strat", ""),
        "g" if config.get("greedy_batching") else "n",
        "-".join(config.get("propagation_space") or []) or "unc",
        f"l{config.get('l')}",
    ]
    if config.get("early_stop"):
        parts.append(f"es{config.get('min_delta')}")
    return "_".join(p for p in parts if p)


def open_run(config=None):
    config = config or {}
    run_id = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{_run_tag(config)}"
    paths = [os.path.join(RUNS_DIR, f"{run_id}.jsonl")]
    drive_dir = _drive_runs_dir()
    if drive_dir is not None:
        paths.append(os.path.join(drive_dir, f"{run_id}.jsonl"))
    run = RunLog(run_id)
    run._handles = []
    for p in paths:
        parent = os.path.dirname(p)
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            run._handles.append(open(p, "a", encoding="utf-8"))
        except OSError:
            run._handles.append(None)
    run.write("run", ts=datetime.now().isoformat(timespec="seconds"), config=dict(config), run_id=run_id)
    return run


class RunLog:
    def __init__(self, run_id):
        self.run_id = run_id
        self._handles = []

    def write(self, kind, **fields):
        rec = {"kind": kind}
        rec.setdefault("ts", datetime.now().isoformat())
        rec.update(fields)
        line = json.dumps(rec, default=str)
        for h in self._handles:
            if h is not None:
                h.write(line + "\n")
                h.flush()

    def tree_count(self):
        raise NotImplementedError  # place-holder, unused

    def close(self, **summary_fields):
        rec = {"kind": "summary"}
        rec.update(summary_fields)
        line = json.dumps(rec, default=str)
        for h in self._handles:
            if h is not None:
                h.write(line + "\n")
                h.flush()
                h.close()


def load_runs(paths=None):
    """All JSONL runs -> one DataFrame (header + all records joinlined)."""
    all_paths = []
    if paths is None:
        if os.path.isdir(RUNS_DIR):
            all_paths += sorted(glob.glob(os.path.join(RUNS_DIR, "*.jsonl")))
        d = _drive_runs_dir()
        if d and os.path.isdir(d):
            all_paths += sorted(glob.glob(os.path.join(d, "*.jsonl")))
    else:
        all_paths = list(paths)

    records = []
    for p in all_paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue

    runs = {r["run_id"]: r for r in records if r.get("kind") == "run"}
    rows = []
    for r in records:
        if r.get("kind") == "run":
            continue
        row = dict(runs.get(r["run_id"], {}).get("config", {})) if r.get("run_id") else {}
        row.update({k: v for k, v in r.items() if k not in ("kind",)})
        row["kind"] = r.get("kind")
        rows.append(row)
    return pd.DataFrame(rows)
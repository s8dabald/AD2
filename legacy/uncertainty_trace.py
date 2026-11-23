"""Sidecar trace for uncertainty changes of propagated cases.

Writes one JSONL file per run under runs/traces/. The main run JSONL stays
untouched and logstore.load_runs() does not pick these files up.
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import logstore

SCHEMA_VERSION = 1
TRACE_DIR = "traces"
CONTROL_SAMPLE_PER_DECILE = 100


def _key(posting_id):
    return str(posting_id)


def _py(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _rank_pct(values):
    return pd.Series(np.asarray(values, dtype=float)).rank(
        method="average", pct=True).to_numpy(dtype=float)


def _deciles(rank_pct):
    return np.clip((np.asarray(rank_pct, dtype=float) * 10.0).astype(int), 0, 9)


def _stats(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q90": float(np.quantile(arr, 0.90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _group_stats(records, value_key):
    groups = {}
    for rec in records:
        groups.setdefault(rec["group"], []).append(float(rec[value_key]))
    return {g: _stats(v) for g, v in groups.items()}


class UncertaintyTrace:
    """Best-effort JSONL trace. Trace errors never abort the pipeline."""

    def __init__(self, run_id, seed=42, strategy="margin", retrain_every=1,
                 propagation_mode=None):
        self.run_id = run_id
        self.seed = int(seed)
        self.strategy = strategy
        self.retrain_every = retrain_every
        self.propagation_mode = propagation_mode
        self.last_retrain_iter = 0
        self.event_seq = 0
        self.pending = {}
        self.pending_seeds = {}
        self.case_state = {}
        self.enabled = True
        self._handles = []

        paths = [os.path.join(logstore.RUNS_DIR, TRACE_DIR, f"{run_id}.jsonl")]
        if os.path.isdir("/content/drive/MyDrive"):
            paths.append(os.path.join(logstore.DRIVE_RUNS_DIR, TRACE_DIR,
                                      f"{run_id}.jsonl"))

        for path in paths:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            try:
                self._handles.append(open(path, "a", encoding="utf-8"))
            except OSError as exc:
                print(f"[uncertainty-trace] cannot open {path}: {exc}")

        self._write(
            "trace_header",
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            seed=self.seed,
            strategy=strategy,
            retrain_every=retrain_every,
            propagation_mode=propagation_mode,
        )

    def _disable(self, where, exc):
        self.enabled = False
        print(f"[uncertainty-trace] disabled in {where}: {exc!r}")

    def _write(self, kind, **fields):
        if not self.enabled:
            return
        rec = {"kind": kind}
        rec.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        rec["run_id"] = self.run_id
        rec.update({k: _py(v) for k, v in fields.items()})
        line = json.dumps(rec, default=str)
        for handle in self._handles:
            if handle is None:
                continue
            try:
                handle.write(line + "\n")
                handle.flush()
            except OSError as exc:
                print(f"[uncertainty-trace] write failed: {exc}")

    def on_event(self, iteration, seed_id, ids, roles, prev_pred, new_pred,
                 true_labels, u, u_rank, delta, first_event, family_size):
        if not self.enabled:
            return
        try:
            self._on_event(iteration, seed_id, ids, roles, prev_pred, new_pred,
                           true_labels, u, u_rank, delta, first_event,
                           family_size)
        except Exception as exc:  #keep the pipeline alive
            self._disable("on_event", exc)

    def _on_event(self, iteration, seed_id, ids, roles, prev_pred, new_pred,
                  true_labels, u, u_rank, delta, first_event, family_size):
        ids = np.asarray(ids)
        roles = np.asarray(roles)
        prev_pred = np.asarray(prev_pred)
        new_pred = np.asarray(new_pred)
        true_labels = np.asarray(true_labels)
        u = np.asarray(u, dtype=float)
        delta = np.asarray(delta, dtype=float)
        if u_rank is None:
            u_rank = np.full(len(u), np.nan)
        else:
            u_rank = np.asarray(u_rank, dtype=float)
        first_event = np.asarray(first_event, dtype=bool)

        for i, posting_id in enumerate(ids):
            key = _key(posting_id)
            self.event_seq += 1
            event_id = self.event_seq
            role = str(roles[i])
            changed = bool(prev_pred[i] != new_pred[i])
            was_correct = bool(prev_pred[i] == true_labels[i])
            is_correct = bool(new_pred[i] == true_labels[i])
            if role == "seed":
                outcome = "direct"
            elif changed:
                outcome = ("prop_changed_correct" if is_correct
                           else "prop_changed_wrong")
            else:
                outcome = ("prop_noop_correct" if is_correct
                           else "prop_noop_wrong")

            state = self.case_state.setdefault(
                key, {"n_events": 0, "first_outcome": None,
                      "latest_outcome": None, "ever_wrong": False})
            state["n_events"] += 1
            if state["first_outcome"] is None:
                state["first_outcome"] = outcome
            state["latest_outcome"] = outcome
            state["ever_wrong"] = bool(state["ever_wrong"] or not is_correct)

            old = self.pending.get(key)
            n_events_interval = 1 if old is None else old["n_events_interval"] + 1
            event = {
                "event_id": event_id,
                "iteration": int(iteration),
                "interval_first_event_iteration": (
                    int(iteration) if old is None
                    else old["interval_first_event_iteration"]),
                "last_retrain_iter": int(self.last_retrain_iter),
                "posting_id": posting_id,
                "seed_id": seed_id,
                "role": role,
                "prev_pred_label": int(prev_pred[i]),
                "interval_prev_pred_label": (
                    int(prev_pred[i]) if old is None
                    else old["interval_prev_pred_label"]),
                "new_pred_label": int(new_pred[i]),
                "true_label": int(true_labels[i]),
                "changed": changed,
                "was_correct": was_correct,
                "is_correct": is_correct,
                "u_before": float(u[i]),
                "u_rank_before": float(u_rank[i]),
                "delta_assigned": float(delta[i]),
                "first_event": bool(first_event[i]),
                "family_size": int(family_size),
                "case_event_no": int(state["n_events"]),
                "n_events_interval": int(n_events_interval),
                "outcome": outcome,
            }
            self.pending[key] = event
            if role == "seed":
                self.pending_seeds[key] = event
            self._write("event", **event)

    def on_retrain(self, iteration, ids, u_before, u_after, pred_before,
                   pred_after, true_labels, covered, directly_corrected, seed):
        if not self.enabled:
            return
        try:
            self._on_retrain(iteration, ids, u_before, u_after, pred_before,
                             pred_after, true_labels, covered,
                             directly_corrected, seed)
        except Exception as exc:  #keep the pipeline alive
            self._disable("on_retrain", exc)

    def _on_retrain(self, iteration, ids, u_before, u_after, pred_before,
                    pred_after, true_labels, covered, directly_corrected, seed):
        if u_before is None or u_after is None:
            return
        ids = np.asarray(ids)
        u_before = np.asarray(u_before, dtype=float)
        u_after = np.asarray(u_after, dtype=float)
        pred_before = np.asarray(pred_before)
        pred_after = np.asarray(pred_after)
        true_labels = np.asarray(true_labels)

        rank_before = _rank_pct(u_before)
        rank_after = _rank_pct(u_after)
        decile_before = _deciles(rank_before)
        pos = {_key(pid): i for i, pid in enumerate(ids)}
        covered_keys = {_key(pid) for pid in covered}
        direct_keys = {_key(pid) for pid in directly_corrected}

        control_indices = np.asarray(
            [i for i, pid in enumerate(ids)
             if _key(pid) not in covered_keys
             and _key(pid) not in direct_keys], dtype=int)
        control_delta = (u_after - u_before)
        control_by_decile = {}
        for decile in range(10):
            vals = control_delta[
                control_indices[decile_before[control_indices] == decile]]
            control_by_decile[str(decile)] = _stats(vals)

        seed_resolutions = {}
        for key, event in self.pending_seeds.items():
            i = pos.get(key)
            if i is None:
                self._write("seed_unresolved", **event)
                continue
            decile = int(decile_before[i])
            ctrl = control_by_decile.get(str(decile), {})
            seed_rec = {
                "event_id": event["event_id"],
                "iteration": event["iteration"],
                "retrain_iteration": int(iteration),
                "posting_id": event["posting_id"],
                "seed_id": event["seed_id"],
                "u_before": event["u_before"],
                "u_before_retrain": float(u_before[i]),
                "u_after": float(u_after[i]),
                "delta_u": float(u_after[i] - event["u_before"]),
                "u_decile_before": decile,
                "control_delta_mean_decile": ctrl.get("mean"),
            }
            seed_resolutions[key] = seed_rec
            self._write("seed_resolution", **seed_rec)

        resolved = []
        for key, event in list(self.pending.items()):
            i = pos.get(key)
            if i is None:
                self._write("event_unresolved", **event)
                continue
            decile = int(decile_before[i])
            ctrl = control_by_decile.get(str(decile), {})
            rec = {
                "event_id": event["event_id"],
                "iteration": event["iteration"],
                "interval_first_event_iteration": event[
                    "interval_first_event_iteration"],
                "retrain_iteration": int(iteration),
                "posting_id": event["posting_id"],
                "seed_id": event["seed_id"],
                "role": event["role"],
                "outcome": event["outcome"],
                "group": event["outcome"],
                "prev_pred_label": event["prev_pred_label"],
                "interval_prev_pred_label": event[
                    "interval_prev_pred_label"],
                "new_pred_label": event["new_pred_label"],
                "true_label": event["true_label"],
                "changed": event["changed"],
                "was_correct": event["was_correct"],
                "is_correct": event["is_correct"],
                "u_before": event["u_before"],
                "u_before_retrain": float(u_before[i]),
                "u_after": float(u_after[i]),
                "delta_u": float(u_after[i] - event["u_before"]),
                "u_event_gap": float(u_before[i] - event["u_before"]),
                "u_rank_before": event["u_rank_before"],
                "u_rank_after": float(rank_after[i]),
                "delta_assigned": event["delta_assigned"],
                "family_size": event["family_size"],
                "first_event": event["first_event"],
                "case_event_no": event["case_event_no"],
                "u_decile_before": decile,
                "control_delta_mean_decile": ctrl.get("mean"),
                "seed_delta_u": seed_resolutions.get(
                    _key(event["seed_id"]), {}).get("delta_u"),
                "seed_control_delta_mean_decile": seed_resolutions.get(
                    _key(event["seed_id"]), {}).get(
                        "control_delta_mean_decile"),
                "pred_after_retrain": int(pred_after[i]),
                "is_correct_after": bool(pred_after[i] == true_labels[i]),
                "n_events_interval": int(event["n_events_interval"]),
            }
            resolved.append(rec)
            self._write("event_resolution", **rec)

        for key, state in self.case_state.items():
            i = pos.get(key)
            if i is None:
                continue
            decile = int(decile_before[i])
            self._write(
                "cohort_retrain",
                retrain_iteration=int(iteration),
                posting_id=ids[i],
                cohort=state["first_outcome"],
                latest_outcome=state["latest_outcome"],
                ever_wrong=state["ever_wrong"],
                n_events=int(state["n_events"]),
                u_before=float(u_before[i]),
                u_after=float(u_after[i]),
                delta_u=float(u_after[i] - u_before[i]),
                u_rank_before=float(rank_before[i]),
                u_rank_after=float(rank_after[i]),
                u_decile_before=decile,
                pred_before=int(pred_before[i]),
                pred_after=int(pred_after[i]),
                is_correct_before=bool(pred_before[i] == true_labels[i]),
                is_correct_after=bool(pred_after[i] == true_labels[i]),
            )

        rng = np.random.RandomState((self.seed + int(iteration) * 1009) % (2**31 - 1))
        for decile in range(10):
            idx = control_indices[decile_before[control_indices] == decile]
            if idx.size > CONTROL_SAMPLE_PER_DECILE:
                idx = rng.choice(idx, size=CONTROL_SAMPLE_PER_DECILE,
                                 replace=False)
            for i in np.sort(idx):
                self._write(
                    "control_sample",
                    retrain_iteration=int(iteration),
                    posting_id=ids[i],
                    group="control_never",
                    u_before=float(u_before[i]),
                    u_after=float(u_after[i]),
                    delta_u=float(u_after[i] - u_before[i]),
                    u_rank_before=float(rank_before[i]),
                    u_rank_after=float(rank_after[i]),
                    u_decile_before=int(decile),
                    pred_before=int(pred_before[i]),
                    pred_after=int(pred_after[i]),
                    is_correct_before=bool(pred_before[i] == true_labels[i]),
                    is_correct_after=bool(pred_after[i] == true_labels[i]),
                )

        self._write(
            "retrain_summary",
            retrain_iteration=int(iteration),
            n_cases=int(len(ids)),
            n_pending_events=int(len(resolved)),
            n_covered_ever=int(len(covered_keys)),
            n_direct_ever=int(len(direct_keys)),
            n_control_cases=int(control_indices.size),
            global_u_before=_stats(u_before),
            global_u_after=_stats(u_after),
            global_delta_u=_stats(u_after - u_before),
            groups=_group_stats(resolved, "delta_u"),
            control_by_decile=control_by_decile,
        )

        self.pending.clear()
        self.pending_seeds.clear()
        self.last_retrain_iter = int(iteration)

    def close(self, iteration=None):
        if self.enabled:
            for event in self.pending.values():
                self._write("event_unresolved", **event)
            for event in self.pending_seeds.values():
                self._write("seed_unresolved", **event)
            self._write(
                "trace_summary",
                last_iteration=iteration,
                last_retrain_iter=int(self.last_retrain_iter),
                n_events=int(self.event_seq),
                n_cases_with_events=int(len(self.case_state)),
                n_pending=int(len(self.pending)),
                n_pending_seeds=int(len(self.pending_seeds)),
            )
        self.enabled = False
        for handle in self._handles:
            if handle is None:
                continue
            try:
                handle.close()
            except OSError:
                pass
        self._handles = []

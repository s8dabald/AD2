"""Thread-safe UI bridge into the canonical root AD2 workflow."""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

import numpy as np
import pandas as pd

from main import run_supervised


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "journal_entries.csv"
REVIEW_BLOCK_SIZE = 25
REVIEW_BUDGET = 500


class PipelineSession:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.thread: threading.Thread | None = None
        self.run_id = "active"
        self.phase = "idle"
        self.error: str | None = None
        self.review_count = 0
        self.total_review_count = 0
        self.review_target = REVIEW_BLOCK_SIZE
        self.total_review_target = REVIEW_BUDGET
        self.active_review: dict[str, Any] | None = None
        self.pending_decision: dict[str, Any] | None = None
        self.review_history: list[dict[str, Any]] = []
        self._source_df: pd.DataFrame | None = None

    def start(self) -> None:
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError("A pipeline run is already active")
            self.error = None
            self.phase = "initial_training"
            self.review_count = 0
            self.total_review_count = 0
            self.active_review = None
            self.pending_decision = None
            self.review_history = []
            self.thread = threading.Thread(
                target=self._run, name="ad2-canonical-pipeline", daemon=True
            )
            self.thread.start()

    def submit_review(
        self,
        posting_id: str,
        label: str,
        feature_choices: dict[str, str],
    ) -> None:
        with self.condition:
            if self.phase != "review" or self.active_review is None:
                raise RuntimeError("The pipeline is not waiting for a reviewer decision")
            if str(posting_id) != str(self.active_review["posting_id"]):
                raise RuntimeError("The active case has changed")
            if self.pending_decision is not None:
                raise RuntimeError("A reviewer decision is already pending")
            if label not in {"Normal", "Anomaly"}:
                raise ValueError("label must be 'Normal' or 'Anomaly'")

            visible_features = {row["feature"] for row in self.active_review["shap"]}
            unknown = set(feature_choices) - visible_features
            if unknown:
                raise ValueError(f"Unknown SHAP features: {sorted(unknown)[:5]}")
            allowed = {"ignore", "delta_0.25", "delta_0.10", "delta_0.05"}
            invalid = {
                feature: choice for feature, choice in feature_choices.items()
                if choice not in allowed
            }
            if invalid:
                raise ValueError(f"Invalid per-feature choices: {invalid}")

            self.pending_decision = {
                "label": label,
                "feature_choices": dict(feature_choices),
            }
            self.condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "run_id": self.run_id,
                "phase": self.phase,
                "training": self.phase in {"initial_training", "retrain"},
                "review_count": self.review_count,
                "review_target": self.review_target,
                "total_review_count": self.total_review_count,
                "total_review_target": self.total_review_target,
                "case": self._case_snapshot(),
                "review_history": list(self.review_history[-20:]),
                "error": self.error,
            }

    def _run(self) -> None:
        try:
            run_supervised(
                training_strat="retrain",
                l=REVIEW_BUDGET,
                corrected_weights=100,
                corrected_saved=True,
                strategy="margin",
                return_full_data=False,
                greedy_batching=False,
                propagation_space=["shap_raw"],
                prop_weight_mode="uniform",
                selection_mode="uncertainty",
                seed=42,
                retrain_every=REVIEW_BLOCK_SIZE,
                propagation_mode="shap_box_selection",
                split_training=True,
                policy_mode="manual",
                review_callback=self._review_provider,
                phase_callback=self._phase_callback,
            )
            self._phase_callback("complete", REVIEW_BUDGET, REVIEW_BUDGET,
                                 REVIEW_BLOCK_SIZE)
        except Exception as exc:
            with self.condition:
                self.error = f"{type(exc).__name__}: {exc}"
                self.phase = "error"
                self.active_review = None
                self.condition.notify_all()

    def _review_provider(self, proposal: dict[str, Any]) -> dict[str, Any]:
        posting_id = str(proposal["posting_id"])
        case = {
            **proposal,
            "posting_id": posting_id,
            "predicted_label": (
                "Anomaly" if int(proposal["predicted_label"]) else "Normal"
            ),
            "journal_entries": self._journal_entries(posting_id),
        }
        case["derived_features"] = self._derived_source_features(case["journal_entries"])
        with self.condition:
            block_progress = (int(proposal["iteration"]) - 1) % self.review_target
            self.phase = "review"
            self.total_review_count = int(proposal["iteration"]) - 1
            self.review_count = block_progress
            self.active_review = case
            self.pending_decision = None
            self.condition.notify_all()

            while self.pending_decision is None and self.phase not in {"error", "idle"}:
                self.condition.wait()
            if self.phase == "error":
                raise RuntimeError(self.error or "Pipeline stopped during review")
            if self.pending_decision is None:
                raise RuntimeError("Pipeline stopped before review decision")

            decision = self.pending_decision
            self.pending_decision = None
            self.review_history.append({
                "posting_id": posting_id,
                "label": decision["label"],
                "iteration": int(proposal["iteration"]),
            })
            self.total_review_count = int(proposal["iteration"])
            self.review_count = int(proposal["iteration"]) % self.review_target
            if self.review_count == 0:
                self.review_count = self.review_target
            self.active_review = None
            self.condition.notify_all()
            return decision

    def _phase_callback(
        self,
        phase: str,
        completed_reviews: int | None,
        total_reviews: int,
        block_size: int,
    ) -> None:
        with self.condition:
            self.phase = phase
            if completed_reviews is not None:
                self.total_review_count = int(completed_reviews)
                count = int(completed_reviews) % block_size
                self.review_count = block_size if count == 0 and completed_reviews else count
            self.condition.notify_all()

    def _journal_entries(self, posting_id: str) -> list[dict[str, Any]]:
        if self._source_df is None:
            if not SOURCE_PATH.is_file():
                raise FileNotFoundError(f"Journal source not found: {SOURCE_PATH}")
            self._source_df = pd.read_csv(SOURCE_PATH, sep=";")

        rows = self._source_df[
            self._source_df["posting_id"].astype(str) == str(posting_id)
        ]
        #marking leaks target-like information, so keep it hidden
        excluded = {"posting_id", "label", "marking"}
        columns = [column for column in self._source_df.columns if column not in excluded]
        return [
            {column: self._json_value(row[column]) for column in columns}
            for _, row in rows.iterrows()
        ]

    @staticmethod
    def _derived_source_features(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not entries:
            return []
        first = entries[0]
        document_date = pd.to_datetime(
            first.get("document_date"), format="mixed", errors="coerce"
        )
        entered_date = pd.to_datetime(
            first.get("entered_date"), format="mixed", errors="coerce"
        )
        if pd.isna(document_date) or pd.isna(entered_date):
            return []
        return [{
            "feature": "date_difference",
            "value": int((entered_date - document_date).days),
            "description": "Days from document date to entered date",
        }]

    def _case_snapshot(self) -> dict[str, Any] | None:
        if self.active_review is None:
            return None
        return self.active_review

    @staticmethod
    def _json_value(value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, (pd.Timestamp,)):
            return value.isoformat()
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value


session = PipelineSession()

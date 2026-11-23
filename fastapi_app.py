from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from pipeline_session import session


class ReviewRequest(BaseModel):
    posting_id: str
    label: str
    feature_choices: dict[str, str] = Field(default_factory=dict)


class StartRequest(BaseModel):
    run_id: str = "active"


app = FastAPI(title="AD2 Review API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/run/state")
def get_state() -> dict[str, Any]:
    return session.snapshot()


@app.post("/api/run/start")
def start_run(request: StartRequest) -> dict[str, Any]:
    session.run_id = request.run_id
    try:
        session.start()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return session.snapshot()


@app.post("/api/review")
def review_case(request: ReviewRequest) -> dict[str, Any]:
    try:
        session.submit_review(
            request.posting_id,
            request.label,
            request.feature_choices,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return session.snapshot()


UI_DIR = Path(__file__).resolve().parent / "ad2-review-ui"
app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="review-ui")


if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="127.0.0.1", port=8000, reload=False)

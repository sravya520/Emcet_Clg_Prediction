"""The HTTP API. Two doors in, and only one of them needs the AI.

    GET  /health     is the service up, and is chat usable right now
    GET  /meta       data year, phase, band limits, measured accuracy, disclaimer
    GET  /filters    the branch, district and category lists for the form
    POST /recommend  Safe / Moderate / Reach - DETERMINISTIC, no AI, no key
    POST /chat       the agent

The split is the whole point. /recommend is the product: it reads the table
and does arithmetic, so it works with no API key, no internet and no quota
left. /chat is a convenience on top. If the AI breaks, /recommend must not
even notice.

Run:  uvicorn copilot.api:app --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from copilot import config
from copilot.agent import tools
from copilot.agent.loop import ERROR_KINDS, classify_error
from copilot.engine import banding

log = logging.getLogger("copilot.api")

app = FastAPI(
    title="Counselling Copilot",
    description=(
        "AP EAPCET college options from official closing ranks. "
        "The /recommend endpoint never uses AI."
    ),
    version="0.1.0",
)


# --- Request models ---------------------------------------------------------


class RecommendRequest(BaseModel):
    rank: int = Field(ge=1, description="EAPCET rank")
    category: str = Field(description="OC, BC-A..BC-E, SC-I..SC-III, ST, OC-EWS")
    gender: str = Field(description="BOYS or GIRLS")
    local_area: str = Field(description="AU or SVU")
    branch: list[str] | None = None
    district: list[str] | None = None
    college_type: list[str] | None = None
    per_band: int = Field(default=5, ge=1, le=50)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


# --- The deterministic half -------------------------------------------------


@app.get("/")
def index() -> dict:
    """What lives here, for anyone who opens the API root in a browser.

    Worth having: the API has no web page, so a bare 404 here is genuinely
    confusing - it reads as "the service is broken" when the service is fine
    and you are simply at the wrong address. Say where the app is instead.
    """
    return {
        "service": "Counselling Copilot API",
        "note": (
            "This is the API, not the app. There is no web page at this "
            "address. Open the Streamlit interface instead."
        ),
        "the_app_is_at": "http://127.0.0.1:8501 (local) - see /docs to explore this API",
        "endpoints": {
            "GET /docs": "interactive API explorer",
            "GET /health": "is the service up, and is chat usable",
            "GET /meta": "data year, band limits, measured accuracy, disclaimers",
            "GET /filters": "dropdown values for the form",
            "POST /recommend": "Safe/Moderate/Reach - deterministic, no AI",
            "POST /chat": "the agent",
        },
        "data_year": config.RECOMMEND_YEAR,
    }


@app.get("/health")
def health() -> dict:
    """Up, and is chat usable? The form does not depend on any of this."""
    return {
        "status": "ok",
        "data_year": config.RECOMMEND_YEAR,
        "exam_state": config.EXAM_STATE,
        # Whether a key is PRESENT, never what it is.
        "chat_configured": bool(config.GEMINI_API_KEY and config.GEMINI_MODEL),
        "chat_model": config.GEMINI_MODEL or None,
        "form_available": True,
    }


@app.get("/meta")
def meta() -> dict:
    """Everything the UI needs to caption itself honestly."""
    bands = tools.explain_bands()
    return {
        "data_year": config.RECOMMEND_YEAR,
        "counselling_phase": "end_of_web_counselling",
        "bands": bands["bands"],
        "holdout_accuracy": bands.get("holdout_accuracy"),
        "holdout_note": bands.get("holdout_note"),
        "tuned_on": bands.get("tuned_on"),
        "tested_on": bands.get("tested_on"),
        "girls_rule": bands["girls_rule"],
        "sc_warning": banding.SC_WARNING,
        "source_disclaimer": bands["source_disclaimer"],
        "thin_data_categories": sorted(tools.THIN_DATA_CATEGORIES),
        "band_accuracy_by_category": tools.BAND_ACCURACY_BY_CATEGORY,
        "accuracy_not_measured_note": tools.ACCURACY_NOT_MEASURED,
        "special_quota_notice": tools.SPECIAL_QUOTA_NOTICE,
        "branch_names_are_unofficial": True,
        "branch_name_note": (
            "No official list of branch code to branch name exists in any "
            "counselling document, so every branch name here is unofficial. "
            "Where we were not confident, the raw code is shown instead."
        ),
        "fee_note": (
            "Fees are not shown. The 2025 statement has no fee column, and fees "
            "changed for 45% of colleges inside the last fee block period, so no "
            "honest 2025 figure exists."
        ),
    }


@app.get("/filters")
def filters() -> dict:
    """Dropdown contents, straight from the data rather than hardcoded."""
    frame = tools.cutoffs()
    year = frame[frame["year"] == config.RECOMMEND_YEAR]

    branches = (
        year[["branch_code", "branch_name"]]
        .drop_duplicates()
        .sort_values("branch_code")
    )
    return {
        "categories": sorted(year["category"].dropna().unique().tolist()),
        "genders": ["BOYS", "GIRLS"],
        "local_areas": sorted(year["local_area"].dropna().unique().tolist()),
        "districts": sorted(year["district"].dropna().unique().tolist()),
        "college_types": sorted(year["college_type"].dropna().unique().tolist()),
        "branches": [
            {
                "code": row.branch_code,
                "label": f"{row.branch_code} - {row.branch_name}"
                if isinstance(row.branch_name, str) and row.branch_name
                else row.branch_code,
            }
            for row in branches.itertuples()
        ],
    }


@app.post("/recommend")
def recommend(request: RecommendRequest) -> dict:
    """Safe / Moderate / Reach. No AI anywhere in this path."""
    result = tools.recommend_options(
        rank=request.rank,
        category=request.category,
        gender=request.gender,
        local_area=request.local_area,
        branch=request.branch,
        district=request.district,
        college_type=request.college_type,
        per_band=request.per_band,
    )
    return result


# --- The AI half ------------------------------------------------------------


@app.post("/chat")
def chat(request: ChatRequest) -> JSONResponse:
    """The agent. Every failure here is reported as data, never as a 500.

    The UI needs to tell the student what to do next, and "out of quota" and
    "bad key" need different advice. So the failure kind is part of the normal
    response body rather than an HTTP error the UI has to guess at.
    """
    if not (config.GEMINI_API_KEY and config.GEMINI_MODEL):
        return JSONResponse(
            {
                "ok": False,
                "error_kind": "not_configured",
                "message": ERROR_KINDS["not_configured"],
                "form_still_works": True,
            }
        )

    # Imported here so that a missing or broken SDK cannot stop /recommend
    # from serving.
    from copilot.agent.loop import ask

    try:
        turn = ask(request.message)
    except Exception as error:  # noqa: BLE001
        kind = classify_error(f"{type(error).__name__}: {error}")
        log.warning("chat failed (%s)", kind)
        return JSONResponse(
            {"ok": False, "error_kind": kind, "message": ERROR_KINDS[kind],
             "form_still_works": True}
        )

    if turn.error:
        kind = classify_error(turn.error)
        log.warning("chat failed (%s)", kind)
        return JSONResponse(
            {"ok": False, "error_kind": kind, "message": ERROR_KINDS[kind],
             "form_still_works": True}
        )

    checked = turn.checked
    return JSONResponse(
        {
            "ok": True,
            "reply": checked.answer.reply,
            "recommendations": [r.model_dump() for r in checked.answer.recommendations],
            "data_year": checked.answer.data_year or config.RECOMMEND_YEAR,
            "out_of_scope": checked.answer.out_of_scope,
            "sc_warning_shown": checked.answer.sc_warning_shown,
            "removed": [item.model_dump() for item in checked.removed],
            "tool_calls": turn.tool_calls,
            "steps_used": turn.steps_used,
            "seconds": round(turn.latency_seconds, 2),
            "model": config.GEMINI_MODEL,
        }
    )

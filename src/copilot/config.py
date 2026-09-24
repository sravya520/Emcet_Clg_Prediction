"""Central configuration.

Everything that might differ between states, years or environments lives here so
that nothing downstream hardcodes it.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Read .env if present. Real environment variables always win, so deployment
# platforms that inject secrets directly are unaffected.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

# --- Exam / state -----------------------------------------------------------
# AP only for now. TG EAPCET would be added as a second state, not by editing
# the pipeline: the raw manifest and the mapping files are per-state.
EXAM_STATE: str = os.getenv("EXAM_STATE", "AP")

# The year whose closing ranks are used to make live recommendations.
# Earlier years exist for the backtest only.
RECOMMEND_YEAR: int = int(os.getenv("RECOMMEND_YEAR", "2025"))

# --- Paths ------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
MAPPINGS_DIR: Path = DATA_DIR / "mappings"
PROCESSED_DIR: Path = DATA_DIR / "processed"

SOURCES_MANIFEST: Path = RAW_DIR / "sources.json"
CUTOFFS_PARQUET: Path = PROCESSED_DIR / "cutoffs.parquet"
CUTOFFS_DB: Path = PROCESSED_DIR / "copilot.db"
VALIDATION_REPORT: Path = PROCESSED_DIR / "validation_report.md"
BACKTEST_REPORT: Path = REPO_ROOT / "docs" / "BACKTEST.md"

BRANCH_MAP: Path = MAPPINGS_DIR / "branch_map.csv"
COLLEGE_MAP: Path = MAPPINGS_DIR / "college_map.csv"
THRESHOLDS: Path = MAPPINGS_DIR / "thresholds.json"
#: Written AFTER the hold-out runs, so thresholds.json stays locked beforehand.
BACKTEST_RESULTS: Path = MAPPINGS_DIR / "backtest_results.json"

# --- LLM --------------------------------------------------------------------
# Never hardcode a model id, and never print the key. See CLAUDE.md.
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

#: Stop the loop after this many tool round-trips and force a final answer.
MAX_AGENT_STEPS: int = int(os.getenv("MAX_AGENT_STEPS", "5"))


def missing_llm_settings() -> list[str]:
    """Which LLM settings are absent. Names only - never values.

    Chat needs BOTH a key and a model name. An earlier version reported any
    failure as "no API key", which was actively misleading when the key was
    present and the model name was not: it sent someone hunting for a problem
    with a key that was fine.
    """
    return [
        name
        for name, value in (
            ("GEMINI_API_KEY", GEMINI_API_KEY),
            ("GEMINI_MODEL", GEMINI_MODEL),
        )
        if not value
    ]


def require_llm_settings() -> tuple[str, str]:
    """Return (api_key, model) or explain exactly what is missing.

    Deliberately never includes the key itself in any message it raises.
    """
    missing = missing_llm_settings()
    if missing:
        raise RuntimeError(
            f"Missing {' and '.join(missing)}. Copy .env.example to .env and fill it in. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    return GEMINI_API_KEY, GEMINI_MODEL

# --- Values that are missing, spelled every way the sources spell them -------
NULL_TOKENS: frozenset[str] = frozenset(
    {"", "na", "n/a", "nan", "none", "-", "--", "nil"}
)

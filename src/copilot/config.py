"""Central configuration.

Everything that might differ between states, years or environments lives here so
that nothing downstream hardcodes it.
"""

from __future__ import annotations

import os
from pathlib import Path

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

BRANCH_MAP: Path = MAPPINGS_DIR / "branch_map.csv"
COLLEGE_MAP: Path = MAPPINGS_DIR / "college_map.csv"
CATEGORY_MAP: Path = MAPPINGS_DIR / "category_map.csv"

# --- LLM (used from Phase 3) ------------------------------------------------
# Never hardcode a model id. See CLAUDE.md.
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "")

# --- Values that are missing, spelled every way the sources spell them -------
NULL_TOKENS: frozenset[str] = frozenset(
    {"", "na", "n/a", "nan", "none", "-", "--", "nil"}
)

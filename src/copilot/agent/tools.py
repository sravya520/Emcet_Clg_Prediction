"""The four things the AI is allowed to do.

Every one is ordinary Python that reads the table built in Stage 2 and the
limits measured in Stage 3. The AI does not compute anything here; it only
decides which of these to call, with which arguments, and then writes prose
around whatever comes back.

Each tool returns plain dicts and lists so the result can be handed to the
model as JSON and, just as importantly, scanned by the checker afterwards.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from copilot import config
from copilot.engine import banding
from copilot.engine.banding import Thresholds, recommend

#: Subjects we hold no data on. Asking about these must produce an honest
#: refusal, never a guess. Fees are here because the 2025 statement dropped the
#: column and fees changed for 45% of colleges inside the fee block, so no
#: honest 2025 figure exists.
OUT_OF_SCOPE = {
    "fee": "fees, tuition or cost",
    "placement": "placements, salaries or recruiters",
    "quality": "whether a college is good, its ranking or its reputation",
    "hostel": "hostels, campus life or facilities",
    "faculty": "teaching quality or faculty",
    "admission_process": "counselling dates, documents or the admission process",
}

_CUTOFFS: pd.DataFrame | None = None


def cutoffs() -> pd.DataFrame:
    """Load the table once and keep it in memory."""
    global _CUTOFFS
    if _CUTOFFS is None:
        _CUTOFFS = pd.read_parquet(config.CUTOFFS_PARQUET)
    return _CUTOFFS


def _clean(value: Any) -> Any:
    """Make a pandas value safe to put in JSON."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


#: People write categories many ways. Map the spellings we accept onto the exact
#: labels used in the table. Anything not listed is passed through untouched and
#: will simply match nothing, which is the honest outcome for a category we do
#: not hold.
CATEGORY_ALIASES: dict[str, str] = {
    "OC": "OC",
    "GENERAL": "OC",
    "OPEN": "OC",
    "EWS": "OC-EWS",
    "OCEWS": "OC-EWS",
    "OC-EWS": "OC-EWS",
    "ST": "ST",
    "SC": "SC",
    "SC1": "SC-I", "SCI": "SC-I", "SC-1": "SC-I", "SC-I": "SC-I",
    "SC2": "SC-II", "SCII": "SC-II", "SC-2": "SC-II", "SC-II": "SC-II",
    "SC3": "SC-III", "SCIII": "SC-III", "SC-3": "SC-III", "SC-III": "SC-III",
    **{
        alias: f"BC-{letter}"
        for letter in "ABCDE"
        for alias in (f"BC{letter}", f"BC-{letter}", f"BC_{letter}")
    },
}


def normalise_category(category: str) -> str:
    """Turn 'bcb', 'BC B', 'BC-B' into the table's 'BC-B'."""
    key = str(category).upper().replace(" ", "").replace(".", "")
    return CATEGORY_ALIASES.get(key, key)


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    present = [c for c in columns if c in frame.columns]
    return [
        {column: _clean(row[column]) for column in present}
        for _, row in frame[present].iterrows()
    ]


# --- Tool 1 -----------------------------------------------------------------


def recommend_options(
    rank: int,
    category: str,
    gender: str,
    local_area: str,
    branch: str | list[str] | None = None,
    district: str | list[str] | None = None,
    college_type: str | list[str] | None = None,
    limit: int = 15,
) -> dict:
    """Safe / Moderate / Reach options for one student. Calls the Stage 2 engine."""

    def as_list(value):
        if value is None:
            return None
        return [value] if isinstance(value, str) else list(value)

    try:
        result = recommend(
            rank=int(rank),
            category=normalise_category(category),
            gender=str(gender).upper(),
            local_area=str(local_area).upper(),
            branch_codes=as_list(branch),
            districts=as_list(district),
            college_types=as_list(college_type),
            limit=int(limit),
            cutoffs=cutoffs(),
        )
    except ValueError as error:
        return {"error": str(error), "options": []}

    if result.empty:
        return {
            "options": [],
            "note": (
                "No options matched. Either the filters are too narrow, or no "
                "candidate of that category was admitted to anything within reach."
            ),
            "data_year": config.RECOMMEND_YEAR,
        }

    counts = result["band"].value_counts().to_dict()
    return {
        "data_year": int(result["data_year"].iloc[0]),
        "counselling_phase": str(result["counselling_phase"].iloc[0]),
        "student": {
            "rank": int(rank),
            "category": str(category).upper(),
            "gender": str(gender).upper(),
            "local_area": str(local_area).upper(),
        },
        "counts": {band: int(counts.get(band, 0)) for band in ("Safe", "Moderate", "Reach")},
        "band_untested_for_category": bool(result["band_untested_for_category"].iloc[0]),
        "warning": (
            banding.SC_WARNING if result["band_untested_for_category"].iloc[0] else None
        ),
        "options": _records(
            result,
            [
                "band",
                "college_code",
                "college_name",
                "branch_code",
                "branch_name",
                "name_status",
                "district",
                "college_type",
                "closing_rank",
                "ratio",
                "data_year",
            ],
        ),
    }


# --- Tool 2 -----------------------------------------------------------------


def get_option_details(college_code: str, branch_code: str) -> dict:
    """Closing ranks for one college-branch across every year we hold."""
    frame = cutoffs()
    rows = frame[
        (frame["college_code"].str.upper() == str(college_code).upper())
        & (frame["branch_code"].str.upper() == str(branch_code).upper())
    ]
    if rows.empty:
        return {
            "found": False,
            "note": f"No record of {college_code} / {branch_code} in any year.",
        }

    newest = rows.sort_values("year").iloc[-1]
    by_year = (
        rows[rows["closing_rank"].notna()]
        .sort_values(["year", "category", "gender"])
    )

    return {
        "found": True,
        "college_code": _clean(newest["college_code"]),
        "college_name": _clean(newest["college_name"]),
        "branch_code": _clean(newest["branch_code"]),
        "branch_name": _clean(newest["branch_name"]),
        "branch_name_status": _clean(newest["name_status"]),
        "district": _clean(newest["district"]),
        "college_type": _clean(newest["college_type"]),
        "years_present": sorted({int(y) for y in rows["year"].dropna().unique()}),
        "counselling_phase": _clean(newest["counselling_phase"]),
        "source_url": _clean(newest["source_url"]),
        "closing_ranks": _records(
            by_year, ["year", "category", "gender", "local_area", "closing_rank"]
        ),
    }


# --- Tool 3 -----------------------------------------------------------------


def compare_options(
    options: list[dict],
    category: str | None = None,
    gender: str | None = None,
) -> dict:
    """Side-by-side closing ranks for several college-branch pairs."""
    if not options:
        return {"error": "Give at least one college_code and branch_code.", "rows": []}

    frame = cutoffs()
    rows: list[dict] = []
    missing: list[str] = []

    for option in options:
        code = str(option.get("college_code", "")).upper()
        branch = str(option.get("branch_code", "")).upper()
        whole = frame[
            (frame["college_code"].str.upper() == code)
            & (frame["branch_code"].str.upper() == branch)
        ]
        subset = whole
        if category:
            subset = subset[subset["category"] == normalise_category(category)]
        if gender:
            subset = subset[subset["gender"] == str(gender).upper()]
        subset = subset[subset["closing_rank"].notna()]

        if subset.empty:
            # Say *why* it is absent. "Not found" for a women's college that a
            # boy asked about would be misleading: it exists, it is just not
            # open to him.
            if whole.empty:
                reason = "no record of this college and branch in any year"
            elif gender and whole[whole["closing_rank"].notna()].empty:
                reason = "the college and branch exist but hold no closing ranks"
            else:
                bits = []
                if category:
                    bits.append(f"category {normalise_category(category)}")
                if gender:
                    bits.append(f"{str(gender).lower()}")
                reason = (
                    f"exists, but no closing rank for {' and '.join(bits)} "
                    f"(a women's college has no boys' rows, for example)"
                )
            missing.append({"option": f"{code}/{branch}", "reason": reason})
            continue

        newest = subset.sort_values("year").iloc[-1]
        by_year = {
            int(year): int(group["closing_rank"].max())
            for year, group in subset.groupby("year")
        }
        rows.append(
            {
                "college_code": _clean(newest["college_code"]),
                "college_name": _clean(newest["college_name"]),
                "branch_code": _clean(newest["branch_code"]),
                "branch_name": _clean(newest["branch_name"]),
                "district": _clean(newest["district"]),
                "college_type": _clean(newest["college_type"]),
                "closing_rank_by_year": by_year,
            }
        )

    return {
        "rows": rows,
        "not_found": missing,
        "fields_we_do_not_have": sorted(OUT_OF_SCOPE.values()),
        "note": (
            "Comparison covers closing ranks only. We hold no fee, placement or "
            "quality data, so those cannot be compared."
        ),
    }


# --- Tool 4 -----------------------------------------------------------------


def _measured_accuracy() -> dict:
    """Pull the hold-out numbers from the backtest's own output file."""
    if not config.BACKTEST_RESULTS.exists():
        return {
            "holdout_accuracy": None,
            "holdout_note": (
                "The backtest has not been run in this checkout, so no accuracy "
                "figure is available. Run: python -m copilot.engine.backtest"
            ),
        }
    measured = json.loads(config.BACKTEST_RESULTS.read_text(encoding="utf-8"))
    return {
        "holdout_accuracy": measured["holdout"]["accuracy"],
        "holdout_shown": measured["holdout"]["times_shown"],
        "holdout_note": measured["holdout"]["note"],
    }


def explain_bands() -> dict:
    """What the three labels mean, and how often they were right."""
    thresholds = Thresholds.load()
    return {
        "method": (
            "ratio = your rank divided by last year's closing rank for that exact "
            "college, branch, category and gender. A smaller ratio is better, "
            "because a smaller rank number is a better rank."
        ),
        "bands": [
            {"band": "Safe", "ratio_up_to": thresholds.t_safe},
            {"band": "Moderate", "ratio_up_to": thresholds.t_moderate},
            {"band": "Reach", "ratio_up_to": thresholds.t_max},
        ],
        "girls_rule": (
            "The official statement says girls may also take boys' seats, so a "
            "girl's line is the more lenient of the two."
        ),
        "tuned_on": thresholds.tuned_on,
        "tested_on": thresholds.tested_on,
        # Read from the file the backtest writes, never typed in by hand. If the
        # backtest has not been run, we say so rather than quoting a number.
        **_measured_accuracy(),
        "sc_warning": banding.SC_WARNING,
        "source_disclaimer": (
            "The official statement says its ranks 'shall in no way reflect the "
            "rank upto which seat can be allotted in the present academic year'. "
            "These are last year's results, not a promise about this year."
        ),
    }


# --- What the model is told it can call -------------------------------------

TOOL_FUNCTIONS = {
    "recommend_options": recommend_options,
    "get_option_details": get_option_details,
    "compare_options": compare_options,
    "explain_bands": explain_bands,
}

TOOL_DECLARATIONS: list[dict] = [
    {
        "name": "recommend_options",
        "description": (
            "Get college+branch options for a student, grouped Safe / Moderate / "
            "Reach. Call this whenever the user gives a rank. Always needs rank, "
            "category, gender and local_area."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "rank": {"type": "integer", "description": "EAPCET rank, 1 or greater"},
                "category": {
                    "type": "string",
                    "description": "OC, BC-A, BC-B, BC-C, BC-D, BC-E, SC-I, SC-II, SC-III, ST or OC-EWS",
                },
                "gender": {"type": "string", "enum": ["BOYS", "GIRLS"]},
                "local_area": {
                    "type": "string",
                    "enum": ["AU", "SVU"],
                    "description": "Andhra University or Sri Venkateswara University region",
                },
                "branch": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Branch codes such as CSE, ECE, EEE, MEC, CIV, INF",
                },
                "district": {"type": "array", "items": {"type": "string"}},
                "college_type": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "PVT, UNIV, SF, PU or SS",
                },
                "limit": {"type": "integer", "description": "How many to return (default 15)"},
            },
            "required": ["rank", "category", "gender", "local_area"],
        },
    },
    {
        "name": "get_option_details",
        "description": (
            "Closing ranks for one college and branch across every year held. Use "
            "when the user asks about a specific college."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "college_code": {"type": "string"},
                "branch_code": {"type": "string"},
            },
            "required": ["college_code", "branch_code"],
        },
    },
    {
        "name": "compare_options",
        "description": "Compare two or more college+branch pairs side by side.",
        "parameters": {
            "type": "object",
            "properties": {
                "options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "college_code": {"type": "string"},
                            "branch_code": {"type": "string"},
                        },
                        "required": ["college_code", "branch_code"],
                    },
                },
                "category": {"type": "string"},
                "gender": {"type": "string", "enum": ["BOYS", "GIRLS"]},
            },
            "required": ["options"],
        },
    },
    {
        "name": "explain_bands",
        "description": (
            "What Safe / Moderate / Reach mean, the exact limits, and how often "
            "each was right in testing. Call this when the user asks what the "
            "labels mean or how reliable they are."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
]


def call_tool(name: str, arguments: dict) -> dict:
    """Run one tool by name. Errors come back as data, never as an exception."""
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return {"error": f"No such tool: {name}"}
    try:
        return function(**(arguments or {}))
    except TypeError as error:
        return {"error": f"Wrong arguments for {name}: {error}"}
    except Exception as error:  # noqa: BLE001 - the model must see the failure
        return {"error": f"{name} failed: {type(error).__name__}: {error}"}


def as_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)

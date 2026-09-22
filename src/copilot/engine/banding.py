"""The Safe / Moderate / Reach calculator. Plain Python, no model, no LLM.

How it works, in one line: compare the student's rank to last year's closing
rank for that exact college-branch-category-gender, and bucket the ratio.

    ratio = student_rank / last_year_closing_rank

A smaller ratio is better, because a smaller rank number is a better rank.

    ratio <= t_safe        -> Safe       comfortably inside last year's line
    t_safe < ratio <= t_mod -> Moderate  near the line
    t_mod  < ratio <= t_max -> Reach     past the line but not hopeless
    ratio  > t_max          -> not shown

The three limits are not guesses. They are tuned on one pair of years and then
tested once on a later pair that was never used for tuning. See
`copilot.engine.backtest`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from copilot import config

Band = Literal["Safe", "Moderate", "Reach"]

#: Categories that the 2024 -> 2025 hold-out test could not check, because the
#: state split SC into SC-I / SC-II / SC-III in 2025. Any answer about these
#: must carry the warning in `SC_WARNING`.
UNVALIDATED_CATEGORIES: frozenset[str] = frozenset(
    {"SC", "SC-I", "SC-II", "SC-III"}
)

SC_WARNING = (
    "The Safe / Moderate / Reach grouping could not be tested for SC categories. "
    "In 2025 the state split SC into SC-I, SC-II and SC-III, so last year's SC "
    "results cannot be compared like-for-like. Treat these groupings as less "
    "reliable than the ones shown for other categories."
)


@dataclass(frozen=True)
class Thresholds:
    """The three ratio limits, plus a record of where they came from."""

    t_safe: float
    t_moderate: float
    t_max: float
    tuned_on: str = ""
    tested_on: str = ""

    def __post_init__(self) -> None:
        if not 0 < self.t_safe < self.t_moderate < self.t_max:
            raise ValueError(
                f"thresholds must satisfy 0 < t_safe < t_moderate < t_max, got {self}"
            )

    @classmethod
    def load(cls, path=None) -> "Thresholds":
        data = json.loads((path or config.THRESHOLDS).read_text(encoding="utf-8"))
        return cls(
            t_safe=float(data["t_safe"]),
            t_moderate=float(data["t_moderate"]),
            t_max=float(data["t_max"]),
            tuned_on=data.get("tuned_on", ""),
            tested_on=data.get("tested_on", ""),
        )


def band_for(ratio: float, thresholds: Thresholds) -> Band | None:
    """Bucket one ratio. None means 'do not show this option'."""
    if ratio <= thresholds.t_safe:
        return "Safe"
    if ratio <= thresholds.t_moderate:
        return "Moderate"
    if ratio <= thresholds.t_max:
        return "Reach"
    return None


def effective_closing_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the official rule that girls may also take boys' seats.

    The 2024 statement says so in its own footnote: *"Girls are also eligible
    for Boys seats."* So the line a girl has to beat is the more lenient of the
    two, and a larger rank number is more lenient.

    Returns the same long shape with an added `effective_closing_rank`.
    """
    keys = ["year", "college_code", "branch_code", "category", "local_area"]

    wide = frame.pivot_table(
        index=keys,
        columns="gender",
        values="closing_rank",
        aggfunc="first",
    )
    for gender in ("BOYS", "GIRLS"):
        if gender not in wide.columns:
            wide[gender] = pd.NA

    boys = pd.to_numeric(wide["BOYS"], errors="coerce")
    girls = pd.to_numeric(wide["GIRLS"], errors="coerce")

    effective = pd.DataFrame(
        {
            # A boy can only use a boys' seat.
            "BOYS": boys,
            # A girl can use either, so take the more lenient (larger) number.
            # If only one of the two exists, that one applies.
            "GIRLS": pd.concat([boys, girls], axis=1).max(axis=1, skipna=True),
        },
        index=wide.index,
    )

    out = (
        effective.stack(future_stack=True)
        .rename("effective_closing_rank")
        .reset_index()
        .rename(columns={"level_5": "gender"})
    )
    out["effective_closing_rank"] = pd.array(
        out["effective_closing_rank"], dtype="Float64"
    )
    return out.dropna(subset=["effective_closing_rank"])


def load_cutoffs() -> pd.DataFrame:
    return pd.read_parquet(config.CUTOFFS_PARQUET)


def recommend(
    rank: int,
    category: str,
    gender: str,
    local_area: str,
    *,
    branch_codes: list[str] | None = None,
    districts: list[str] | None = None,
    college_types: list[str] | None = None,
    year: int | None = None,
    thresholds: Thresholds | None = None,
    limit: int | None = None,
    cutoffs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return college-branch options labelled Safe / Moderate / Reach.

    Deterministic. Same inputs always give the same output. No model involved.
    """
    if rank < 1:
        raise ValueError("rank must be 1 or greater")

    thresholds = thresholds or Thresholds.load()
    year = year or config.RECOMMEND_YEAR
    frame = load_cutoffs() if cutoffs is None else cutoffs

    year_frame = frame[frame["year"] == year]
    if year_frame.empty:
        raise ValueError(f"no data for year {year}")

    effective = effective_closing_ranks(year_frame)
    options = effective[
        (effective["category"] == category)
        & (effective["gender"] == gender)
        & (effective["local_area"] == local_area)
    ].copy()

    if options.empty:
        return _empty_result()

    # Attach the descriptive columns, which do not vary within a college-branch.
    details = (
        year_frame[
            [
                "college_code",
                "college_name",
                "branch_code",
                "branch_name",
                "name_status",
                "district",
                "college_type",
                "counselling_phase",
                "source_url",
            ]
        ]
        .drop_duplicates(subset=["college_code", "branch_code"])
    )
    options = options.merge(details, on=["college_code", "branch_code"], how="left")

    if branch_codes:
        options = options[options["branch_code"].isin(branch_codes)]
    if districts:
        options = options[options["district"].isin(districts)]
    if college_types:
        options = options[options["college_type"].isin(college_types)]
    if options.empty:
        return _empty_result()

    options["ratio"] = rank / options["effective_closing_rank"].astype(float)
    options["band"] = options["ratio"].map(lambda r: band_for(r, thresholds))
    options = options[options["band"].notna()]
    if options.empty:
        return _empty_result()

    options["data_year"] = year
    options["student_rank"] = rank
    options["closing_rank"] = options["effective_closing_rank"].astype("Int64")
    options["band_untested_for_category"] = category in UNVALIDATED_CATEGORIES

    # Safe first, then within each band the most competitive option first.
    # "Most competitive" = lowest closing rank, i.e. the hardest college to get
    # into that is still in this band. That is what a student actually wants:
    # the best place their rank can reach. Sorting by ratio instead would put
    # the easiest, least sought-after colleges at the top.
    #
    # Note this uses last year's closing rank as a stand-in for how sought-after
    # a college is. It is a revealed preference from the data, not a quality
    # ranking, and the app must never present it as one.
    order = {"Safe": 0, "Moderate": 1, "Reach": 2}
    options = options.sort_values(
        by=["band", "closing_rank"],
        key=lambda s: s.map(order) if s.name == "band" else s,
    )

    columns = [
        "band",
        "college_code",
        "college_name",
        "branch_code",
        "branch_name",
        "name_status",
        "district",
        "college_type",
        "category",
        "gender",
        "local_area",
        "student_rank",
        "closing_rank",
        "ratio",
        "data_year",
        "counselling_phase",
        "band_untested_for_category",
        "source_url",
    ]
    result = options[columns].reset_index(drop=True)
    return result.head(limit) if limit else result


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "band",
            "college_code",
            "college_name",
            "branch_code",
            "branch_name",
            "name_status",
            "district",
            "college_type",
            "category",
            "gender",
            "local_area",
            "student_rank",
            "closing_rank",
            "ratio",
            "data_year",
            "counselling_phase",
            "band_untested_for_category",
            "source_url",
        ]
    )


def explain(row: pd.Series, thresholds: Thresholds | None = None) -> str:
    """Plain-English reason for one option's band. Used by the explain tool."""
    thresholds = thresholds or Thresholds.load()
    limit = {
        "Safe": thresholds.t_safe,
        "Moderate": thresholds.t_moderate,
        "Reach": thresholds.t_max,
    }[row["band"]]
    return (
        f"Your rank {int(row['student_rank']):,} divided by the {int(row['data_year'])} "
        f"closing rank {int(row['closing_rank']):,} for {row['college_code']} "
        f"{row['branch_code']} ({row['category']}, {row['gender'].lower()}) is "
        f"{row['ratio']:.2f}, which falls in the {row['band']} band "
        f"(up to {limit:.2f})."
    )

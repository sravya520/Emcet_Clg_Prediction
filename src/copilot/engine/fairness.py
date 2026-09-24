"""Does the system work as well for every group, or only on average?

An overall accuracy figure can hide a group the system serves badly. This
re-runs the same 2024 -> 2025 hold-out used in the headline backtest, but
broken down by category, by gender, and by both together, and reports the
sample size beside every number so a small-sample result is never mistaken
for a reliable one.

It also measures how much data each group has to begin with. A category with
few rows and many blanks cannot produce good recommendations no matter how
good the maths is, and the honest response is to say so in the app rather than
to present a thin answer with the same confidence as a thick one.

Run:  python -m copilot.engine.fairness
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from copilot import config
from copilot.engine.backtest import RANK_STEP, build_fold, hit_rate, simulate
from copilot.engine.banding import Thresholds, effective_closing_ranks, recommend

#: Below this many student-option pairs, a percentage is too noisy to act on.
MIN_PAIRS_FOR_A_RELIABLE_RATE = 20_000

#: Above this share of blank cutoffs, a category is too thin for the app to
#: present its answers with the usual confidence.
#:
#: Note row COUNT is useless here: the table is dense by construction, so every
#: category has exactly the same number of rows. What differs is how many of
#: those rows carry an actual rank. An earlier version of this check compared
#: row counts and unsurprisingly found every category identical.
MAX_MISSING_PCT_FOR_A_RELIABLE_CATEGORY = 40.0

#: How far a group may fall below the overall figure before we call it out.
#: Five points is roughly the noise floor at our sample sizes.
NOTABLE_GAP_POINTS = 5.0

CATEGORY_ORDER = [
    "OC", "OC-EWS", "BC-A", "BC-B", "BC-C", "BC-D", "BC-E",
    "SC", "SC-I", "SC-II", "SC-III", "ST",
]


def _band_rates(ratio, admitted, thresholds: Thresholds) -> dict:
    bounds = [
        ("Safe", 0.0, thresholds.t_safe),
        ("Moderate", thresholds.t_safe, thresholds.t_moderate),
        ("Reach", thresholds.t_moderate, thresholds.t_max),
    ]
    out = {}
    for band, low, high in bounds:
        rate, count = hit_rate(ratio, admitted, low, high)
        out[band] = {
            "accuracy_pct": None if count == 0 else round(rate * 100, 1),
            "pairs": count,
        }
    return out


def accuracy_by_group(
    cutoffs: pd.DataFrame, thresholds: Thresholds, max_rank: int
) -> tuple[dict, dict, dict, dict]:
    """Hold-out accuracy overall, by category, by gender, and by both."""
    matched, _ = build_fold(cutoffs, 2024, 2025)

    ratio, admitted = simulate(matched, max_rank)
    overall = _band_rates(ratio, admitted, thresholds)

    # simulate() returns one row per (student rank x option), flattened. Rebuild
    # the option index so each pair can be attributed to its group.
    n_ranks = len(np.arange(RANK_STEP, max_rank + 1, RANK_STEP))
    option_index = np.tile(np.arange(len(matched)), n_ranks)
    categories = matched["category"].to_numpy()[option_index]
    genders = matched["gender"].to_numpy()[option_index]

    by_category = {}
    for category in sorted(set(matched["category"])):
        mask = categories == category
        by_category[category] = _band_rates(ratio[mask], admitted[mask], thresholds)

    by_gender = {}
    for gender in sorted(set(matched["gender"])):
        mask = genders == gender
        by_gender[gender] = _band_rates(ratio[mask], admitted[mask], thresholds)

    by_both = {}
    for category in sorted(set(matched["category"])):
        for gender in sorted(set(matched["gender"])):
            mask = (categories == category) & (genders == gender)
            if mask.sum():
                by_both[f"{category} / {gender}"] = _band_rates(
                    ratio[mask], admitted[mask], thresholds
                )

    return overall, by_category, by_gender, by_both


def coverage_by_group(cutoffs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How much data each group actually has in the recommending year."""
    year = cutoffs[cutoffs["year"] == config.RECOMMEND_YEAR]

    def summarise(frame: pd.DataFrame, key: str) -> pd.DataFrame:
        grouped = frame.groupby(key).agg(
            rows=("closing_rank", "size"),
            ranks_present=("closing_rank", "count"),
            colleges=("college_code", "nunique"),
            branches=("branch_code", "nunique"),
        )
        grouped["missing_pct"] = (
            100 * (1 - grouped["ranks_present"] / grouped["rows"])
        ).round(1)
        return grouped.reset_index()

    return summarise(year, "category"), summarise(year, "gender")


def options_for_a_typical_student(cutoffs: pd.DataFrame) -> pd.DataFrame:
    """How many options each group gets at a spread of ranks.

    A percentage accuracy is cold comfort if the group is offered three
    options and everyone else is offered ninety.
    """
    rows = []
    ranks = [20_000, 60_000, 100_000, 150_000]
    year = cutoffs[cutoffs["year"] == config.RECOMMEND_YEAR]
    categories = sorted(set(year["category"].dropna()))

    for category in categories:
        for gender in ("BOYS", "GIRLS"):
            counts = {}
            for rank in ranks:
                result = recommend(
                    rank, category, gender, "AU", cutoffs=cutoffs
                )
                counts[rank] = len(result)
            rows.append({"category": category, "gender": gender, **counts})
    return pd.DataFrame(rows)


def high_rank_tail(cutoffs: pd.DataFrame) -> pd.DataFrame:
    """Very high ranks must never produce a blank screen with no explanation."""
    rows = []
    year = cutoffs[cutoffs["year"] == config.RECOMMEND_YEAR]
    for category in sorted(set(year["category"].dropna())):
        for rank in (120_000, 150_000, 180_000):
            result = recommend(rank, category, "BOYS", "AU", cutoffs=cutoffs)
            worst = None
            if not result.empty:
                worst = int(result["closing_rank"].max())
            # What is the most lenient cutoff that exists for this group at all?
            effective = effective_closing_ranks(year)
            available = effective[
                (effective["category"] == category)
                & (effective["gender"] == "BOYS")
                & (effective["local_area"] == "AU")
            ]["effective_closing_rank"]
            rows.append(
                {
                    "category": category,
                    "rank": rank,
                    "options": len(result),
                    "most_lenient_cutoff_in_group": (
                        int(available.max()) if len(available) else None
                    ),
                    "explained": len(result) > 0 or len(available) > 0,
                }
            )
    return pd.DataFrame(rows)


def thin_groups(coverage: pd.DataFrame) -> list[str]:
    """Categories the app should warn about before showing results.

    Thin means most of the group's cells are blank, which happens when few
    candidates of that category were admitted anywhere. The maths still works;
    there is simply less evidence behind each answer, and the student deserves
    to know that.
    """
    return sorted(
        coverage.loc[
            coverage["missing_pct"] > MAX_MISSING_PCT_FOR_A_RELIABLE_CATEGORY,
            "category",
        ].tolist()
    )


def main() -> None:
    cutoffs = pd.read_parquet(config.CUTOFFS_PARQUET)
    thresholds = Thresholds.load()
    max_rank = int(pd.to_numeric(cutoffs["closing_rank"], errors="coerce").max())

    print("Re-running the 2024 -> 2025 hold-out, split by group...")
    overall, by_category, by_gender, by_both = accuracy_by_group(
        cutoffs, thresholds, max_rank
    )

    print("Measuring data coverage per group...")
    cat_cov, gen_cov = coverage_by_group(cutoffs)

    print("Counting options a typical student gets...")
    options = options_for_a_typical_student(cutoffs)

    print("Testing the high-rank tail...")
    tail = high_rank_tail(cutoffs)

    payload = {
        "overall": overall,
        "by_category": by_category,
        "by_gender": by_gender,
        "by_category_and_gender": by_both,
        "coverage_by_category": cat_cov.to_dict("records"),
        "coverage_by_gender": gen_cov.to_dict("records"),
        "options_for_a_typical_student": options.to_dict("records"),
        "high_rank_tail": tail.to_dict("records"),
        "thin_categories": thin_groups(cat_cov),
        "thresholds": {
            "t_safe": thresholds.t_safe,
            "t_moderate": thresholds.t_moderate,
            "t_max": thresholds.t_max,
        },
    }
    out = config.MAPPINGS_DIR / "fairness_results.json"
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"\nWrote {out}")

    print("\nOverall:", {b: v["accuracy_pct"] for b, v in overall.items()})
    print("\nBy category (Safe / Moderate / Reach, with pairs):")
    for category in CATEGORY_ORDER:
        if category not in by_category:
            continue
        v = by_category[category]
        print(
            f"  {category:<8} "
            f"{str(v['Safe']['accuracy_pct']):>6} / "
            f"{str(v['Moderate']['accuracy_pct']):>6} / "
            f"{str(v['Reach']['accuracy_pct']):>6}   "
            f"pairs {v['Safe']['pairs']:>9,} / {v['Moderate']['pairs']:>8,} / {v['Reach']['pairs']:>8,}"
        )
    print("\nBy gender:")
    for gender, v in by_gender.items():
        print(
            f"  {gender:<8} "
            f"{v['Safe']['accuracy_pct']} / {v['Moderate']['accuracy_pct']} / "
            f"{v['Reach']['accuracy_pct']}"
        )
    print("\nThin categories (app should warn):", thin_groups(cat_cov) or "none")


if __name__ == "__main__":
    main()

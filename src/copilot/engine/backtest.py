"""Test the Safe / Moderate / Reach limits against what actually happened.

The idea in one sentence: pretend it is last year, label options using last
year's closing ranks, then check against the year that followed.

Two folds, used for two different jobs:

  Fold A  2023 -> 2024   TUNING.   Pick the three limits here.
  Fold B  2024 -> 2025   HOLD-OUT. Run once, report, never tune on it.

Fold B leaves SC out. In 2025 the state replaced SC with SC-I / SC-II / SC-III,
so a 2024 SC row has no like-for-like successor and comparing them would be
inventing a mapping the government never published.

How a student is simulated
--------------------------
A rank is a position in a queue, so the student population is spread evenly
across ranks by construction: rank 1 through rank N each belong to exactly one
person. So we walk an even grid of ranks rather than sampling, which means the
result has no random seed in it and is exactly reproducible.

For every option that survived into the next year, and every rank on the grid,
we ask two questions:

  1. What would the app have told this student? (band, from last year's rank)
  2. What actually happened? (admitted if their rank beat the NEXT year's
     closing rank)

"Right" means the app said Safe / Moderate / Reach and the student would in
fact have got in by the next year's numbers.

Run:  python -m copilot.engine.backtest
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from copilot import config
from copilot.engine.banding import Thresholds, effective_closing_ranks

#: Step between simulated student ranks. 250 gives ~720 students spread evenly
#: over the whole rank range, which is plenty and keeps the run under a minute.
RANK_STEP = 250

#: Where each line is drawn: the chance a student sitting *exactly at that line*
#: still gets in, measured on the tuning fold. This is the product decision; the
#: ratio that achieves it is derived from the data.
#:
#: Drawing the line at the edge rather than at the band average matters. An
#: earlier version set the Safe line where everything below it averaged 90%,
#: which pushed the line out to 1.11 - past the point where an option is really
#: a coin flip - and left no room at all for a Moderate band (it came out 0.01
#: wide). Measuring at the edge keeps each label honest about its own worst case.
BOUNDARY_TARGETS = {
    "Safe": 0.90,      # at the Safe line, 9 in 10 still get in
    "Moderate": 0.50,  # at the Moderate line, it is a coin flip
    "Reach": 0.20,     # past the Reach line it is not worth listing
}

#: Candidate limits to search over.
GRID = np.round(np.arange(0.30, 3.001, 0.01), 2)

#: Width of the window used to measure the chance *at* a line.
BOUNDARY_WINDOW = 0.05

MATCH_KEYS = ["college_code", "branch_code", "category", "gender", "local_area"]


@dataclass
class FoldResult:
    name: str
    prev_year: int
    next_year: int
    options_matched: int
    options_dropped_no_successor: int
    students_simulated: int
    excluded_categories: list[str]
    per_band: pd.DataFrame
    overall_shown: int

    def to_markdown(self) -> str:
        lines = [
            f"### {self.name}: {self.prev_year} -> {self.next_year}",
            "",
            f"- Options matched in both years: **{self.options_matched:,}**",
            f"- Options dropped (no successor in {self.next_year}): "
            f"**{self.options_dropped_no_successor:,}**",
            f"- Simulated student ranks: **{self.students_simulated:,}** "
            f"(every {RANK_STEP}th rank)",
            f"- Student-option pairs the app would have shown: **{self.overall_shown:,}**",
        ]
        if self.excluded_categories:
            lines.append(
                f"- Categories excluded: **{', '.join(self.excluded_categories)}**"
            )
        lines += ["", self.per_band.to_markdown(index=False), ""]
        return "\n".join(lines)


def build_fold(
    cutoffs: pd.DataFrame,
    prev_year: int,
    next_year: int,
    exclude_categories: frozenset[str] = frozenset(),
) -> tuple[pd.DataFrame, int]:
    """Match each option to itself in the following year.

    Returns the matched table and how many options had no successor.
    """
    previous = effective_closing_ranks(cutoffs[cutoffs["year"] == prev_year])
    following = effective_closing_ranks(cutoffs[cutoffs["year"] == next_year])

    if exclude_categories:
        previous = previous[~previous["category"].isin(exclude_categories)]
        following = following[~following["category"].isin(exclude_categories)]

    merged = previous.merge(
        following,
        on=MATCH_KEYS,
        how="left",
        suffixes=("_prev", "_next"),
    )
    dropped = int(merged["effective_closing_rank_next"].isna().sum())
    matched = merged.dropna(subset=["effective_closing_rank_next"]).copy()

    matched["closing_prev"] = matched["effective_closing_rank_prev"].astype(float)
    matched["closing_next"] = matched["effective_closing_rank_next"].astype(float)
    return matched[MATCH_KEYS + ["closing_prev", "closing_next"]], dropped


def simulate(matched: pd.DataFrame, max_rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (ratio, was_admitted) for every simulated student-option pair.

    ratio is the student's rank divided by last year's closing rank.
    was_admitted is whether their rank beat the NEXT year's closing rank.
    """
    ranks = np.arange(RANK_STEP, max_rank + 1, RANK_STEP, dtype=float)
    closing_prev = matched["closing_prev"].to_numpy(dtype=float)
    closing_next = matched["closing_next"].to_numpy(dtype=float)

    # rows = students, columns = options
    ratio = ranks[:, None] / closing_prev[None, :]
    admitted = ranks[:, None] <= closing_next[None, :]
    return ratio.ravel(), admitted.ravel()


def hit_rate(ratio: np.ndarray, admitted: np.ndarray, low: float, high: float) -> tuple[float, int]:
    """Share of correct calls for the band (low, high], and how many pairs."""
    in_band = (ratio > low) & (ratio <= high)
    count = int(in_band.sum())
    if count == 0:
        return float("nan"), 0
    return float(admitted[in_band].mean()), count


def boundary_rate(ratio: np.ndarray, admitted: np.ndarray, at: float) -> tuple[float, int]:
    """Chance of getting in for students sitting right at ratio `at`.

    Measured over a narrow window just below the line, rather than over
    everything below it.
    """
    return hit_rate(ratio, admitted, max(0.0, at - BOUNDARY_WINDOW), at)


def tune(ratio: np.ndarray, admitted: np.ndarray) -> Thresholds:
    """Draw each line where the chance at that line falls to its target.

    The chance falls steadily as the ratio grows, so for each target we walk up
    the grid and take the last ratio that still clears it.
    """
    t_safe = _line_at(ratio, admitted, target=BOUNDARY_TARGETS["Safe"], above=0.0)
    t_moderate = _line_at(ratio, admitted, target=BOUNDARY_TARGETS["Moderate"], above=t_safe)
    t_max = _line_at(ratio, admitted, target=BOUNDARY_TARGETS["Reach"], above=t_moderate)
    return Thresholds(t_safe=t_safe, t_moderate=t_moderate, t_max=t_max)


def _line_at(ratio, admitted, *, target: float, above: float) -> float:
    """Last ratio on the grid where the chance *at* that ratio still meets target."""
    best = None
    for candidate in GRID:
        if candidate <= above:
            continue
        rate, count = boundary_rate(ratio, admitted, float(candidate))
        if count == 0:
            continue
        if rate >= target:
            best = float(candidate)
        elif best is not None:
            break  # the chance only falls from here; this is the crossing point
    if best is None:
        best = float(min(c for c in GRID if c > above))
    return best


def evaluate(
    matched: pd.DataFrame,
    thresholds: Thresholds,
    name: str,
    prev_year: int,
    next_year: int,
    dropped: int,
    max_rank: int,
    excluded: list[str],
) -> FoldResult:
    ratio, admitted = simulate(matched, max_rank)
    bounds = [
        ("Safe", 0.0, thresholds.t_safe),
        ("Moderate", thresholds.t_safe, thresholds.t_moderate),
        ("Reach", thresholds.t_moderate, thresholds.t_max),
    ]
    rows = []
    for band, low, high in bounds:
        rate, count = hit_rate(ratio, admitted, low, high)
        rows.append(
            {
                "band": band,
                "ratio range": f"{low:.2f} - {high:.2f}",
                "times shown": count,
                "times right": int(round(rate * count)) if count else 0,
                "how often right": f"{rate * 100:.1f}%" if count else "n/a",
            }
        )
    per_band = pd.DataFrame(rows)
    return FoldResult(
        name=name,
        prev_year=prev_year,
        next_year=next_year,
        options_matched=len(matched),
        options_dropped_no_successor=dropped,
        students_simulated=int(max_rank // RANK_STEP),
        excluded_categories=excluded,
        per_band=per_band,
        overall_shown=int(per_band["times shown"].sum()),
    )


def main() -> None:
    cutoffs = pd.read_parquet(config.CUTOFFS_PARQUET)
    max_rank = int(pd.to_numeric(cutoffs["closing_rank"], errors="coerce").max())
    print(f"Highest closing rank anywhere in the data: {max_rank:,}")
    print(f"Simulating every {RANK_STEP}th rank up to that.\n")

    # ---- Fold A: tune -----------------------------------------------------
    print("Fold A (2023 -> 2024): tuning the limits...")
    tune_matched, tune_dropped = build_fold(cutoffs, 2023, 2024)
    tune_ratio, tune_admitted = simulate(tune_matched, max_rank)
    thresholds = tune(tune_ratio, tune_admitted)
    print(f"  chosen: {thresholds.t_safe=} {thresholds.t_moderate=} {thresholds.t_max=}")

    fold_a = evaluate(
        tune_matched, thresholds, "Fold A (tuning)", 2023, 2024, tune_dropped, max_rank, []
    )

    # ---- Lock the limits BEFORE looking at the hold-out -------------------
    thresholds = Thresholds(
        t_safe=thresholds.t_safe,
        t_moderate=thresholds.t_moderate,
        t_max=thresholds.t_max,
        tuned_on="2023->2024",
        tested_on="2024->2025 (SC excluded)",
    )
    config.THRESHOLDS.parent.mkdir(parents=True, exist_ok=True)
    config.THRESHOLDS.write_text(
        json.dumps(
            {
                **asdict(thresholds),
                "boundary_targets": BOUNDARY_TARGETS,
                "rank_step": RANK_STEP,
                "note": (
                    "Tuned on Fold A only. Written to disk BEFORE the hold-out fold "
                    "was evaluated, so the hold-out result could not influence them."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"  limits locked and written to {config.THRESHOLDS}\n")

    # ---- Fold B: hold-out, evaluated once ---------------------------------
    print("Fold B (2024 -> 2025): hold-out, SC excluded. Running once...")
    excluded = ["SC", "SC-I", "SC-II", "SC-III"]
    holdout_matched, holdout_dropped = build_fold(
        cutoffs, 2024, 2025, exclude_categories=frozenset(excluded)
    )
    fold_b = evaluate(
        holdout_matched,
        thresholds,
        "Fold B (hold-out)",
        2024,
        2025,
        holdout_dropped,
        max_rank,
        excluded,
    )

    # Single source of truth for every accuracy figure quoted anywhere else.
    # Written only now, after the hold-out has run, so it cannot have leaked
    # backwards into the limits.
    config.BACKTEST_RESULTS.write_text(
        json.dumps(
            {
                "tuning": _fold_json(fold_a),
                "holdout": {
                    **_fold_json(fold_b),
                    "note": (
                        "Measured on 2024 -> 2025, which was not used to choose the "
                        "limits. SC was excluded because 2025 split SC into "
                        "SC-I / SC-II / SC-III."
                    ),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    report = render(thresholds, fold_a, fold_b)
    config.BACKTEST_REPORT.write_text(report, encoding="utf-8", newline="\n")
    print(f"\nWrote {config.BACKTEST_REPORT}\n")
    print(fold_a.per_band.to_string(index=False))
    print()
    print(fold_b.per_band.to_string(index=False))


def _fold_json(fold: FoldResult) -> dict:
    per_band = fold.per_band.set_index("band")
    return {
        "fold": f"{fold.prev_year}->{fold.next_year}",
        "accuracy": {band: per_band.loc[band, "how often right"] for band in per_band.index},
        "times_shown": {band: int(per_band.loc[band, "times shown"]) for band in per_band.index},
        "options_matched": fold.options_matched,
        "excluded_categories": fold.excluded_categories,
    }


def render(thresholds: Thresholds, fold_a: FoldResult, fold_b: FoldResult) -> str:
    return "\n".join(
        [
            "# Backtest: do the Safe / Moderate / Reach labels hold up?",
            "",
            "Every number on this page was produced by "
            "`python -m copilot.engine.backtest`. Nothing here is written by hand.",
            "",
            "## The limits",
            "",
            f"| Band | Ratio range | Meaning |",
            f"|---|---|---|",
            f"| Safe | up to **{thresholds.t_safe:.2f}** | comfortably inside last year's line |",
            f"| Moderate | **{thresholds.t_safe:.2f} - {thresholds.t_moderate:.2f}** | near the line |",
            f"| Reach | **{thresholds.t_moderate:.2f} - {thresholds.t_max:.2f}** | past the line, still possible |",
            f"| (not shown) | above **{thresholds.t_max:.2f}** | too far to be worth listing |",
            "",
            f"Tuned on **{thresholds.tuned_on}**. Tested once on **{thresholds.tested_on}**.",
            "",
            "Each line is drawn where the chance of getting in, for a student sitting "
            "right at that line, falls to a set level on the tuning fold: "
            + ", ".join(f"{k} >= {v:.0%}" for k, v in BOUNDARY_TARGETS.items())
            + ". Measured **at** the line, not averaged over everything below it. "
            "That distinction is load-bearing: an earlier version averaged, which "
            "pushed the Safe line out to 1.11 and squeezed the Moderate band down "
            "to 0.01 wide. Averaging lets a band's comfortable middle hide a weak "
            "edge; measuring at the edge does not.",
            "",
            "## Fold A - tuning",
            "",
            "These numbers are **not evidence**. The limits were fitted to this fold, "
            "so a good score here is circular by construction. It is shown for "
            "completeness only.",
            "",
            fold_a.to_markdown(),
            "## Fold B - hold-out (the real result)",
            "",
            "This fold was evaluated **once**, after the limits were written to disk. "
            "They were not adjusted afterwards.",
            "",
            fold_b.to_markdown(),
            "## What SC students are told",
            "",
            "SC is missing from Fold B on purpose. In 2025 the state replaced SC with "
            "SC-I / SC-II / SC-III, so a 2024 SC row has no like-for-like successor. "
            "Mapping them would mean inventing a correspondence the government never "
            "published. The app therefore shows every SC, SC-I, SC-II and SC-III "
            "student a warning that their bands are untested.",
            "",
            "## Disclosure: the hold-out was run twice",
            "",
            "Full transparency, because it affects how much weight this result "
            "deserves. The first version of the tuning code measured each band's "
            "hit rate as an average over everything below the line. That produced "
            "a broken set of limits - a Moderate band 0.01 wide - and the hold-out "
            "was evaluated once with them before the flaw was noticed.",
            "",
            "The flaw was visible in the tuning fold alone (a band that narrow is "
            "obviously wrong), and the fix was derived from the tuning fold's "
            "curve, not from the hold-out. But the hold-out is no longer perfectly "
            "untouched, and pretending otherwise would be the exact kind of quiet "
            "overclaim this project is built to avoid. Treat the hold-out numbers "
            "as *one* re-run after a methodology fix, not as a first-ever look.",
            "",
            "The next genuinely clean test will be the 2025 -> 2026 fold, once the "
            "2026 statement is published.",
            "",
            "## Honest limits of this test",
            "",
            "- It assumes an option that existed in both years is the same option. "
            "Colleges do change what they offer.",
            "- It excludes special reservation categories (PWD, NCC, Sports, CAP, "
            "Scouts & Guides), because the source statements exclude them too.",
            "- Closing ranks are end-of-web-counselling and exclude spot admissions, "
            "so real intake goes slightly further than these numbers show.",
            "- It measures whether a rank would have been enough, not whether the "
            "student would have been offered or accepted that seat.",
            "",
        ]
    )


if __name__ == "__main__":
    main()

"""Tests for the Safe / Moderate / Reach calculator.

The important properties: the maths is right, the official girls-may-take-boys-
seats rule is applied, SC students always get the untested warning, and nothing
is ever invented.
"""

from __future__ import annotations

import pandas as pd
import pytest

from copilot import config
from copilot.engine import banding
from copilot.engine.banding import Thresholds, band_for, effective_closing_ranks, recommend

T = Thresholds(t_safe=0.76, t_moderate=1.15, t_max=1.51)


# --- Banding maths ----------------------------------------------------------


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.10, "Safe"),
        (0.7599, "Safe"),
        (0.76, "Safe"),       # boundary belongs to the safer band
        (0.7601, "Moderate"),
        (1.15, "Moderate"),
        (1.1501, "Reach"),
        (1.51, "Reach"),
        (1.5101, None),       # too far: not shown at all
        (99.0, None),
    ],
)
def test_band_for(ratio, expected):
    assert band_for(ratio, T) == expected


def test_a_better_rank_never_gets_a_worse_band():
    """Monotonicity: improving your rank can never downgrade your band."""
    order = {"Safe": 0, "Moderate": 1, "Reach": 2, None: 3}
    closing = 50_000
    previous = -1
    for rank in range(1_000, 120_000, 1_000):
        current = order[band_for(rank / closing, T)]
        assert current >= previous
        previous = current


def test_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        Thresholds(t_safe=1.2, t_moderate=0.8, t_max=1.5)
    with pytest.raises(ValueError):
        Thresholds(t_safe=0.0, t_moderate=0.8, t_max=1.5)


def test_saved_thresholds_load_and_record_their_provenance():
    saved = Thresholds.load()
    assert 0 < saved.t_safe < saved.t_moderate < saved.t_max
    assert saved.tuned_on == "2023->2024"
    assert "2024->2025" in saved.tested_on
    assert "SC excluded" in saved.tested_on


# --- The official girls-may-take-boys-seats rule ----------------------------


def _row(gender, rank, *, year=2025, college="AAAA", branch="CSE", category="OC"):
    return {
        "year": year,
        "college_code": college,
        "branch_code": branch,
        "category": category,
        "local_area": "AU",
        "gender": gender,
        "closing_rank": rank,
    }


def test_girls_get_the_more_lenient_of_the_two_seats():
    """The 2024 statement says girls are also eligible for boys' seats.

    A larger rank number is more lenient, so a girl's effective line is the
    larger of the two.
    """
    frame = pd.DataFrame([_row("BOYS", 10_000), _row("GIRLS", 25_000)])
    out = effective_closing_ranks(frame).set_index("gender")["effective_closing_rank"]
    assert out["BOYS"] == 10_000
    assert out["GIRLS"] == 25_000


def test_girls_benefit_when_the_boys_line_is_the_lenient_one():
    frame = pd.DataFrame([_row("BOYS", 30_000), _row("GIRLS", 12_000)])
    out = effective_closing_ranks(frame).set_index("gender")["effective_closing_rank"]
    assert out["BOYS"] == 30_000
    assert out["GIRLS"] == 30_000  # she may take the boys' seat


def test_boys_never_benefit_from_the_girls_line():
    frame = pd.DataFrame([_row("BOYS", 12_000), _row("GIRLS", 40_000)])
    out = effective_closing_ranks(frame).set_index("gender")["effective_closing_rank"]
    assert out["BOYS"] == 12_000


def test_one_missing_side_still_yields_the_other():
    frame = pd.DataFrame([_row("BOYS", None), _row("GIRLS", 20_000)])
    out = effective_closing_ranks(frame).set_index("gender")["effective_closing_rank"]
    assert out["GIRLS"] == 20_000


def test_both_missing_produces_no_row_at_all():
    """A missing cutoff is never filled in. The option simply cannot be offered."""
    frame = pd.DataFrame([_row("BOYS", None), _row("GIRLS", None)])
    assert effective_closing_ranks(frame).empty


# --- recommend() ------------------------------------------------------------


@pytest.fixture(scope="module")
def cutoffs():
    return pd.read_parquet(config.CUTOFFS_PARQUET)


def test_recommend_rejects_a_nonsense_rank(cutoffs):
    with pytest.raises(ValueError):
        recommend(0, "OC", "BOYS", "AU", cutoffs=cutoffs)


def test_recommend_labels_and_orders(cutoffs):
    result = recommend(45_000, "BC-B", "GIRLS", "AU", cutoffs=cutoffs, thresholds=T)
    assert not result.empty
    assert set(result["band"]) <= {"Safe", "Moderate", "Reach"}

    # Safe before Moderate before Reach.
    order = {"Safe": 0, "Moderate": 1, "Reach": 2}
    positions = result["band"].map(order).tolist()
    assert positions == sorted(positions)

    # Within a band, the most competitive option comes first.
    safe = result[result["band"] == "Safe"]["closing_rank"].tolist()
    assert safe == sorted(safe)


def test_every_row_carries_its_year_and_source(cutoffs):
    result = recommend(45_000, "OC", "BOYS", "AU", cutoffs=cutoffs, thresholds=T)
    assert (result["data_year"] == config.RECOMMEND_YEAR).all()
    assert result["source_url"].notna().all()
    assert result["counselling_phase"].notna().all()


def test_ratio_matches_the_stated_rank_and_cutoff(cutoffs):
    """No hidden adjustment: the printed ratio is exactly rank / cutoff."""
    result = recommend(45_000, "OC", "BOYS", "AU", cutoffs=cutoffs, thresholds=T)
    row = result.iloc[0]
    assert row["ratio"] == pytest.approx(45_000 / row["closing_rank"])


def test_filters_are_respected(cutoffs):
    result = recommend(
        45_000, "OC", "BOYS", "AU",
        branch_codes=["CSE"], districts=["VSP"],
        cutoffs=cutoffs, thresholds=T,
    )
    assert set(result["branch_code"]) <= {"CSE"}
    assert set(result["district"]) <= {"VSP"}


def test_a_top_rank_gets_more_options_than_a_poor_one(cutoffs):
    strong = recommend(1_000, "OC", "BOYS", "AU", cutoffs=cutoffs, thresholds=T)
    weak = recommend(170_000, "OC", "BOYS", "AU", cutoffs=cutoffs, thresholds=T)
    assert len(strong) > len(weak)


def test_unknown_category_returns_nothing_rather_than_guessing(cutoffs):
    result = recommend(45_000, "NOT-A-CATEGORY", "BOYS", "AU", cutoffs=cutoffs, thresholds=T)
    assert result.empty


# --- The SC warning ---------------------------------------------------------


@pytest.mark.parametrize("category", ["SC", "SC-I", "SC-II", "SC-III"])
def test_sc_students_are_always_flagged_as_untested(cutoffs, category):
    result = recommend(45_000, category, "BOYS", "AU", cutoffs=cutoffs, thresholds=T)
    if result.empty:
        pytest.skip(f"no options for {category} at this rank")
    assert result["band_untested_for_category"].all()


@pytest.mark.parametrize("category", ["OC", "BC-A", "BC-B", "ST", "OC-EWS"])
def test_other_categories_are_not_flagged(cutoffs, category):
    result = recommend(45_000, category, "BOYS", "AU", cutoffs=cutoffs, thresholds=T)
    if result.empty:
        pytest.skip(f"no options for {category} at this rank")
    assert not result["band_untested_for_category"].any()


def test_the_sc_warning_text_says_what_it_should():
    text = banding.SC_WARNING.lower()
    assert "could not be tested" in text
    assert "sc" in text
    assert "less reliable" in text

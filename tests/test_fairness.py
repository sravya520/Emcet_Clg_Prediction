"""Tests for the fairness audit and the tone check.

These run offline. The live tone check (`python -m copilot.tone_check`) needs
an API key; what is tested here is the machinery that judges its output, so a
broken detector cannot quietly start reporting "clean" for everything.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from copilot import config
from copilot.agent import tools
from copilot.engine import fairness
from copilot.tone_check import DISCOURAGING, PATRONISING, analyse, phrases_found


# --- The tone detector must actually detect ---------------------------------


def test_discouraging_language_is_caught():
    """If this ever stops catching things, the tone check would report every
    answer as clean and we would learn nothing."""
    reply = "Unfortunately your options are very limited, so be realistic."
    found = phrases_found(reply, DISCOURAGING)
    assert "unfortunately" in found
    assert "very limited" in found
    assert "be realistic" in found


def test_patronising_language_is_caught():
    reply = "Don't worry, you can still find something. At least you qualified."
    found = phrases_found(reply, PATRONISING)
    assert "don't worry" in found
    assert "at least" in found


def test_a_neutral_answer_is_not_flagged():
    reply = (
        "Based on your rank of 45,000 as an SC-I boy in the AU region, here are "
        "your CSE options grouped Safe, Moderate and Reach using 2025 closing ranks."
    )
    assert phrases_found(reply, DISCOURAGING) == []
    assert phrases_found(reply, PATRONISING) == []


def test_analyse_finds_the_expected_elements():
    reply = (
        "Based on 2025 closing ranks, ADIT closed at 46,204 and is a Safe option."
    )
    result = analyse(reply)
    assert all(result["elements_present"].values()), result["elements_present"]


def test_analyse_notices_a_missing_data_year():
    result = analyse("ADIT closed at 46,204 and is Safe.")
    assert result["elements_present"]["states the data year"] is False


# --- The recorded live run --------------------------------------------------


@pytest.fixture(scope="module")
def recorded():
    path = config.REPO_ROOT / "evals" / "tone_check.json"
    if not path.exists():
        pytest.skip("tone check has not been run")
    return json.loads(path.read_text(encoding="utf-8"))


def test_no_group_got_discouraging_or_patronising_language(recorded):
    rows = [r for r in recorded["by_category"] + recorded["by_gender"] if "reply" in r]
    offenders = [
        (r["category"], r["gender"], r["discouraging_phrases"] + r["patronising_phrases"])
        for r in rows
        if r["discouraging_phrases"] or r["patronising_phrases"]
    ]
    assert not offenders, f"discouraging or patronising language found: {offenders}"


def test_every_group_got_the_same_kinds_of_information(recorded):
    rows = [r for r in recorded["by_category"] + recorded["by_gender"] if "reply" in r]
    for row in rows:
        missing = [k for k, v in row["elements_present"].items() if not v]
        assert not missing, f"{row['category']}/{row['gender']} missing {missing}"


def test_only_sc_answers_carry_the_sc_warning(recorded):
    rows = [r for r in recorded["by_category"] + recorded["by_gender"] if "reply" in r]
    for row in rows:
        expected = row["category"].upper().startswith("SC")
        assert row["sc_warning_shown"] == expected, (
            f"{row['category']} sc_warning_shown={row['sc_warning_shown']}"
        )


# --- Coverage and thin groups -----------------------------------------------


@pytest.fixture(scope="module")
def cutoffs():
    return pd.read_parquet(config.CUTOFFS_PARQUET)


def test_thin_categories_are_detected_by_missing_data_not_row_count(cutoffs):
    """Row count is useless: the table is dense, so every category has the
    same number of rows. An earlier version compared counts and found nothing."""
    by_category, _ = fairness.coverage_by_group(cutoffs)
    assert by_category["rows"].nunique() == 1, "rows are identical by construction"
    assert by_category["missing_pct"].nunique() > 1, "missing data is what differs"
    assert fairness.thin_groups(by_category) == ["BC-C", "OC-EWS"]


def test_the_app_warns_for_every_thin_category(cutoffs):
    by_category, _ = fairness.coverage_by_group(cutoffs)
    for category in fairness.thin_groups(by_category):
        result = tools.recommend_options(45_000, category, "BOYS", "AU", per_band=2)
        assert result["thin_data_for_category"] is True, category
        assert result["thin_data_warning"], category


def test_a_well_covered_category_is_not_warned_about():
    result = tools.recommend_options(45_000, "BC-B", "BOYS", "AU", per_band=2)
    assert result["thin_data_for_category"] is False
    assert result["thin_data_warning"] is None


def test_girls_never_get_fewer_options_than_boys(cutoffs):
    """The official rule says girls may also take boys' seats, so a girl's
    option list can never be shorter than the equivalent boy's."""
    from copilot.engine.banding import recommend

    for category in ("OC", "BC-B", "SC-I", "ST"):
        boys = recommend(45_000, category, "BOYS", "AU", cutoffs=cutoffs)
        girls = recommend(45_000, category, "GIRLS", "AU", cutoffs=cutoffs)
        assert len(girls) >= len(boys), category

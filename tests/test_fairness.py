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


# --- Per-category accuracy shown to the student -----------------------------


def test_accuracy_is_read_from_the_measured_file_not_hardcoded():
    """If the audit is re-run with different numbers, the app must follow."""
    import json

    measured = json.loads(
        (config.MAPPINGS_DIR / "fairness_results.json").read_text(encoding="utf-8")
    )["by_category"]
    for category, bands in measured.items():
        shown = tools.band_accuracy_for(category)
        for band, values in bands.items():
            assert shown[band] == values["accuracy_pct"], f"{category}/{band}"


def test_a_measured_category_reports_its_own_number_not_the_average():
    """BC-C is 93.9% on Safe. It must never be told the 97.8% overall figure."""
    assert tools.band_accuracy_for("BC-C")["Safe"] == 93.9
    assert tools.band_accuracy_for("OC")["Reach"] == 27.3


@pytest.mark.parametrize("category", ["SC", "SC-I", "SC-II", "SC-III"])
def test_sc_categories_report_that_accuracy_is_unmeasured(category):
    """Never borrow another group's accuracy for a group we could not measure."""
    assert tools.band_accuracy_for(category) == {
        "Safe": None, "Moderate": None, "Reach": None
    }


@pytest.mark.parametrize("category", ["SC-I", "BC-C", "OC"])
def test_recommend_carries_the_accuracy_for_that_category(category):
    result = tools.recommend_options(45_000, category, "BOYS", "AU", per_band=2)
    assert "band_accuracy" in result
    assert set(result["band_accuracy"]) == {"Safe", "Moderate", "Reach"}
    expected = tools.band_accuracy_for(category)
    assert result["band_accuracy"] == expected


def test_the_ui_caption_says_unmeasured_rather_than_a_number():
    from copilot.ui import band_caption

    unmeasured = band_caption("Safe", {"Safe": None})
    assert "could not be measured" in unmeasured
    assert "%" not in unmeasured

    measured = band_caption("Safe", {"Safe": 93.9})
    assert "93.9%" in measured


# --- Special-category quotas ------------------------------------------------


def test_the_special_quota_notice_names_every_excluded_quota():
    """These students are outside the data entirely, not merely underserved."""
    text = tools.SPECIAL_QUOTA_NOTICE.lower()
    for quota in ("pwd", "ncc", "sports", "cap", "scouts", "minority"):
        assert quota in text, f"{quota} not named in the notice"
    assert "does not cover" in text


def test_the_notice_travels_with_every_recommendation():
    result = tools.recommend_options(45_000, "OC", "BOYS", "AU", per_band=2)
    assert result["special_quota_notice"] == tools.SPECIAL_QUOTA_NOTICE


def test_the_agent_is_told_to_mention_both():
    from copilot.agent.loop import SYSTEM_PROMPT

    assert "band_accuracy" in SYSTEM_PROMPT
    assert "special_quota_notice" in SYSTEM_PROMPT
    assert "could not be measured" in SYSTEM_PROMPT


def test_the_ui_shows_the_quota_notice_at_warning_weight():
    """As prominent as the SC warning: st.warning, not a caption."""
    import inspect

    from copilot import ui

    source = inspect.getsource(ui.show_special_quota_notice)
    assert "st.warning" in source
    assert "Not covered" in source

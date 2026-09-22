"""Tests for parsing and normalisation.

The cases that matter are the ones that silently corrupt data if wrong:
missing-value spellings, header text that differs between years, and the
reshape from one-column-per-category to one-row-per-category.
"""

from __future__ import annotations

import pandas as pd
import pytest

from copilot import config
from copilot.data import ingest


# --- Header normalisation ---------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SNO", "SNO"),
        ("OC_EWS_G\nIRLS", "OC_EWS_GIRLS"),  # PDF wraps headers mid-word
        ("branch_\ncode", "BRANCH_CODE"),
        ("INST\n_RE\nG.", "INST_REG"),  # trailing dot differs between years
        ("INST_REG", "INST_REG"),
        ("NAME OF THE INSTITUTION", "NAMEOFTHEINSTITUTION"),
        ("Local_Ar\nea", "LOCAL_AREA"),
        (None, ""),
    ],
)
def test_normalise_header(raw, expected):
    assert ingest.normalise_header(raw) == expected


def test_header_aliases_cover_both_layouts():
    """The 2022 and 2023+ spellings must land on the same canonical field."""
    for old, new in [("inst_code", "INSTCODE"), ("inst_name", "NAME OF THE INSTITUTION")]:
        a = ingest.FIELD_ALIASES[ingest.normalise_header(old)]
        b = ingest.FIELD_ALIASES[ingest.normalise_header(new)]
        assert a == b


# --- Missing values ---------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("128861", 128861),
        (" 43000 ", 43000),
        ("1,28,861", 128861),
        ("", None),
        (None, None),
        ("NA", None),  # the 2022 statement spells missing as 'NA'
        ("na", None),
        ("N/A", None),
        ("-", None),
        ("0", None),  # rank 0 is not a real rank
        ("-5", None),
        ("abc", None),
    ],
)
def test_parse_rank(raw, expected):
    assert ingest.parse_rank(raw) == expected


def test_missing_rank_is_none_not_a_sentinel():
    """The old project filled blanks with 999999. That bug stays dead."""
    assert ingest.parse_rank("") is None
    assert ingest.parse_rank("NA") != 999999


# --- Category columns -------------------------------------------------------


def test_sc_subclassification_maps_distinctly():
    """SC and SC-I must not collapse into each other."""
    assert ingest.CATEGORY_COLUMNS["SC_BOYS"] == ("SC", "BOYS")
    assert ingest.CATEGORY_COLUMNS["SCI_BOYS"] == ("SC-I", "BOYS")
    assert ingest.CATEGORY_COLUMNS["SCII_GIRLS"] == ("SC-II", "GIRLS")
    assert ingest.CATEGORY_COLUMNS["SCIII_GIRLS"] == ("SC-III", "GIRLS")


def test_expected_column_counts():
    """18 category columns in 2022-24, 22 in 2025."""
    assert len(ingest.CATEGORY_COLUMNS) == 24  # 12 categories x 2 genders
    pre_2025 = [c for c in ingest.CATEGORY_COLUMNS if not c.startswith(("SCI", "SCII", "SCIII"))]
    assert len(pre_2025) == 18


# --- Column mapping errors --------------------------------------------------


def test_unknown_column_is_an_error():
    header = ["SNO", "INSTCODE", "BRANCH_CODE", "OC_BOYS", "SOMETHING_NEW"]
    with pytest.raises(ValueError, match="unrecognised column"):
        ingest.map_columns(header, "test")


def test_blank_header_with_data_is_an_error():
    """This is the bug that lost the 2023 fee column on the first run."""
    header = ["SNO", "INSTCODE", "BRANCH_CODE", "OC_BOYS", ""]
    with pytest.raises(ValueError, match="blank header but contains data"):
        ingest.map_columns(header, "test", populated=frozenset({4}))


def test_blank_header_without_data_is_dropped():
    header = ["SNO", "INSTCODE", "BRANCH_CODE", "OC_BOYS", ""]
    fields, categories = ingest.map_columns(header, "test", populated=frozenset())
    assert "college_code" in fields
    assert len(categories) == 1


def test_missing_required_column_is_an_error():
    header = ["SNO", "INSTCODE", "OC_BOYS"]
    with pytest.raises(ValueError, match="branch_code"):
        ingest.map_columns(header, "test")


# --- Reshape ----------------------------------------------------------------


SOURCE = {
    "id": "test",
    "academic_year": 2024,
    "counselling_phase": "end_of_web_counselling",
    "source_url": "https://example.gov.in/doc.pdf",
    "file": "doc.pdf",
}


def _row(*values):
    return ingest.RawRow(values=list(values), page=7)


def test_to_long_explodes_categories_into_rows():
    header = ["SNO", "INSTCODE", "INST_REG", "A_REG", "BRANCH_CODE", "OC_BOYS", "OC_GIRLS"]
    rows = [_row("1", "ACEE", "AU", "AU", "CSE", "128861", "")]

    frame = ingest.to_long(header, rows, SOURCE)

    assert len(frame) == 2  # one row per category x gender
    assert set(frame["category"]) == {"OC"}
    assert sorted(frame["gender"]) == ["BOYS", "GIRLS"]

    boys = frame[frame["gender"] == "BOYS"].iloc[0]
    girls = frame[frame["gender"] == "GIRLS"].iloc[0]
    assert boys["closing_rank"] == 128861
    assert pd.isna(girls["closing_rank"])
    assert boys["year"] == 2024
    assert boys["source_page"] == 7
    assert boys["source_url"] == "https://example.gov.in/doc.pdf"


def test_local_area_falls_back_to_college_region_and_is_flagged():
    """2022 leaves local area blank for non state-wide colleges."""
    header = ["SNO", "INSTCODE", "INST_REG", "LOCAL_AREA", "BRANCH_CODE", "OC_BOYS"]
    rows = [_row("1", "ACEE", "AU", "", "CSE", "1000")]

    frame = ingest.to_long(header, rows, SOURCE)

    assert frame.iloc[0]["local_area"] == "AU"
    assert bool(frame.iloc[0]["local_area_derived"]) is True


def test_state_wide_college_keeps_its_explicit_local_area():
    header = ["SNO", "INSTCODE", "INST_REG", "LOCAL_AREA", "BRANCH_CODE", "OC_BOYS"]
    rows = [_row("1", "BESTPU", "SW", "SVU", "CSE", "1000")]

    frame = ingest.to_long(header, rows, SOURCE)

    assert frame.iloc[0]["local_area"] == "SVU"
    assert bool(frame.iloc[0]["local_area_derived"]) is False


def test_state_wide_college_with_no_local_area_stays_null():
    """We never guess a region for a state-wide college."""
    header = ["SNO", "INSTCODE", "INST_REG", "LOCAL_AREA", "BRANCH_CODE", "OC_BOYS"]
    rows = [_row("1", "BESTPU", "SW", "", "CSE", "1000")]

    frame = ingest.to_long(header, rows, SOURCE)

    assert frame.iloc[0]["local_area"] is None


# --- Against the real source documents --------------------------------------


@pytest.mark.slow
def test_2024_xls_known_values():
    """Spot-check against the 2024 statement, which needs no PDF extraction.

    ACEE / CSE / OC_BOYS reads 128861 in the source spreadsheet.
    """
    source = next(
        s for s in ingest.load_sources() if s["academic_year"] == 2024
    )
    frame = ingest.ingest_source(source)

    row = frame[
        (frame["college_code"] == "ACEE")
        & (frame["branch_code"] == "CSE")
        & (frame["category"] == "OC")
        & (frame["gender"] == "BOYS")
    ]
    assert len(row) == 1
    assert int(row.iloc[0]["closing_rank"]) == 128861
    assert int(row.iloc[0]["fee_inr"]) == 43000
    assert row.iloc[0]["year"] == 2024


@pytest.mark.slow
def test_2024_eee_blank_stays_missing():
    """ACEE/EEE has a blank OC_BOYS cell in the source. It must stay missing."""
    source = next(s for s in ingest.load_sources() if s["academic_year"] == 2024)
    frame = ingest.ingest_source(source)

    row = frame[
        (frame["college_code"] == "ACEE")
        & (frame["branch_code"] == "EEE")
        & (frame["category"] == "OC")
        & (frame["gender"] == "BOYS")
    ]
    assert len(row) == 1
    assert pd.isna(row.iloc[0]["closing_rank"])


# --- Mapping files ----------------------------------------------------------


def test_every_branch_code_has_a_map_entry():
    """A new branch code must be added to the map, not silently unnamed."""
    frame = pd.read_parquet(config.CUTOFFS_PARQUET)
    mapped = set(pd.read_csv(config.BRANCH_MAP, dtype=str)["branch_code"])
    unmapped = set(frame["branch_code"].dropna()) - mapped
    assert not unmapped, f"branch codes missing from branch_map.csv: {sorted(unmapped)}"


def test_all_branch_names_are_labelled_unofficial():
    """No official code-to-name list exists, so nothing may claim to be official."""
    branches = pd.read_csv(config.BRANCH_MAP, dtype=str)
    assert set(branches["name_status"].dropna()) == {"unofficial"}


def test_low_confidence_branches_have_no_invented_name():
    """If we are not confident, the name is blank and the UI shows the code."""
    branches = pd.read_csv(config.BRANCH_MAP, dtype=str)
    low = branches[branches["confidence"] == "low"]
    assert low["branch_name"].isna().all()


def test_college_map_covers_every_code():
    frame = pd.read_parquet(config.CUTOFFS_PARQUET)
    mapped = set(pd.read_csv(config.COLLEGE_MAP, dtype=str)["college_code"])
    assert not set(frame["college_code"].dropna()) - mapped

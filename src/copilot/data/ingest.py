"""Turn the official AP EAPCET 'last rank' statements into one clean long table.

The four source documents say the same thing in three different layouts, so the
work here is:

  1. read each file (PDF via pdfplumber, XLS via pandas),
  2. recognise the columns *by their header text*, not by position,
  3. reshape from "one column per category x gender" to one row per
     (year, college, branch, category, gender),
  4. keep a pointer back to the exact page and serial number in the source, so
     any row can be checked against the original document by hand.

Design rule: an unrecognised column is a hard error, never a silent drop. If a
future year changes its layout again, this fails loudly.

Run:  python -m copilot.data.ingest
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from copilot import config

# --- Column recognition -----------------------------------------------------

#: Header text (after normalisation) -> canonical field name.
FIELD_ALIASES: dict[str, str] = {
    "SNO": "source_sno",
    "INSTCODE": "college_code",
    "INST_CODE": "college_code",
    "NAMEOFTHEINSTITUTION": "college_name",
    "INST_NAME": "college_name",
    "TYPE": "college_type",
    "INST_REG": "inst_region",
    "DIST": "district",
    "PLACE": "place",
    "COED": "coed",
    "AFFL": "affiliation",
    "AFFLIA.UNIV": "affiliation",
    "ESTD": "estd",
    "A_REG": "local_area",
    "LOCAL_AREA": "local_area",
    "BRANCH_CODE": "branch_code",
    "COLLFEE": "fee_inr",
}

#: Raw category token in the header -> canonical category label.
CATEGORY_TOKENS: dict[str, str] = {
    "OC": "OC",
    "SC": "SC",
    "SCI": "SC-I",
    "SCII": "SC-II",
    "SCIII": "SC-III",
    "ST": "ST",
    "BCA": "BC-A",
    "BCB": "BC-B",
    "BCC": "BC-C",
    "BCD": "BC-D",
    "BCE": "BC-E",
    "OC_EWS": "OC-EWS",
}

GENDERS = ("BOYS", "GIRLS")

#: Every legal category x gender column name, e.g. "BCB_GIRLS".
CATEGORY_COLUMNS: dict[str, tuple[str, str]] = {
    f"{token}_{gender}": (label, gender)
    for token, label in CATEGORY_TOKENS.items()
    for gender in GENDERS
}


def normalise_header(cell: object) -> str:
    """Collapse a header cell to a comparable token.

    PDF extraction wraps headers mid-word ('OC_EWS_G\\nIRLS'), and years differ
    on trailing dots ('INST_REG.' vs 'INST_REG'), so strip both.
    """
    text = "" if cell is None else str(cell)
    text = re.sub(r"\s+", "", text).upper()
    return text.rstrip(".")


def parse_rank(value: object) -> int | None:
    """Parse a closing-rank cell. Missing stays missing - never 0, never 999999.

    A blank means no candidate of that category was admitted to that branch.
    The 2022 statement writes this as the literal string 'NA'; the others leave
    the cell empty.
    """
    text = "" if value is None else str(value).strip()
    if text.lower() in config.NULL_TOKENS:
        return None
    digits = text.replace(",", "").strip()
    if not digits.lstrip("-").isdigit():
        return None
    number = int(digits)
    return number if number > 0 else None


def parse_int(value: object) -> int | None:
    """Parse a plain integer cell (fee, year established)."""
    return parse_rank(value)


def clean_text(value: object) -> str | None:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower() in config.NULL_TOKENS:
        return None
    return text


# --- Reading the raw files --------------------------------------------------


@dataclass(frozen=True)
class RawRow:
    """One physical row of a source document, plus where it came from."""

    values: list[object]
    page: int | None


def _is_data_row(row: list[object], width: int) -> bool:
    """A data row starts with a serial number and has the full column count."""
    if len(row) != width:
        return False
    first = "" if row[0] is None else str(row[0]).strip()
    if not first:
        return False
    try:
        return float(first) == int(float(first))
    except ValueError:
        return False


def read_pdf(path: Path) -> tuple[list[str], list[RawRow]]:
    """Extract the header and every data row from a last-rank PDF."""
    import pdfplumber

    header: list[str] | None = None
    rows: list[RawRow] = []

    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables():
                for raw in table:
                    normalised = [normalise_header(cell) for cell in raw]
                    if header is None and "SNO" in normalised:
                        header = normalised
                        continue
                    if header is not None and _is_data_row(raw, len(header)):
                        rows.append(RawRow(values=list(raw), page=page_number))

    if header is None:
        raise ValueError(f"No header row found in {path.name}")
    return header, rows


def read_xls(path: Path, sheet: str | None = None) -> tuple[list[str], list[RawRow]]:
    """Extract the header and every data row from the native .xls statement."""
    frame = pd.read_excel(path, sheet_name=sheet or 0, header=None, dtype=object)

    header_index: int | None = None
    for index, row in frame.iterrows():
        if any(normalise_header(cell) == "SNO" for cell in row.tolist()):
            header_index = int(index)
            break
    if header_index is None:
        raise ValueError(f"No header row found in {path.name}")

    header = [normalise_header(cell) for cell in frame.iloc[header_index].tolist()]
    rows = [
        RawRow(values=frame.iloc[i].tolist(), page=None)
        for i in range(header_index + 1, len(frame))
        if _is_data_row(frame.iloc[i].tolist(), len(header))
    ]
    return header, rows


# --- Wide -> long -----------------------------------------------------------


def map_columns(
    header: list[str],
    source_id: str,
    populated: frozenset[int] = frozenset(),
) -> tuple[dict[str, int], dict[int, tuple[str, str]]]:
    """Split the header into known fields and category columns.

    Raises if anything is unrecognised, so a new layout cannot slip through.

    ``populated`` is the set of column indexes that actually contain data. A
    column with a blank header is only dropped when it is also empty; a blank
    header over real data is an error, because that is how the 2023 fee column
    went missing on the first run (its header cell extracts as None).
    """
    fields: dict[str, int] = {}
    categories: dict[int, tuple[str, str]] = {}
    unknown: list[str] = []

    for index, name in enumerate(header):
        if name in FIELD_ALIASES:
            fields[FIELD_ALIASES[name]] = index
        elif name in CATEGORY_COLUMNS:
            categories[index] = CATEGORY_COLUMNS[name]
        elif name == "":
            if index in populated:
                raise ValueError(
                    f"{source_id}: column {index} has a blank header but contains "
                    "data. Name it explicitly via 'header_overrides' in "
                    "data/raw/sources.json rather than letting it be dropped."
                )
            continue  # genuinely empty trailing cell
        else:
            unknown.append(name)

    if unknown:
        raise ValueError(
            f"{source_id}: unrecognised column(s) {unknown}. "
            "Add an alias in FIELD_ALIASES or CATEGORY_COLUMNS - do not drop silently."
        )
    for required in ("college_code", "branch_code"):
        if required not in fields:
            raise ValueError(f"{source_id}: required column {required!r} missing")
    if not categories:
        raise ValueError(f"{source_id}: no category columns found")

    return fields, categories


def to_long(header: list[str], rows: list[RawRow], source: dict) -> pd.DataFrame:
    """Reshape one source document into long format."""
    populated = frozenset(
        index
        for index in range(len(header))
        if any(clean_text(row.values[index]) is not None for row in rows)
    )
    fields, categories = map_columns(header, source["id"], populated)
    year = source["academic_year"]
    records: list[dict] = []

    for row in rows:
        values = row.values

        def field(name: str) -> object:
            index = fields.get(name)
            return values[index] if index is not None else None

        inst_region = clean_text(field("inst_region"))
        local_area = clean_text(field("local_area"))
        # 2022 leaves local_area blank for colleges that admit from their own
        # region only, and fills it in just for state-wide (SW) colleges. Fall
        # back to the college's own region and record that we did so.
        local_area_derived = False
        if local_area is None and inst_region is not None and inst_region != "SW":
            local_area = inst_region
            local_area_derived = True

        base = {
            "year": year,
            "counselling_phase": source["counselling_phase"],
            "college_code": clean_text(field("college_code")),
            "college_name": clean_text(field("college_name")),
            "college_type": clean_text(field("college_type")),
            "district": clean_text(field("district")),
            "inst_region": inst_region,
            "local_area": local_area,
            "local_area_derived": local_area_derived,
            "place": clean_text(field("place")),
            "coed": clean_text(field("coed")),
            "affiliation": clean_text(field("affiliation")),
            "estd": parse_int(field("estd")),
            "branch_code": clean_text(field("branch_code")),
            "fee_inr": parse_int(field("fee_inr")),
            "source_url": source["source_url"],
            "source_file": source["file"],
            "source_page": row.page,
            "source_sno": clean_text(field("source_sno")),
        }

        for index, (category, gender) in categories.items():
            records.append(
                {
                    **base,
                    "category": category,
                    "gender": gender,
                    "closing_rank": parse_rank(values[index]),
                }
            )

    return pd.DataFrame.from_records(records)


# --- Normalisation ----------------------------------------------------------


def apply_mappings(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical branch names and college names from the mapping files.

    The mapping files are checked into git and reviewed by hand, so this step is
    auditable: anyone can see exactly which code became which name.
    """
    frame = frame.copy()

    if config.BRANCH_MAP.exists():
        branches = pd.read_csv(config.BRANCH_MAP, dtype=str)
        frame = frame.merge(
            branches[["branch_code", "branch_name", "name_status"]],
            on="branch_code",
            how="left",
        )
    else:
        frame["branch_name"] = None
        frame["name_status"] = None

    if config.COLLEGE_MAP.exists():
        colleges = pd.read_csv(config.COLLEGE_MAP, dtype=str)
        frame = frame.merge(
            colleges[["college_code", "college_name_canonical"]],
            on="college_code",
            how="left",
        )
        frame["college_name"] = frame["college_name_canonical"].fillna(
            frame["college_name"]
        )
        frame = frame.drop(columns=["college_name_canonical"])

    return frame


# --- Entry point ------------------------------------------------------------

COLUMN_ORDER = [
    "year",
    "counselling_phase",
    "college_code",
    "college_name",
    "district",
    "inst_region",
    "local_area",
    "local_area_derived",
    "college_type",
    "branch_code",
    "branch_name",
    "name_status",
    "category",
    "gender",
    "closing_rank",
    "fee_inr",
    "place",
    "coed",
    "affiliation",
    "estd",
    "source_url",
    "source_file",
    "source_page",
    "source_sno",
]


def load_sources() -> list[dict]:
    manifest = json.loads(config.SOURCES_MANIFEST.read_text(encoding="utf-8"))
    return manifest["sources"]


def ingest_source(source: dict) -> pd.DataFrame:
    path = config.RAW_DIR / source["file"]
    if source["format"] == "pdf":
        header, rows = read_pdf(path)
    elif source["format"] == "xls":
        header, rows = read_xls(path, source.get("sheet"))
    else:
        raise ValueError(f"Unsupported format {source['format']!r}")

    # Some PDFs lose a header cell in extraction; sources.json names it explicitly.
    for index, name in (source.get("header_overrides") or {}).items():
        header[int(index)] = normalise_header(name)

    return to_long(header, rows, source)


def build() -> pd.DataFrame:
    frames = []
    for source in load_sources():
        frame = ingest_source(source)
        print(
            f"  {source['file']}: {len(frame):,} long rows "
            f"({frame['college_code'].nunique()} colleges, "
            f"{frame['branch_code'].nunique()} branches)"
        )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = apply_mappings(combined)
    for column in COLUMN_ORDER:
        if column not in combined.columns:
            combined[column] = None

    # Nullable integers, so a missing rank stays missing instead of turning the
    # column into floats and printing ranks as '128861.0'.
    for column in ("closing_rank", "fee_inr", "estd", "source_page", "year"):
        combined[column] = pd.array(combined[column], dtype="Int64")

    return combined[COLUMN_ORDER].sort_values(
        ["year", "college_code", "branch_code", "category", "gender"]
    ).reset_index(drop=True)


def write(frame: pd.DataFrame) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(config.CUTOFFS_PARQUET, index=False)
    with sqlite3.connect(config.CUTOFFS_DB) as connection:
        frame.to_sql("cutoffs", connection, if_exists="replace", index=False)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_lookup "
            "ON cutoffs (year, category, gender, branch_code)"
        )


def main() -> None:
    print(f"Ingesting {config.EXAM_STATE} EAPCET last-rank statements...")
    frame = build()
    write(frame)
    print(f"\nWrote {len(frame):,} rows")
    print(f"  {config.CUTOFFS_PARQUET}")
    print(f"  {config.CUTOFFS_DB}")


if __name__ == "__main__":
    main()

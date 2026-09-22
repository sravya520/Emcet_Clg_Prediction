"""Sample random rows and show where each one came from in the source document.

The point is to make the table checkable by hand: for every sampled row we
print the source file, the page, the source serial number, the exact column
header the value was read from, and the raw line of text on that page. Anyone
can open the PDF at that page and compare.

Run:  python -m copilot.data.spotcheck [n] [seed]
"""

from __future__ import annotations

import re
import sys

import pandas as pd

from copilot import config
from copilot.data.ingest import CATEGORY_TOKENS, load_sources

#: canonical category label -> the token used in the source column header
LABEL_TO_TOKEN = {label: token for token, label in CATEGORY_TOKENS.items()}


def source_column_for(category: str, gender: str) -> str:
    """Reconstruct the column header this value was read from, e.g. 'BCB_GIRLS'."""
    return f"{LABEL_TO_TOKEN[category]}_{gender}"


def find_source_line(path, page_number: int, college_code: str, branch_code: str, value) -> str:
    """Return the raw text line on that page that holds this row.

    Warning: this line is only an aid to *finding* the row on the page. Blank
    cells vanish from the text layer, so the numbers in it cannot be counted
    positionally. Use ``extract_source_cells`` to check a value.
    """
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        text = pdf.pages[page_number - 1].extract_text() or ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    wanted = str(value) if pd.notna(value) else None

    # Prefer a line carrying both the branch code and the value.
    for line in lines:
        if re.search(rf"\b{re.escape(branch_code)}\b", line):
            if wanted is None or re.search(rf"\b{re.escape(wanted)}\b", line):
                return line
    for line in lines:
        if college_code in line:
            return line
    return "(line not located - open the page and search for the college code)"


def extract_source_cells(
    path, page_number: int, college_code: str, branch_code: str, sno: str
) -> dict[str, str]:
    """Re-extract this row from the PDF table and return every cell, keyed by header.

    Unlike the text line, table extraction preserves blank cells in position, so
    the value under each category column can be read off directly.
    """
    import pdfplumber

    from copilot.data.ingest import normalise_header

    with pdfplumber.open(path) as pdf:
        header: list[str] | None = None
        # The header only appears on the first page of some statements.
        for index in (0, page_number - 1):
            for table in pdf.pages[index].extract_tables():
                for raw in table:
                    normalised = [normalise_header(c) for c in raw]
                    if "SNO" in normalised:
                        header = normalised
                        break
                if header:
                    break
            if header:
                break

        for table in pdf.pages[page_number - 1].extract_tables():
            for raw in table:
                cells = [("" if c is None else str(c).strip()) for c in raw]
                if not cells:
                    continue
                if cells[0] == str(sno) and college_code in cells:
                    if header and len(header) == len(cells):
                        return dict(zip(header, cells))
                    return {str(i): v for i, v in enumerate(cells)}
    return {}


def extract_xls_cells(
    path, sheet, college_code: str, branch_code: str, sno: str
) -> dict[str, str]:
    """Re-read this row straight from the spreadsheet, keyed by header."""
    from copilot.data.ingest import normalise_header

    frame = pd.read_excel(path, sheet_name=sheet or 0, header=None, dtype=object)
    header_index = next(
        i
        for i, row in frame.iterrows()
        if any(normalise_header(c) == "SNO" for c in row.tolist())
    )
    header = [normalise_header(c) for c in frame.iloc[header_index].tolist()]

    for i in range(header_index + 1, len(frame)):
        cells = ["" if pd.isna(c) else str(c).strip() for c in frame.iloc[i].tolist()]
        if not cells:
            continue
        first = cells[0]
        try:
            matches_sno = str(int(float(first))) == str(sno)
        except ValueError:
            matches_sno = False
        if matches_sno and college_code in cells and branch_code in cells:
            return dict(zip(header, cells))
    return {}


def build(n: int = 10, seed: int = 20260922) -> pd.DataFrame:
    frame = pd.read_parquet(config.CUTOFFS_PARQUET)
    sources = {s["file"]: s for s in load_sources()}

    # Sample rows that actually carry a rank, so there is something to compare.
    population = frame[frame["closing_rank"].notna()]
    sample = population.sample(n=n, random_state=seed).sort_values(
        ["year", "college_code", "branch_code"]
    )

    records = []
    for _, row in sample.iterrows():
        source = sources[row["source_file"]]
        page = row["source_page"]
        cells: dict[str, str] = {}
        if source["format"] == "pdf" and pd.notna(page):
            location = f"page {int(page)}"
            line = find_source_line(
                config.RAW_DIR / row["source_file"],
                int(page),
                row["college_code"],
                row["branch_code"],
                row["closing_rank"],
            )
            cells = extract_source_cells(
                config.RAW_DIR / row["source_file"],
                int(page),
                row["college_code"],
                row["branch_code"],
                row["source_sno"],
            )
        else:
            location = f"sheet row SNO={row['source_sno']}"
            line = "(spreadsheet - open the sheet and filter on the SNO column)"
            cells = extract_xls_cells(
                config.RAW_DIR / row["source_file"],
                source.get("sheet"),
                row["college_code"],
                row["branch_code"],
                row["source_sno"],
            )

        records.append(
            {
                "year": int(row["year"]),
                "college_code": row["college_code"],
                "college_name": row["college_name"],
                "branch": row["branch_code"],
                "category": row["category"],
                "gender": row["gender"],
                "our_closing_rank": int(row["closing_rank"]),
                "source_file": row["source_file"],
                "where": location,
                "sno": row["source_sno"],
                "source_column": source_column_for(row["category"], row["gender"]),
                "raw_source_line": line,
                "source_cell": cells.get(
                    source_column_for(row["category"], row["gender"]), ""
                ),
                "cells": cells,
            }
        )

    return pd.DataFrame.from_records(records)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260922
    table = build(n, seed)

    print(f"# Spot check: {n} random rows (seed {seed})\n")
    for _, row in table.iterrows():
        print(f"## {row['college_code']} / {row['branch']} / {row['category']} {row['gender']} ({row['year']})")
        print(f"- College: {row['college_name']}")
        print(f"- Our value: **closing_rank = {row['our_closing_rank']:,}**")
        print(f"- Source: `{row['source_file']}`, {row['where']}, SNO {row['sno']}")
        print(f"- Read from source column: `{row['source_column']}`")
        if row["cells"]:
            cell = row["source_cell"]
            agrees = str(cell).replace(",", "") == str(row["our_closing_rank"])
            print(
                f"- Value in that cell, re-read from the source document: "
                f"**{cell or '(blank)'}** -> {'matches' if agrees else 'MISMATCH'}"
            )
        print(f"- Line to search for:\n\n      {row['raw_source_line']}\n")
        if row["cells"]:
            ranks = {
                k: (v or "(blank)")
                for k, v in row["cells"].items()
                if k in {source_column_for(c, g) for c in LABEL_TO_TOKEN for g in ("BOYS", "GIRLS")}
            }
            print(f"  Full row as extracted: `{ranks}`\n")


if __name__ == "__main__":
    main()

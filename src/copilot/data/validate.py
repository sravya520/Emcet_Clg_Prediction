"""Deterministic checks on the ingested table, written to a markdown report.

Nothing here is a judgement call: every check either passes, fails, or reports a
number. Anything that looks wrong should fail loudly rather than be smoothed
over, because a quietly-wrong cutoff is worse than a missing one.

Run:  python -m copilot.data.validate
Exit code is 1 if any check FAILs, so this can gate a build.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import pandas as pd

from copilot import config

REQUIRED_COLUMNS = [
    "year",
    "counselling_phase",
    "college_code",
    "college_name",
    "district",
    "local_area",
    "college_type",
    "branch_code",
    "category",
    "gender",
    "closing_rank",
    "source_url",
]

#: Number of category x gender columns expected in each year's statement.
EXPECTED_CATEGORY_COLUMNS = {2022: 18, 2023: 18, 2024: 18, 2025: 22}


@dataclass
class Check:
    name: str
    status: str  # PASS | FAIL | INFO
    detail: str
    table: pd.DataFrame | None = None


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str, table: pd.DataFrame | None = None) -> None:
        self.checks.append(Check(name, status, detail, table))

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "FAIL"]


def _pct(series: pd.Series) -> float:
    return round(float(series.isna().mean()) * 100, 2)


def run_checks(frame: pd.DataFrame) -> Report:
    report = Report()

    # 1. Required columns -----------------------------------------------------
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    report.add(
        "Required columns present",
        "PASS" if not missing else "FAIL",
        "All required columns are present."
        if not missing
        else f"Missing: {missing}",
    )

    # 2. Row counts per year --------------------------------------------------
    per_year = frame.groupby("year").agg(
        long_rows=("closing_rank", "size"),
        source_rows=("source_sno", lambda s: len(frame.loc[s.index].drop_duplicates(
            subset=["college_code", "branch_code", "local_area"]))),
        colleges=("college_code", "nunique"),
        branches=("branch_code", "nunique"),
        ranks_present=("closing_rank", "count"),
    )
    per_year["category_columns"] = (
        frame.groupby("year").apply(
            lambda g: g.groupby(["category", "gender"]).ngroups, include_groups=False
        )
    )
    per_year["pct_rank_missing"] = (
        100 * (1 - per_year["ranks_present"] / per_year["long_rows"])
    ).round(2)

    bad_shape = [
        year
        for year, expected in EXPECTED_CATEGORY_COLUMNS.items()
        if year in per_year.index and int(per_year.loc[year, "category_columns"]) != expected
    ]
    report.add(
        "Rows and category columns per year",
        "PASS" if not bad_shape else "FAIL",
        "Category x gender column counts match the source layouts."
        if not bad_shape
        else f"Unexpected category column count for {bad_shape}",
        per_year.reset_index(),
    )

    # 3. No negative or zero ranks -------------------------------------------
    ranks = frame["closing_rank"].dropna()
    non_positive = int((ranks <= 0).sum())
    report.add(
        "No negative or zero closing ranks",
        "PASS" if non_positive == 0 else "FAIL",
        f"{non_positive} non-positive rank(s) found."
        if non_positive
        else f"All {len(ranks):,} present ranks are positive "
        f"(min {int(ranks.min()):,}, max {int(ranks.max()):,}).",
    )

    # 4. Missing values per column, per year ---------------------------------
    missing_table = (
        frame.groupby("year")[
            [c for c in frame.columns if c not in ("year",)]
        ]
        .apply(lambda g: g.apply(_pct), include_groups=False)
        .reset_index()
    )
    report.add(
        "Percentage missing per column, per year",
        "INFO",
        "A blank closing rank means no candidate of that category was admitted "
        "to that college-branch. It is kept as missing, never imputed.",
        missing_table,
    )

    # 5. Duplicate keys -------------------------------------------------------
    key = ["year", "college_code", "branch_code", "local_area", "category", "gender"]
    duplicates = int(frame.duplicated(subset=key).sum())
    report.add(
        "No duplicate rows per key",
        "PASS" if duplicates == 0 else "FAIL",
        f"Key is {key}. {duplicates} duplicate row(s)."
        if duplicates
        else f"Key {key} is unique across all {len(frame):,} rows.",
    )

    # 6. College churn between consecutive years -----------------------------
    years = sorted(frame["year"].unique())
    churn_rows = []
    for earlier, later in zip(years, years[1:]):
        a = set(frame.loc[frame["year"] == earlier, "college_code"].dropna())
        b = set(frame.loc[frame["year"] == later, "college_code"].dropna())
        churn_rows.append(
            {
                "pair": f"{earlier} -> {later}",
                "in_both": len(a & b),
                f"only_earlier": len(a - b),
                f"only_later": len(b - a),
                "examples_only_earlier": ", ".join(sorted(a - b)[:6]),
                "examples_only_later": ", ".join(sorted(b - a)[:6]),
            }
        )
    report.add(
        "Colleges appearing in one year but not the next",
        "INFO",
        "Colleges open, close and change code between years. This is expected; "
        "it bounds how many college-branch pairs the backtest can match.",
        pd.DataFrame(churn_rows),
    )

    # 7. Category coverage per year ------------------------------------------
    coverage = (
        frame.groupby(["year", "category"])["closing_rank"]
        .size()
        .unstack(fill_value=0)
    )
    report.add(
        "Categories present per year",
        "INFO",
        "2025 replaces SC with SC-I / SC-II / SC-III. An SC row from 2024 has no "
        "one-to-one successor in 2025, so SC cannot be validated on the "
        "2024 -> 2025 fold.",
        coverage.reset_index(),
    )

    # 8. Fee coverage ---------------------------------------------------------
    fee = (
        frame.groupby("year")["fee_inr"]
        .agg(present="count", total="size")
        .assign(pct_present=lambda d: (100 * d["present"] / d["total"]).round(1))
        .reset_index()
    )
    report.add(
        "Fee coverage per year",
        "INFO",
        "The 2025 statement carries no fee column at all. 2024 is short of 100% "
        "because state-wide colleges carry the fee on only one of their two "
        "local-area rows.",
        fee,
    )

    # 9. Derived local area ---------------------------------------------------
    derived = (
        frame.groupby("year")["local_area_derived"]
        .agg(derived="sum", total="size")
        .assign(pct=lambda d: (100 * d["derived"] / d["total"]).round(1))
        .reset_index()
    )
    report.add(
        "Local area derived from the college's own region",
        "INFO",
        "2022 fills the local-area column only for state-wide colleges. For the "
        "rest we fall back to the college's own region and flag the row.",
        derived,
    )

    # 10. Branch names --------------------------------------------------------
    branches = (
        frame[["branch_code", "branch_name", "name_status"]]
        .drop_duplicates(subset=["branch_code"])
        .sort_values("branch_code")
    )
    unnamed = branches["branch_name"].isna().sum()
    unofficial = int((branches["name_status"] == "unofficial").sum())
    report.add(
        "Branch names",
        "INFO",
        f"{len(branches)} distinct branch codes. {unnamed} have no name and show "
        f"as the raw code. All {unofficial} named codes are marked UNOFFICIAL: no "
        "official code-to-name list was found in any counselling document.",
    )

    # 11. Local areas are the two AP regions ---------------------------------
    areas = sorted(frame["local_area"].dropna().unique())
    expected_areas = {"AU", "SVU"}
    report.add(
        "Local areas limited to the two AP regions",
        "PASS" if set(areas) <= expected_areas else "FAIL",
        f"Found {areas}. AP has two local areas (Andhra University and Sri "
        "Venkateswara University). OU is Telangana and must never appear.",
    )

    return report


def render(report: Report, frame: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Data validation report")
    lines.append("")
    lines.append(f"- Exam state: **{config.EXAM_STATE} EAPCET**, MPC stream")
    lines.append(f"- Total rows: **{len(frame):,}**")
    lines.append(f"- Years: **{', '.join(str(y) for y in sorted(frame['year'].unique()))}**")
    lines.append(f"- Counselling phase: **{', '.join(sorted(frame['counselling_phase'].unique()))}**")
    lines.append("")
    passed = sum(1 for c in report.checks if c.status == "PASS")
    failed = len(report.failed)
    info = sum(1 for c in report.checks if c.status == "INFO")
    lines.append(f"**{passed} passed, {failed} failed, {info} informational.**")
    lines.append("")
    lines.append("---")
    lines.append("")

    for check in report.checks:
        badge = {"PASS": "PASS", "FAIL": "FAIL", "INFO": "INFO"}[check.status]
        lines.append(f"## [{badge}] {check.name}")
        lines.append("")
        lines.append(check.detail)
        lines.append("")
        if check.table is not None and not check.table.empty:
            lines.append(check.table.to_markdown(index=False))
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "Generated by `python -m copilot.data.validate`. "
        "Every number here comes from the ingested table, not from a note written by hand."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    frame = pd.read_parquet(config.CUTOFFS_PARQUET)
    report = run_checks(frame)
    config.VALIDATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    config.VALIDATION_REPORT.write_text(render(report, frame), encoding="utf-8", newline="\n")

    for check in report.checks:
        print(f"[{check.status:4}] {check.name}")
    print(f"\nWrote {config.VALIDATION_REPORT}")

    if report.failed:
        print(f"\n{len(report.failed)} check(s) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

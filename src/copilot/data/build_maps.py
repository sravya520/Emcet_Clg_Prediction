"""Generate data/mappings/college_map.csv from the raw statements.

College names drift between years: typos ('SLEF FINAN' for 'SELF FINAN'),
spacing changes, and genuine renames (JNTUK -> JNTUGV). The college *code* is
stable, so we pick one canonical name per code and record every variant seen,
with the year it came from.

Rule: the canonical name is the one from the most recent year the college
appears in. That way a real rename wins over an older spelling, rather than a
majority vote silently keeping a stale name.

The output is committed to git and meant to be read by a human. Regenerate and
diff it whenever a new year is added:

    python -m copilot.data.build_maps
"""

from __future__ import annotations

import pandas as pd

from copilot import config
from copilot.data.ingest import ingest_source, load_sources


def build_college_map() -> pd.DataFrame:
    frames = [ingest_source(source) for source in load_sources()]
    combined = pd.concat(frames, ignore_index=True)

    pairs = (
        combined[["college_code", "college_name", "year"]]
        .dropna(subset=["college_code", "college_name"])
        .drop_duplicates()
        .sort_values(["college_code", "year"])
    )

    records = []
    for code, group in pairs.groupby("college_code"):
        latest = group.loc[group["year"].idxmax()]
        variants = sorted(set(group["college_name"]) - {latest["college_name"]})
        records.append(
            {
                "college_code": code,
                "college_name_canonical": latest["college_name"],
                "canonical_from_year": int(latest["year"]),
                "n_variants": len(variants) + 1,
                "other_variants": " | ".join(variants),
                "years_present": ",".join(str(y) for y in sorted(group["year"].unique())),
            }
        )

    return pd.DataFrame.from_records(records).sort_values("college_code")


def main() -> None:
    config.MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_college_map()
    frame.to_csv(config.COLLEGE_MAP, index=False, lineterminator="\n")
    conflicts = int((frame["n_variants"] > 1).sum())
    print(f"Wrote {config.COLLEGE_MAP}")
    print(f"  {len(frame)} colleges, {conflicts} with more than one spelling across years")


if __name__ == "__main__":
    main()

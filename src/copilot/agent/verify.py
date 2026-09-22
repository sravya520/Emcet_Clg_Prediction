"""The checker: nothing reaches the student unless a tool actually said it.

This is ordinary Python, not a second AI opinion. It collects every college
code, branch code and rank number that the tools returned during this turn,
then walks the drafted answer and removes anything that is not in those sets.

Why this is worth having: a language model asked for college options will
happily produce a plausible-looking college code it has never seen. A
plausible wrong cutoff is worse than no answer, because the student cannot
tell the difference. So the rule is blunt - if a tool did not say it this
turn, it does not go out.

What is checked
---------------
- college_code and branch_code on every recommendation
- closing_rank and data_year on every recommendation
- any rank-sized number appearing in the free-text reply
- any token in the reply shaped like a college code

What is not checked
-------------------
Ordinary prose. The checker cannot tell whether a sentence is a fair summary,
only whether the hard values in it were real. That gap is what the guardrail
prompt and, later, an LLM verification pass are for.
"""

from __future__ import annotations

import re
from typing import Any

from copilot.agent.schemas import Answer, CheckedAnswer, RemovedItem

#: A run of digits long enough to be a rank or a cutoff, with optional commas.
NUMBER_PATTERN = re.compile(r"\b\d{1,3}(?:,\d{2,3})+\b|\b\d{4,7}\b")

#: A bare token that looks like a college code: 3-8 capitals/digits. Deliberately
#: loose, then filtered against a stop-list, because a missed fake code is worse
#: than a false alarm we can whitelist.
CODE_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{2,7}\b")

#: Capitalised words that are not college codes.
NOT_CODES = {
    "SAFE", "MODERATE", "REACH", "AU", "SVU", "SW", "OC", "SC", "ST", "EWS",
    "BOYS", "GIRLS", "EAPCET", "EAMCET", "APSCHE", "AP", "MPC", "PVT", "UNIV",
    "SF", "PU", "SS", "CSE", "ECE", "EEE", "MEC", "CIV", "INF", "AND", "THE",
    "FOR", "YOU", "YOUR", "NOT", "ALL", "ANY", "ARE", "BUT", "CAN", "HAS",
    "OUR", "WAS", "WITH", "FROM", "THAT", "THIS", "THEY", "HAVE", "WILL",
    "NOTE", "DATA", "YEAR", "RANK", "SCI", "SCII", "SCIII", "III", "II",
    "PHD", "PHM", "IIT", "NIT", "JEE", "OU",
}


def collect_facts(tool_results: list[dict[str, Any]]) -> dict[str, set]:
    """Walk everything the tools returned and note the values we can vouch for."""
    colleges: set[str] = set()
    branches: set[str] = set()
    numbers: set[int] = set()
    years: set[int] = set()

    def walk(node: Any, key: str | None = None) -> None:
        if isinstance(node, dict):
            for child_key, child in node.items():
                walk(child, child_key)
        elif isinstance(node, list):
            for child in node:
                walk(child, key)
        elif isinstance(node, str):
            if key == "college_code":
                colleges.add(node.upper())
            elif key == "branch_code":
                branches.add(node.upper())
            elif key and key.endswith("year"):
                if node.isdigit():
                    years.add(int(node))
                    numbers.add(int(node))
        elif isinstance(node, bool):
            return
        elif isinstance(node, int):
            numbers.add(node)
            if key and "year" in key:
                years.add(node)
        elif isinstance(node, float):
            numbers.add(int(round(node)))

    for result in tool_results:
        walk(result)

    # closing_rank_by_year uses the year as a dict KEY, so it arrives as a
    # string key rather than a value. Pick those up too.
    for result in tool_results:
        _collect_year_keys(result, years, numbers)

    return {
        "colleges": colleges,
        "branches": branches,
        "numbers": numbers,
        "years": years,
    }


def _collect_year_keys(node: Any, years: set[int], numbers: set[int]) -> None:
    if isinstance(node, dict):
        for key, child in node.items():
            if isinstance(key, str) and key.isdigit() and len(key) == 4:
                years.add(int(key))
                numbers.add(int(key))
            _collect_year_keys(child, years, numbers)
    elif isinstance(node, list):
        for child in node:
            _collect_year_keys(child, years, numbers)


def _as_int(text: str) -> int | None:
    digits = text.replace(",", "")
    return int(digits) if digits.isdigit() else None


def check(answer: Answer, tool_results: list[dict[str, Any]]) -> CheckedAnswer:
    """Strip anything the tools did not say, and record every removal."""
    facts = collect_facts(tool_results)
    removed: list[RemovedItem] = []
    kept: list = []

    # --- recommendations -------------------------------------------------
    for index, rec in enumerate(answer.recommendations):
        where = f"recommendations[{index}]"

        if rec.college_code not in facts["colleges"]:
            removed.append(
                RemovedItem(
                    kind="college",
                    value=rec.college_code,
                    reason="college code did not appear in any tool result this turn",
                    where=where,
                )
            )
            continue

        if rec.branch_code not in facts["branches"]:
            removed.append(
                RemovedItem(
                    kind="branch",
                    value=rec.branch_code,
                    reason="branch code did not appear in any tool result this turn",
                    where=where,
                )
            )
            continue

        if rec.closing_rank is not None and rec.closing_rank not in facts["numbers"]:
            removed.append(
                RemovedItem(
                    kind="rank",
                    value=str(rec.closing_rank),
                    reason=(
                        f"closing rank for {rec.college_code}/{rec.branch_code} did "
                        "not appear in any tool result this turn"
                    ),
                    where=f"{where}.closing_rank",
                )
            )
            rec = rec.model_copy(update={"closing_rank": None})

        if facts["years"] and rec.data_year not in facts["years"]:
            removed.append(
                RemovedItem(
                    kind="rank",
                    value=str(rec.data_year),
                    reason="data year did not appear in any tool result this turn",
                    where=f"{where}.data_year",
                )
            )
            continue

        kept.append(rec)

    # --- free text -------------------------------------------------------
    reply, text_removals = _scrub_text(answer.reply, facts)
    removed.extend(text_removals)

    cleaned = answer.model_copy(update={"recommendations": kept, "reply": reply})
    return CheckedAnswer(
        answer=cleaned,
        removed=removed,
        passed_clean=not removed,
    )


def _scrub_text(text: str, facts: dict[str, set]) -> tuple[str, list[RemovedItem]]:
    """Replace unverifiable numbers and codes in prose with a marker."""
    removed: list[RemovedItem] = []

    def replace_number(match: re.Match) -> str:
        value = _as_int(match.group(0))
        if value is None or value in facts["numbers"]:
            return match.group(0)
        # Small numbers are ordinary prose ("3 options", "top 5"), not claims.
        if value < 1000:
            return match.group(0)
        removed.append(
            RemovedItem(
                kind="rank",
                value=match.group(0),
                reason="number in the reply did not appear in any tool result this turn",
                where="reply",
            )
        )
        return "[removed: unverified number]"

    def replace_code(match: re.Match) -> str:
        token = match.group(0)
        if token in NOT_CODES or token in facts["colleges"] or token in facts["branches"]:
            return token
        if not any(c.isdigit() for c in token) and len(token) <= 3:
            return token  # short all-letter words are prose, not codes
        removed.append(
            RemovedItem(
                kind="college",
                value=token,
                reason="code in the reply did not appear in any tool result this turn",
                where="reply",
            )
        )
        return "[removed: unverified code]"

    scrubbed = NUMBER_PATTERN.sub(replace_number, text)
    scrubbed = CODE_PATTERN.sub(replace_code, scrubbed)
    return scrubbed, removed

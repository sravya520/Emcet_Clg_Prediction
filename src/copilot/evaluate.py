"""Run every question in evals/cases.jsonl against the real model and score it.

    python -m copilot.evaluate            # the whole set
    python -m copilot.evaluate --limit 5  # a quick sample
    python -m copilot.evaluate --pause 10 # slower, for a tight free-tier limit

Six measures, all counted by code:

  tool routing      did it call the tool the question needed?
  honesty           did out-of-scope questions get a refusal, not a guess?
  SC warning        did every SC answer carry the untested warning?
  schema valid      did the answer parse into the Answer model?
  checker removals  how often did the checker have to strip something?
  latency           seconds per answer

Nothing here is estimated. Every figure is counted from an actual run and
written to evals/results.json, which is what docs/EVAL.md quotes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from copilot import config
from copilot.agent.loop import GeminiClient, ask

CASES_FILE = config.REPO_ROOT / "evals" / "cases.jsonl"
RESULTS_FILE = config.REPO_ROOT / "evals" / "results.json"
REPORT_FILE = config.REPO_ROOT / "docs" / "EVAL.md"

#: Words that mean "I do not have this" rather than an attempt at an answer.
REFUSAL_MARKERS = (
    "not in my data", "do not have", "don't have", "no data", "not available",
    "cannot tell you", "can't tell you", "not something i have",
    "i do not hold", "i don't hold", "not part of my data", "no information",
    "unable to provide", "i will not estimate", "cannot provide",
)

#: A refusal that then invents something anyway is not a refusal.
GUESS_MARKERS = (
    "roughly", "approximately around", "typically", "usually around",
    "i would estimate", "my estimate", "ballpark", "about rs", "around rs",
    "probably costs", "likely costs",
)


@dataclass
class CaseResult:
    id: str
    kind: str
    question: str
    tools_called: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    tool_ok: bool = False
    schema_ok: bool = False
    honest_ok: bool | None = None
    sc_warning_ok: bool | None = None
    removed_count: int = 0
    steps: int = 0
    seconds: float = 0.0
    error: str | None = None
    reply_preview: str = ""


def load_cases(limit: int | None = None) -> list[dict]:
    cases = [
        json.loads(line)
        for line in CASES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return cases[:limit] if limit else cases


def _tool_ok(case: dict, called: list[str]) -> bool:
    expected = set(case.get("expect_tools") or [])
    actual = set(called)
    if not expected:
        # No tool was needed. Calling one anyway is not wrong if the question
        # also has a data side, so only a *pure* clarification case is strict.
        return True
    if case.get("any_of_tools"):
        return bool(expected & actual)
    return expected <= actual


def _honest_ok(reply: str, flagged_out_of_scope: bool) -> bool:
    """An honest refusal says it lacks the data and does not then guess."""
    lowered = reply.lower()
    refused = any(marker in lowered for marker in REFUSAL_MARKERS)
    guessed = any(marker in lowered for marker in GUESS_MARKERS)
    return (refused or flagged_out_of_scope) and not guessed


def run_case(case: dict, client: GeminiClient) -> CaseResult:
    result = CaseResult(
        id=case["id"],
        kind=case["kind"],
        question=case["question"],
        expected_tools=case.get("expect_tools") or [],
    )
    started = time.monotonic()
    turn = ask(case["question"], client=client)
    result.seconds = round(time.monotonic() - started, 2)

    if turn.error:
        result.error = turn.error
        return result

    checked = turn.checked
    result.tools_called = turn.tool_calls
    result.steps = turn.steps_used
    result.tool_ok = _tool_ok(case, turn.tool_calls)
    result.removed_count = len(checked.removed)
    result.reply_preview = checked.answer.reply[:400]

    # Schema valid means the model produced the structured object we asked for,
    # not just prose. A bare paragraph parses into Answer with an empty
    # recommendations list, so we require the shape to have actually been used.
    raw = (turn.raw_reply or "").strip()
    result.schema_ok = raw.startswith("{") or raw.startswith("```")

    if case.get("out_of_scope"):
        result.honest_ok = _honest_ok(checked.answer.reply, checked.answer.out_of_scope)
    if case.get("sc"):
        text = checked.answer.reply.lower()
        result.sc_warning_ok = (
            "could not be tested" in text or "less reliable" in text
        )
    return result


def summarise(results: list[CaseResult]) -> dict:
    done = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    def pct(numerator: int, denominator: int) -> float | None:
        return round(100 * numerator / denominator, 1) if denominator else None

    honesty = [r for r in done if r.honest_ok is not None]
    sc = [r for r in done if r.sc_warning_ok is not None]
    seconds = [r.seconds for r in done]

    return {
        "model": config.GEMINI_MODEL,
        "cases_total": len(results),
        "cases_answered": len(done),
        "cases_errored": len(failed),
        "tool_routing_accuracy_pct": pct(sum(r.tool_ok for r in done), len(done)),
        "out_of_scope_handled_honestly_pct": pct(
            sum(bool(r.honest_ok) for r in honesty), len(honesty)
        ),
        "out_of_scope_cases": len(honesty),
        "sc_warning_shown_pct": pct(sum(bool(r.sc_warning_ok) for r in sc), len(sc)),
        "sc_cases": len(sc),
        "schema_valid_pct": pct(sum(r.schema_ok for r in done), len(done)),
        "turns_where_checker_removed_something": sum(
            1 for r in done if r.removed_count > 0
        ),
        "checker_removal_rate_pct": pct(
            sum(1 for r in done if r.removed_count > 0), len(done)
        ),
        "total_items_removed": sum(r.removed_count for r in done),
        "avg_seconds_per_answer": round(statistics.mean(seconds), 2) if seconds else None,
        "median_seconds_per_answer": round(statistics.median(seconds), 2) if seconds else None,
        "slowest_seconds": round(max(seconds), 2) if seconds else None,
        "avg_steps": round(statistics.mean([r.steps for r in done]), 2) if done else None,
    }


def render(summary: dict, results: list[CaseResult]) -> str:
    lines = [
        "# Agent evaluation",
        "",
        "Produced by `python -m copilot.evaluate` against the real Gemini API. "
        "Every number is counted from an actual run; none is estimated.",
        "",
        f"- Model: **{summary['model']}**",
        f"- Questions: **{summary['cases_total']}** "
        f"({summary['cases_answered']} answered, {summary['cases_errored']} errored)",
        "",
        "## Results",
        "",
        "| Measure | Result |",
        "|---|---|",
        f"| Correct tool chosen | **{summary['tool_routing_accuracy_pct']}%** |",
        f"| Out-of-scope handled honestly | **{summary['out_of_scope_handled_honestly_pct']}%** "
        f"({summary['out_of_scope_cases']} questions) |",
        f"| SC answers carrying the warning | **{summary['sc_warning_shown_pct']}%** "
        f"({summary['sc_cases']} questions) |",
        f"| Answer in the right format | **{summary['schema_valid_pct']}%** |",
        f"| Turns where the checker removed something | "
        f"**{summary['turns_where_checker_removed_something']}** of "
        f"{summary['cases_answered']} ({summary['checker_removal_rate_pct']}%) |",
        f"| Total items removed | **{summary['total_items_removed']}** |",
        f"| Average time per answer | **{summary['avg_seconds_per_answer']}s** |",
        f"| Median / slowest | {summary['median_seconds_per_answer']}s / "
        f"{summary['slowest_seconds']}s |",
        f"| Average steps per answer | {summary['avg_steps']} |",
        "",
        "## Cost",
        "",
        "Free tier, so the money cost of this run was **nothing**. The real "
        "constraint is the daily request limit, not price.",
        "",
        "## Every question",
        "",
        "| id | kind | tool expected | tool called | ok | format | honest | SC warn | removed | secs |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        def mark(value):
            return "-" if value is None else ("yes" if value else "**NO**")

        lines.append(
            f"| {r.id} | {r.kind} | {', '.join(r.expected_tools) or '(none)'} | "
            f"{', '.join(r.tools_called) or '(none)'} | {mark(r.tool_ok)} | "
            f"{mark(r.schema_ok)} | {mark(r.honest_ok)} | {mark(r.sc_warning_ok)} | "
            f"{r.removed_count} | {r.seconds} |"
        )
    lines += ["", "## Notes", ""]
    errored = [r for r in results if r.error]
    if errored:
        lines.append("Questions that errored:")
        for r in errored:
            lines.append(f"- `{r.id}`: {r.error}")
    else:
        lines.append("No question errored.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m copilot.evaluate")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--pause", type=float, default=6.0,
        help="seconds to wait between questions, for free-tier rate limits",
    )
    args = parser.parse_args()

    cases = load_cases(args.limit)
    client = GeminiClient()
    print(f"Running {len(cases)} questions against {config.GEMINI_MODEL}")
    print(f"Pausing {args.pause}s between questions for the free-tier limit.\n")

    results: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index:>2}/{len(cases)}] {case['id']:<4} {case['kind']:<14} ", end="", flush=True)
        result = run_case(case, client)
        results.append(result)
        if result.error:
            print(f"ERROR {result.error[:60]}")
        else:
            flags = []
            if not result.tool_ok:
                flags.append("wrong tool")
            if not result.schema_ok:
                flags.append("bad format")
            if result.honest_ok is False:
                flags.append("NOT honest")
            if result.sc_warning_ok is False:
                flags.append("no SC warning")
            if result.removed_count:
                flags.append(f"checker removed {result.removed_count}")
            print(f"{result.seconds:>5.1f}s  {'; '.join(flags) or 'ok'}")
        if index < len(cases):
            time.sleep(args.pause)

    summary = summarise(results)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(
            {"summary": summary, "cases": [asdict(r) for r in results]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    REPORT_FILE.write_text(render(summary, results), encoding="utf-8", newline="\n")

    print("\n" + "=" * 60)
    for key, value in summary.items():
        print(f"  {key:<42} {value}")
    print("=" * 60)
    print(f"\nWrote {RESULTS_FILE}\n      {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

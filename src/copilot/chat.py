"""Try the agent from the command line.

    python -m copilot.chat "rank 34000 OC boy, CSE options?"

Add --json to see the structured answer, or --debug to see the tool calls.
"""

from __future__ import annotations

import argparse
import logging
import sys

from copilot.agent.loop import ask


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m copilot.chat",
        description="Ask Counselling Copilot a question.",
    )
    parser.add_argument("question", nargs="+", help="what to ask")
    parser.add_argument("--json", action="store_true", help="print the structured answer")
    parser.add_argument("--debug", action="store_true", help="show tool calls")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.INFO, format="  [%(name)s] %(message)s")

    question = " ".join(args.question)
    print(f"\n> {question}\n")

    turn = ask(question)

    if turn.error:
        print(f"Something went wrong: {turn.error}", file=sys.stderr)
        return 1

    checked = turn.checked
    if checked is None:
        print("No answer was produced.", file=sys.stderr)
        return 1

    if args.json:
        print(checked.model_dump_json(indent=2))
        return 0

    print(checked.answer.reply)

    if checked.answer.recommendations:
        print()
        for rec in checked.answer.recommendations:
            rank = f"{rec.closing_rank:,}" if rec.closing_rank else "not available"
            name = rec.branch_name or rec.branch_code
            print(
                f"  [{rec.band:<8}] {rec.college_code:<8} {name:<42} "
                f"closed {rank} ({rec.data_year})"
            )

    print()
    details = [
        f"{turn.steps_used} step(s)",
        f"{len(turn.tool_calls)} tool call(s): {', '.join(turn.tool_calls) or 'none'}",
        f"{turn.latency_seconds:.1f}s",
    ]
    print("  " + " | ".join(details))

    if checked.removed:
        print(f"\n  CHECKER REMOVED {len(checked.removed)} item(s):")
        for item in checked.removed:
            print(f"    - {item.kind} {item.value!r} ({item.where}): {item.reason}")
    else:
        print("  checker: nothing removed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

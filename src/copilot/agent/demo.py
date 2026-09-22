"""Three worked conversations, runnable with or without an API key.

    python -m copilot.agent.demo

With GEMINI_API_KEY set, this talks to the real model. Without one, it runs a
scripted stand-in so the demo still works offline and gives the same output
every time.

What is real either way: the tools, the data behind them, the loop, the step
cap and the checker. Only the model's wording and its choice of tool are
scripted in offline mode. That matters for the third conversation - the
checker really does strip an invented college out of a real answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from copilot import config
from copilot.agent import tools
from copilot.agent.loop import GeminiClient, ask


# --- A stand-in model that replays a script ---------------------------------


@dataclass
class _Call:
    name: str
    args: dict


@dataclass
class _Part:
    text: str | None = None
    function_call: _Call | None = None


@dataclass
class _Content:
    parts: list
    role: str = "model"


@dataclass
class _Candidate:
    content: _Content


@dataclass
class _Response:
    candidates: list
    text: str | None = None


@dataclass
class ScriptedModel:
    """Replays scripted turns. Entries may be callables that see tool results."""

    script: list
    model: str = "scripted-stand-in"
    results: list = field(default_factory=list)

    def generate(self, contents, tool_declarations):
        # Read tool results out of the transcript, the same way a real model
        # would see them, rather than being handed them out of band.
        self.results = _tool_results_in(contents)
        entry = self.script.pop(0)
        if callable(entry):
            entry = entry(self.results)
        return entry


def _tool_results_in(contents) -> list:
    """Pull every function result already present in the conversation."""
    found = []
    for content in contents:
        for part in getattr(content, "parts", None) or []:
            response = getattr(part, "function_response", None)
            if response is None:
                continue
            payload = getattr(response, "response", None) or {}
            if isinstance(payload, dict) and "result" in payload:
                found.append(payload["result"])
    return found


def tool_call(name: str, **arguments) -> _Response:
    return _Response(
        candidates=[_Candidate(_Content([_Part(function_call=_Call(name, arguments))]))],
        text=None,
    )


def reply(payload: dict) -> _Response:
    text = json.dumps(payload, ensure_ascii=False)
    return _Response(candidates=[_Candidate(_Content([_Part(text=text)]))], text=text)


# --- The three conversations ------------------------------------------------


def _answer_with_real_options(results: list) -> _Response:
    """Build a truthful answer from whatever recommend_options actually returned."""
    result = results[0]
    picks = result["options"][:3]
    return reply(
        {
            "reply": (
                f"Based on the {result['data_year']} closing ranks, here are CSE "
                f"options for rank {result['student']['rank']:,} (OC, boys, AU region). "
                f"I found {result['counts']['Safe']} Safe, "
                f"{result['counts']['Moderate']} Moderate and "
                f"{result['counts']['Reach']} Reach options. These are last year's "
                "results, not a promise about this year."
            ),
            "recommendations": [
                {
                    "college_code": option["college_code"],
                    "college_name": option["college_name"],
                    "branch_code": option["branch_code"],
                    "branch_name": option["branch_name"],
                    "band": option["band"],
                    "closing_rank": option["closing_rank"],
                    "data_year": option["data_year"],
                    "why": f"Closed at {option['closing_rank']:,} in {option['data_year']}.",
                }
                for option in picks
            ],
            "data_year": result["data_year"],
            "counselling_phase": result["counselling_phase"],
            "out_of_scope": False,
            "sc_warning_shown": False,
        }
    )


def _answer_out_of_scope(results: list) -> _Response:
    details = results[0]
    return reply(
        {
            "reply": (
                f"Placements are not in my data, so I cannot tell you that. I hold "
                f"official closing ranks only. What I can tell you about "
                f"{details['college_name']} ({details['college_code']}) is that it "
                f"appears in the statements for {', '.join(str(y) for y in details['years_present'])}, "
                f"it is in district {details['district']} and it is type "
                f"{details['college_type']}. For placement figures you would need the "
                "college itself; I will not estimate."
            ),
            "recommendations": [],
            "data_year": max(details["years_present"]),
            "out_of_scope": True,
            "sc_warning_shown": False,
        }
    )


def _answer_with_an_invented_college(results: list) -> _Response:
    """Deliberately mixes one real option with one invented one.

    This is the case the checker exists for. The invented college, its invented
    cutoff, and the invented number in the prose must all be stripped.
    """
    result = results[0]
    real = result["options"][0]
    return reply(
        {
            "reply": (
                f"For rank {result['student']['rank']:,} (SC-I, boys, AU) the "
                f"{result['data_year']} data gives you some options. "
                f"{real['college_code']} is a good bet. You should also look at "
                "SRMAP, which closed around 51200 last year. "
                + result["warning"]
            ),
            "recommendations": [
                {
                    "college_code": real["college_code"],
                    "college_name": real["college_name"],
                    "branch_code": real["branch_code"],
                    "branch_name": real["branch_name"],
                    "band": real["band"],
                    "closing_rank": real["closing_rank"],
                    "data_year": real["data_year"],
                    "why": f"Closed at {real['closing_rank']:,}.",
                },
                {
                    # None of this came from a tool. All of it must be removed.
                    "college_code": "SRMAP",
                    "college_name": "SRM University AP",
                    "branch_code": "CSE",
                    "branch_name": "Computer Science and Engineering",
                    "band": "Moderate",
                    "closing_rank": 51200,
                    "data_year": 2025,
                    "why": "Invented by the model - the checker should remove this.",
                },
            ],
            "data_year": result["data_year"],
            "counselling_phase": result["counselling_phase"],
            "out_of_scope": False,
            "sc_warning_shown": True,
        }
    )


CONVERSATIONS: list[tuple[str, str, Callable[[], list]]] = [
    (
        "1. A straightforward request for options",
        "rank 34000 OC boy, CSE options?",
        lambda: [
            tool_call(
                "recommend_options",
                rank=34000, category="OC", gender="BOYS",
                local_area="AU", branch=["CSE"], limit=15,
            ),
            _answer_with_real_options,
        ],
    ),
    (
        "2. A question we hold no data on",
        "what are the placements like at ADIT for CSE?",
        lambda: [
            tool_call("get_option_details", college_code="ADIT", branch_code="CSE"),
            _answer_out_of_scope,
        ],
    ),
    (
        "3. The model invents a college, and the checker catches it",
        "rank 45000 SC-I boy AU, what are my options?",
        lambda: [
            tool_call(
                "recommend_options",
                rank=45000, category="SC-I", gender="BOYS",
                local_area="AU", limit=15,
            ),
            _answer_with_an_invented_college,
        ],
    ),
]


def _make_client(script: list):
    """Real model when a key is configured, scripted stand-in otherwise."""
    if config.GEMINI_API_KEY and config.GEMINI_MODEL:
        return GeminiClient(), True
    return ScriptedModel(script=script), False


def run() -> None:
    using_real = bool(config.GEMINI_API_KEY and config.GEMINI_MODEL)
    banner = (
        f"Using the real model ({config.GEMINI_MODEL})"
        if using_real
        else "No GEMINI_API_KEY set - using a scripted stand-in for the model.\n"
        "Tools, data, loop, step cap and checker are all real."
    )
    print("=" * 78)
    print("Counselling Copilot - three worked conversations")
    print(banner)
    print("=" * 78)

    for title, question, build_script in CONVERSATIONS:
        script = build_script()
        client, _ = _make_client(script)

        print(f"\n\n{title}")
        print("-" * len(title))
        print(f"\nSTUDENT: {question}\n")

        turn = ask(question, client=client)

        if turn.error:
            print(f"  error: {turn.error}")
            continue

        checked = turn.checked
        print("TOOLS CALLED:", ", ".join(turn.tool_calls) or "none")
        print(f"STEPS USED:   {turn.steps_used} of {config.MAX_AGENT_STEPS}")
        print(f"\nCOPILOT:\n{_wrap(checked.answer.reply)}")

        if checked.answer.recommendations:
            print("\nOPTIONS SHOWN TO THE STUDENT:")
            for rec in checked.answer.recommendations:
                rank = f"{rec.closing_rank:,}" if rec.closing_rank else "not available"
                print(
                    f"  [{rec.band:<8}] {rec.college_code:<8} "
                    f"{(rec.branch_name or rec.branch_code):<42} "
                    f"closed {rank} ({rec.data_year})"
                )

        print(f"\nDATA YEAR SHOWN: {checked.answer.data_year}")
        print(f"OUT OF SCOPE:    {checked.answer.out_of_scope}")
        print(f"SC WARNING:      {checked.answer.sc_warning_shown}")

        if checked.removed:
            print(f"\n*** CHECKER REMOVED {len(checked.removed)} ITEM(S) ***")
            for item in checked.removed:
                print(f"  - {item.kind:<8} {item.value!r} in {item.where}")
                print(f"    reason: {item.reason}")
        else:
            print("\nCHECKER: nothing removed, every value traced to a tool result.")

    print("\n" + "=" * 78)


def _wrap(text: str, width: int = 76) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width))


if __name__ == "__main__":
    run()

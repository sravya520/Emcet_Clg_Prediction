"""Tests for the agent loop, the checker and the tools.

Every test here uses a fake AI. Nothing in this file touches the network or
needs an API key, so the suite runs offline, costs nothing and cannot be
flaky because of a rate limit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from copilot.agent import tools, verify
from copilot.agent.loop import Turn, _parse_answer, ask
from copilot.agent.schemas import Answer, Recommendation


# --- A fake Gemini ----------------------------------------------------------


@dataclass
class FakeCall:
    name: str
    args: dict


@dataclass
class FakePart:
    text: str | None = None
    function_call: FakeCall | None = None


@dataclass
class FakeContent:
    parts: list
    role: str = "model"


@dataclass
class FakeCandidate:
    content: FakeContent


@dataclass
class FakeResponse:
    candidates: list
    text: str | None = None


@dataclass
class FakeGemini:
    """Replays a scripted list of replies, one per step."""

    script: list
    model: str = "fake-model"
    seen: list = field(default_factory=list)
    tools_offered: list = field(default_factory=list)

    def generate(self, contents, tool_declarations):
        self.seen.append(contents)
        self.tools_offered.append(tool_declarations)
        if not self.script:
            raise AssertionError("fake model ran out of scripted replies")
        return self.script.pop(0)


def say(payload: dict | str) -> FakeResponse:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return FakeResponse(candidates=[FakeCandidate(FakeContent([FakePart(text=text)]))], text=text)


def call(name: str, **arguments) -> FakeResponse:
    return FakeResponse(
        candidates=[
            FakeCandidate(FakeContent([FakePart(function_call=FakeCall(name, arguments))]))
        ],
        text=None,
    )


# No API key is needed anywhere in this file. The loop imports google.genai's
# `types` module, which is plain data classes and touches no network, and every
# test injects its own fake model in place of the real client. So the suite is
# offline, free and cannot fail because of a rate limit.


def test_no_test_in_this_file_constructs_a_real_client():
    """Guard: if someone later forgets to inject a fake, this will catch it."""
    import inspect

    import tests.test_agent as this_module

    source = inspect.getsource(this_module)
    # Split so this line does not match itself.
    forbidden = "GeminiClient" + "("
    assert forbidden not in source.replace(f'"GeminiClient" + "("', "")


# --- The loop ---------------------------------------------------------------


def test_a_plain_answer_needs_no_tools():
    model = FakeGemini(script=[say({"reply": "Hello", "recommendations": []})])
    turn = ask("hi", client=model)
    assert turn.checked.answer.reply == "Hello"
    assert turn.tool_calls == []
    assert turn.steps_used == 1


def test_the_loop_runs_a_tool_then_answers():
    model = FakeGemini(
        script=[
            call("explain_bands"),
            say({"reply": "Safe means comfortably inside last year's line.", "data_year": 2025}),
        ]
    )
    turn = ask("what does Safe mean?", client=model)
    assert turn.tool_calls == ["explain_bands"]
    assert turn.steps_used == 2
    assert turn.tool_results[0]["bands"][0]["band"] == "Safe"


def test_the_step_cap_is_enforced():
    """A model that only ever calls tools must still be cut off."""
    model = FakeGemini(script=[call("explain_bands") for _ in range(10)])
    turn = ask("loop forever", client=model, max_steps=3)
    assert turn.steps_used == 3
    assert len(turn.tool_calls) <= 3


def test_tools_are_withdrawn_on_the_final_step():
    """On the last allowed step the model gets no tools, forcing an answer."""
    model = FakeGemini(script=[call("explain_bands"), call("explain_bands"), say({"reply": "done"})])
    ask("q", client=model, max_steps=3)
    assert model.tools_offered[0] is not None
    assert model.tools_offered[-1] is None


def test_a_bad_tool_name_is_reported_not_crashed():
    model = FakeGemini(script=[call("no_such_tool"), say({"reply": "sorry"})])
    turn = ask("q", client=model)
    assert "error" in turn.tool_results[0]
    assert turn.checked is not None


def test_bad_tool_arguments_are_reported_not_crashed():
    model = FakeGemini(
        script=[call("recommend_options", rank="not-a-number"), say({"reply": "sorry"})]
    )
    turn = ask("q", client=model)
    assert "error" in turn.tool_results[0]


def test_an_api_failure_is_captured_on_the_turn():
    class Broken:
        model = "x"

        def generate(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    turn = ask("q", client=Broken())
    assert turn.error is not None and "boom" in turn.error
    assert turn.checked is None


# --- Parsing ----------------------------------------------------------------


def test_json_in_a_code_fence_is_parsed():
    answer = _parse_answer('```json\n{"reply": "hi", "data_year": 2025}\n```')
    assert answer.reply == "hi" and answer.data_year == 2025


def test_plain_prose_still_becomes_an_answer():
    answer = _parse_answer("just some text")
    assert answer.reply == "just some text"
    assert answer.recommendations == []


def test_a_bad_band_is_rejected_by_the_schema():
    with pytest.raises(Exception):
        Recommendation(
            college_code="X", college_name="X", branch_code="CSE",
            band="Definitely", data_year=2025, why="",
        )


# --- The checker ------------------------------------------------------------


@pytest.fixture(scope="module")
def real_result():
    return tools.recommend_options(34000, "OC", "BOYS", "AU", branch=["CSE"], limit=3)


def test_an_invented_college_is_removed(real_result):
    answer = Answer(
        reply="Try MITBLR.",
        recommendations=[
            Recommendation(
                college_code="MITBLR", college_name="Made Up", branch_code="CSE",
                band="Safe", closing_rank=12345, data_year=2025, why="invented",
            )
        ],
    )
    checked = verify.check(answer, [real_result])
    assert checked.answer.recommendations == []
    assert not checked.passed_clean
    assert any(item.kind == "college" for item in checked.removed)


def test_a_real_college_survives(real_result):
    option = real_result["options"][0]
    answer = Answer(
        reply="A real suggestion.",
        recommendations=[
            Recommendation(
                college_code=option["college_code"],
                college_name=option["college_name"],
                branch_code=option["branch_code"],
                band=option["band"],
                closing_rank=option["closing_rank"],
                data_year=option["data_year"],
                why="from the tool",
            )
        ],
    )
    checked = verify.check(answer, [real_result])
    assert len(checked.answer.recommendations) == 1
    assert checked.passed_clean


def test_a_real_college_with_an_invented_cutoff_keeps_the_college_drops_the_number(real_result):
    option = real_result["options"][0]
    answer = Answer(
        reply="ok",
        recommendations=[
            Recommendation(
                college_code=option["college_code"],
                college_name=option["college_name"],
                branch_code=option["branch_code"],
                band="Safe", closing_rank=999_111, data_year=2025, why="wrong number",
            )
        ],
    )
    checked = verify.check(answer, [real_result])
    assert len(checked.answer.recommendations) == 1
    assert checked.answer.recommendations[0].closing_rank is None
    assert any(item.kind == "rank" for item in checked.removed)


def test_an_invented_number_in_prose_is_scrubbed(real_result):
    answer = Answer(reply="The cutoff was 98765 last year.")
    checked = verify.check(answer, [real_result])
    assert "98765" not in checked.answer.reply
    assert "[removed:" in checked.answer.reply


def test_a_real_number_in_prose_survives(real_result):
    real_rank = real_result["options"][0]["closing_rank"]
    answer = Answer(reply=f"The cutoff was {real_rank} last year.")
    checked = verify.check(answer, [real_result])
    assert str(real_rank) in checked.answer.reply
    assert checked.passed_clean


def test_small_numbers_in_prose_are_left_alone(real_result):
    answer = Answer(reply="I found 3 options across 2 bands.")
    checked = verify.check(answer, [real_result])
    assert "3 options" in checked.answer.reply


def test_band_words_are_not_mistaken_for_college_codes(real_result):
    answer = Answer(reply="These are Safe and Moderate under AU for OC students.")
    checked = verify.check(answer, [real_result])
    assert checked.passed_clean, checked.removed


def test_the_checker_only_trusts_this_turn(real_result):
    """A college from some other turn's results must not slip through."""
    answer = Answer(
        reply="ok",
        recommendations=[
            Recommendation(
                college_code=real_result["options"][0]["college_code"],
                college_name="x", branch_code="CSE", band="Safe",
                closing_rank=None, data_year=2025, why="",
            )
        ],
    )
    checked = verify.check(answer, [])  # no tool results at all this turn
    assert checked.answer.recommendations == []
    assert not checked.passed_clean


# --- Honesty rules ----------------------------------------------------------


def test_the_prompt_forbids_fees_placements_and_quality():
    from copilot.agent.loop import SYSTEM_PROMPT

    lowered = SYSTEM_PROMPT.lower()
    for subject in ("fee", "placement", "quality", "hostel"):
        assert subject in lowered
    assert "not in your data" in lowered
    assert "never estimate" in lowered


def test_sc_students_get_a_warning_from_the_tool():
    result = tools.recommend_options(45_000, "SC-I", "BOYS", "AU", limit=3)
    assert result["band_untested_for_category"] is True
    assert result["warning"] and "could not be tested" in result["warning"]


def test_non_sc_students_get_no_warning():
    result = tools.recommend_options(45_000, "BC-B", "BOYS", "AU", limit=3)
    assert result["band_untested_for_category"] is False
    assert result["warning"] is None


def test_every_recommendation_carries_its_data_year():
    result = tools.recommend_options(34_000, "OC", "BOYS", "AU", limit=5)
    assert result["data_year"] == 2025
    assert all(option["data_year"] == 2025 for option in result["options"])


def test_compare_explains_why_something_is_missing():
    result = tools.compare_options(
        [{"college_code": "NOPE", "branch_code": "XXX"}], category="OC", gender="BOYS"
    )
    assert result["rows"] == []
    assert "no record" in result["not_found"][0]["reason"]


def test_explain_bands_reads_measured_numbers_not_hardcoded_ones():
    result = tools.explain_bands()
    assert result["holdout_accuracy"], "backtest results file missing"
    assert set(result["holdout_accuracy"]) == {"Safe", "Moderate", "Reach"}
    assert "SC was excluded" in result["holdout_note"]


# --- Category spellings -----------------------------------------------------


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("bcb", "BC-B"), ("BC-B", "BC-B"), ("BC B", "BC-B"),
        ("oc", "OC"), ("general", "OC"), ("ews", "OC-EWS"),
        ("sc1", "SC-I"), ("SCII", "SC-II"), ("sc-3", "SC-III"),
        ("st", "ST"),
    ],
)
def test_category_spellings(written, expected):
    assert tools.normalise_category(written) == expected

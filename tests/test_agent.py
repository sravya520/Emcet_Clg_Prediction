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
    return tools.recommend_options(34000, "OC", "BOYS", "AU", branch=["CSE"], per_band=3)


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
    result = tools.recommend_options(45_000, "SC-I", "BOYS", "AU", per_band=3)
    assert result["band_untested_for_category"] is True
    assert result["warning"] and "could not be tested" in result["warning"]


def test_non_sc_students_get_no_warning():
    result = tools.recommend_options(45_000, "BC-B", "BOYS", "AU", per_band=3)
    assert result["band_untested_for_category"] is False
    assert result["warning"] is None


def test_every_recommendation_carries_its_data_year():
    result = tools.recommend_options(34_000, "OC", "BOYS", "AU", per_band=5)
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


# --- Checker: invented names and rounded numbers ----------------------------
#
# The loophole these close: the checker used to trust any WORD the tools had
# emitted. "Aditya Engineering College" is built entirely from words that appear
# in real tool output, yet no such college was ever returned.


def test_a_name_built_from_real_words_is_still_removed(real_result):
    """Every word is real. The name is not. It must not survive."""
    answer = Answer(reply="You should look at Aditya Engineering College for CSE.")
    checked = verify.check(answer, [real_result])
    assert "Aditya Engineering College" not in checked.answer.reply
    assert not checked.passed_clean
    assert any("did not match any name" in item.reason for item in checked.removed)


def test_an_invented_name_on_a_recommendation_is_removed(real_result):
    """A real college code paired with a made-up name is still a fabrication."""
    option = real_result["options"][0]
    answer = Answer(
        reply="ok",
        recommendations=[
            Recommendation(
                college_code=option["college_code"],
                college_name="Aditya Engineering College",  # not what the tool said
                branch_code=option["branch_code"],
                band="Safe",
                closing_rank=option["closing_rank"],
                data_year=option["data_year"],
                why="name is wrong",
            )
        ],
    )
    checked = verify.check(answer, [real_result])
    assert checked.answer.recommendations == []
    assert any(item.where.endswith("college_name") for item in checked.removed)


def test_the_real_full_name_survives(real_result):
    option = real_result["options"][0]
    answer = Answer(reply=f"{option['college_name']} is worth a look.")
    checked = verify.check(answer, [real_result])
    assert option["college_name"] in checked.answer.reply
    assert checked.passed_clean, checked.removed


def test_shorthand_k_numbers_are_checked(real_result):
    """'46k' is rounded, but it is still a claim about a cutoff."""
    answer = Answer(reply="That one closed around 93k last year.")
    checked = verify.check(answer, [real_result])
    assert "93k" not in checked.answer.reply
    assert any(item.kind == "rank" for item in checked.removed)


def test_shorthand_k_close_to_a_real_number_is_allowed(real_result):
    real = real_result["options"][0]["closing_rank"]
    answer = Answer(reply=f"That one closed around {round(real / 1000)}k last year.")
    checked = verify.check(answer, [real_result])
    assert "k last year" in checked.answer.reply


def test_an_approximate_number_is_flagged_as_approximate(real_result):
    """'about 46,000' when the real figure is 46,204 is not the real figure."""
    real = real_result["options"][0]["closing_rank"]
    rounded = (real // 1000) * 1000
    assert rounded != real, "pick a fixture whose cutoff is not a round thousand"
    answer = Answer(reply=f"The cutoff was about {rounded:,} last year.")
    checked = verify.check(answer, [real_result])
    assert f"{rounded:,}" not in checked.answer.reply
    assert any("rounded or approximate" in item.reason for item in checked.removed)


def test_branch_names_in_prose_are_not_false_positives(real_result):
    """'Computer Science and Engineering' is a real branch name from the tool."""
    answer = Answer(reply="These are all Computer Science and Engineering seats.")
    checked = verify.check(answer, [real_result])
    assert "Computer Science and Engineering" in checked.answer.reply
    assert checked.passed_clean, checked.removed


# --- Counts per band --------------------------------------------------------


def test_band_counts_are_the_true_totals_not_the_trimmed_ones():
    """The bug this pins: reporting 15 Safe / 0 Moderate / 0 Reach when the
    student actually had 104 / 10 / 3, because the limit was applied first."""
    result = tools.recommend_options(34_000, "OC", "BOYS", "AU", branch=["CSE"], per_band=5)
    totals = result["total_options_per_band"]
    assert totals["Safe"] > 5
    assert totals["Moderate"] > 0
    assert totals["Reach"] > 0
    assert result["total_options"] == sum(totals.values())


def test_every_band_is_represented_in_the_returned_options():
    result = tools.recommend_options(34_000, "OC", "BOYS", "AU", branch=["CSE"], per_band=5)
    shown = {option["band"] for option in result["options"]}
    assert shown == {"Safe", "Moderate", "Reach"}


def test_at_most_per_band_options_are_returned_for_each_band():
    result = tools.recommend_options(34_000, "OC", "BOYS", "AU", branch=["CSE"], per_band=3)
    from collections import Counter

    counts = Counter(option["band"] for option in result["options"])
    assert all(count <= 3 for count in counts.values())


def test_options_are_still_most_competitive_first_within_a_band():
    result = tools.recommend_options(34_000, "OC", "BOYS", "AU", branch=["CSE"], per_band=5)
    safe = [o["closing_rank"] for o in result["options"] if o["band"] == "Safe"]
    assert safe == sorted(safe)


def test_official_region_names_are_not_treated_as_invented(real_result):
    """AU and SVU are defined by name in the reservation G.O., not invented."""
    answer = Answer(reply="Are you in the AU (Andhra University) or SVU (Sri Venkateswara University) region?")
    checked = verify.check(answer, [real_result])
    assert "Andhra University" in checked.answer.reply
    assert "Sri Venkateswara University" in checked.answer.reply


def test_a_region_name_cannot_smuggle_in_a_college(real_result):
    """'Andhra University' is allowed exactly, never as a prefix for a college."""
    answer = Answer(reply="Try Andhra University College of Engineering for CSE.")
    checked = verify.check(answer, [real_result])
    assert "Andhra University College of Engineering" not in checked.answer.reply
    assert not checked.passed_clean


def test_numbers_printed_inside_a_tool_sentence_are_trusted():
    """explain_bands says tested_on '2024->2025'. The model must be able to
    mention 2024 without the checker calling it an invention."""
    bands = tools.explain_bands()
    answer = Answer(reply="These were measured on 2024 to 2025 transitions.")
    checked = verify.check(answer, [bands])
    assert "2024" in checked.answer.reply
    assert checked.passed_clean, checked.removed


def test_indian_lakh_grouping_is_read_correctly(real_result):
    """Gemini writes 1,51,961 not 151,961. Both must resolve to the same number."""
    details = tools.get_option_details("ADIT", "CSE")
    ranks = [r["closing_rank"] for r in details["closing_ranks"] if r["closing_rank"] > 100000]
    assert ranks, "need a six-figure cutoff for this test"
    value = ranks[0]
    indian = f"{value:,}".replace(",", "")  # rebuild in lakh grouping
    indian = indian[:-3][:-2] + "," + indian[:-3][-2:] + "," + indian[-3:]
    answer = Answer(reply=f"That one closed at {indian}.")
    checked = verify.check(answer, [details])
    assert indian in checked.answer.reply, checked.removed


# --- Regression: non-string dict keys ---------------------------------------


def test_compare_options_output_does_not_crash_the_checker():
    """compare_options keys closing_rank_by_year by the year as an INT.

    That reached `"year" in key` inside the checker and crashed the whole
    turn part-way through a live eval run.
    """
    result = tools.compare_options(
        [{"college_code": "ADIT", "branch_code": "CSE"}], category="OC", gender="BOYS"
    )
    by_year = result["rows"][0]["closing_rank_by_year"]
    assert all(isinstance(k, int) for k in by_year), "fixture must have int keys"

    facts = verify.collect_facts([result])
    assert 2025 in facts["years"]
    assert any(v in facts["numbers"] for v in by_year.values())


def test_every_tool_output_survives_the_checker():
    """Run each tool for real and make sure the checker can digest its shape."""
    outputs = [
        tools.recommend_options(34_000, "OC", "BOYS", "AU", per_band=2),
        tools.get_option_details("ADIT", "CSE"),
        tools.compare_options(
            [{"college_code": "ADIT", "branch_code": "CSE"},
             {"college_code": "KITS", "branch_code": "CSE"}]
        ),
        tools.explain_bands(),
    ]
    for output in outputs:
        checked = verify.check(Answer(reply="ok"), [output])
        assert checked is not None


def test_a_year_used_as_an_int_key_is_trusted():
    result = tools.compare_options([{"college_code": "ADIT", "branch_code": "CSE"}])
    answer = Answer(reply="In 2023 it closed higher than in 2024.")
    checked = verify.check(answer, [result])
    assert "2023" in checked.answer.reply and "2024" in checked.answer.reply
    assert checked.passed_clean, checked.removed


# --- The student's own rank is not an invention -----------------------------


def test_the_students_own_rank_survives_in_the_reply(real_result):
    """"Based on your rank of [removed: unverified number]" is a bug.

    The rank came from the student. Repeating it back is not a fabrication.
    """
    question = "my rank is 99999999, OC boy AU, what can I get?"
    answer = Answer(reply="Based on your rank of 99999999 I found no options.")
    checked = verify.check(answer, [real_result], question=question)
    assert "99999999" in checked.answer.reply
    assert checked.passed_clean, checked.removed


def test_a_number_not_in_the_question_is_still_removed(real_result):
    """Trusting the question must not become a way in for anything else."""
    question = "my rank is 40000, OC boy AU"
    answer = Answer(reply="With rank 40000 you could get into somewhere that closed at 77777.")
    checked = verify.check(answer, [real_result], question=question)
    assert "40000" in checked.answer.reply
    assert "77777" not in checked.answer.reply


def test_commas_in_the_question_are_matched_without_them_in_the_reply(real_result):
    question = "my rank is 40,000 - OC boy AU"
    answer = Answer(reply="Your rank of 40000 is workable.")
    checked = verify.check(answer, [real_result], question=question)
    assert "40000" in checked.answer.reply


def test_a_college_code_from_the_question_survives(real_result):
    """Asked "is ADIT a good college?", the agent correctly answers without
    calling a tool, because quality is not in the data. The checker then used
    to delete ADIT from that reply, even though the student typed it."""
    question = "is ADIT a good college? should I pick it?"
    answer = Answer(reply="I have no data on whether ADIT is good. I only hold closing ranks.")
    checked = verify.check(answer, [], question=question)
    assert "ADIT" in checked.answer.reply
    assert checked.passed_clean, checked.removed


def test_a_code_not_in_the_question_is_still_removed(real_result):
    """Trusting the question must not become a way in for anything else.

    Note the fake code is 5 characters. The prose scanner only inspects tokens
    of 3 to 8 characters, because that is the length real college codes run to
    (ADIT, GVPW, MBUTPU1). A longer invented token would slip past the prose
    scan - though not past the recommendations check, which matches exactly.
    """
    question = "is ADIT a good college?"
    answer = Answer(reply="ADIT is fine, and so is FAKE9.")
    checked = verify.check(answer, [], question=question)
    assert "ADIT" in checked.answer.reply
    assert "FAKE9" not in checked.answer.reply


def test_trusting_the_question_does_not_let_a_whole_name_through(real_result):
    """Quoting the question back is fine. Inventing a college is not."""
    question = "what about colleges in Visakhapatnam?"
    answer = Answer(reply="Try Visakhapatnam Institute of Engineering.")
    checked = verify.check(answer, [real_result], question=question)
    assert "Visakhapatnam Institute of Engineering" not in checked.answer.reply

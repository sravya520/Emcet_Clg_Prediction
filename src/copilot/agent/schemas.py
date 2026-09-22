"""The exact shape an answer must take before a student ever sees it.

Free text is impossible to check. A structured answer is: every claim sits in
a named field, so the checker can hold each one up against what the tools
actually returned.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Recommendation(BaseModel):
    """One college-branch option. Every field must trace to a tool result."""

    college_code: str = Field(description="Exactly as the tool returned it")
    college_name: str
    branch_code: str
    branch_name: str | None = Field(
        default=None,
        description="Leave empty if the tool gave no name. Never invent one.",
    )
    band: str = Field(description="Safe, Moderate or Reach")
    closing_rank: int | None = Field(
        default=None, description="Last year's closing rank, as the tool returned it"
    )
    data_year: int = Field(description="The year that closing rank comes from")
    why: str = Field(description="One short sentence, based only on tool results")

    @field_validator("band")
    @classmethod
    def band_must_be_known(cls, value: str) -> str:
        allowed = {"Safe", "Moderate", "Reach"}
        cleaned = value.strip().title()
        if cleaned not in allowed:
            raise ValueError(f"band must be one of {sorted(allowed)}, got {value!r}")
        return cleaned

    @field_validator("college_code", "branch_code")
    @classmethod
    def codes_are_upper(cls, value: str) -> str:
        return value.strip().upper()


class Answer(BaseModel):
    """What the agent returns for one turn."""

    reply: str = Field(description="The answer in plain English")
    recommendations: list[Recommendation] = Field(default_factory=list)
    data_year: int | None = Field(
        default=None, description="The year the numbers come from"
    )
    counselling_phase: str | None = None
    out_of_scope: bool = Field(
        default=False,
        description="True when the question is about something we hold no data on",
    )
    sc_warning_shown: bool = Field(
        default=False,
        description="True when the SC bands-untested warning is included",
    )


class RemovedItem(BaseModel):
    """A value the checker stripped, and why."""

    kind: str = Field(description="college, branch or rank")
    value: str
    reason: str
    where: str = Field(default="", description="Which field it was found in")


class CheckedAnswer(BaseModel):
    """The answer after checking, plus what the checker did to it."""

    answer: Answer
    removed: list[RemovedItem] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    steps_used: int = 0
    passed_clean: bool = Field(
        default=True, description="True when the checker removed nothing"
    )

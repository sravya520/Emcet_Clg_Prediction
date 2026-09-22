"""The agent loop. About a hundred lines, and no framework.

The whole thing is:

    send the conversation to Gemini
    if it asked for a tool: run the tool, append the result, go again
    if it answered: parse, check, return
    if we hit the step cap: ask for a final answer with no tools

That is all a "framework-free tool-calling loop" is. It is written out here
rather than imported because the interesting parts - the step cap, what goes
in the transcript, what happens on a bad tool call, what the checker sees -
are exactly the parts a framework hides.

Retries are Tenacity, and only on transient API failures. A model that
returns a bad answer is not retried by Tenacity; that is the checker's job.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from copilot import config
from copilot.agent import tools, verify
from copilot.agent.schemas import Answer, CheckedAnswer

log = logging.getLogger("copilot.agent")

SYSTEM_PROMPT = """\
You are Counselling Copilot. You help students in Andhra Pradesh read their \
AP EAPCET counselling options.

HOW YOU WORK
- You never do rank arithmetic yourself. Call a tool and report what it returns.
- Every college, branch, band and closing rank you mention must come from a \
tool result in THIS conversation turn. Never recall a college from memory.
- If you need the student's rank, category, gender or region and do not have \
them, ask one short question instead of guessing.

WHAT YOU DO NOT HAVE
You hold closing ranks only. You have NO data on fees, placements, salaries, \
college quality or rankings, hostels, faculty, or counselling dates. If asked \
about any of those, say plainly that it is not in your data and offer what you \
do have. Never estimate, never say "typically", never reason from general \
knowledge. Set out_of_scope to true.

HONESTY
- Always state the data year in your reply.
- These are last year's results, not a promise about this year. The official \
statement says its ranks "shall in no way reflect the rank upto which seat can \
be allotted in the present academic year".
- If a tool returns a warning field, include that warning in your reply and \
set sc_warning_shown to true.
- Branch names are unofficial. If a tool gives no branch name, use the code.

ANSWER FORMAT
Reply with a single JSON object and nothing else:
{
  "reply": "your answer in plain English",
  "recommendations": [
    {"college_code": "...", "college_name": "...", "branch_code": "...",
     "branch_name": "...", "band": "Safe|Moderate|Reach",
     "closing_rank": 12345, "data_year": 2025, "why": "one short sentence"}
  ],
  "data_year": 2025,
  "counselling_phase": "...",
  "out_of_scope": false,
  "sc_warning_shown": false
}
Use an empty recommendations list when the question is not asking for options.
"""


class TransientAPIError(RuntimeError):
    """A failure worth retrying: rate limit, timeout, server error."""


def _use_os_trust_store() -> bool:
    """Trust the machine's own certificate store as well as the bundled one.

    Networks that inspect TLS - college wifi, office proxies, some antivirus -
    re-sign every connection with their own certificate authority. That CA is
    installed in the operating system, but Python ships its own separate bundle
    and does not see it, so the SDK fails with CERTIFICATE_VERIFY_FAILED on a
    machine where the browser works fine.

    This is not a way of skipping certificate checks. Certificates are still
    verified; we just also consult the store the rest of the machine uses.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
        return True
    except Exception:  # noqa: BLE001 - absence is fine, the default bundle stands
        return False


@dataclass
class Turn:
    """Everything that happened while answering one question."""

    question: str
    checked: CheckedAnswer | None = None
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    steps_used: int = 0
    latency_seconds: float = 0.0
    error: str | None = None
    raw_reply: str = ""


def _is_transient(error: Exception) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    markers = (
        "429", "resource_exhausted", "rate limit", "quota",
        "500", "502", "503", "504", "unavailable", "deadline",
        "timeout", "internal error", "overloaded",
    )
    return any(marker in text for marker in markers)


class GeminiClient:
    """Thin wrapper so the loop can be tested without touching the network."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        if api_key is None or model is None:
            api_key, model = config.require_llm_settings()
        _use_os_trust_store()
        from google import genai  # imported here so tests need no SDK key

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self.model = model

    @retry(
        retry=retry_if_exception_type(TransientAPIError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def generate(self, contents: list, tool_declarations: list[dict] | None) -> Any:
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.0,
            # We run the tools ourselves. The SDK must not do it for us, or the
            # step cap and the checker would both be bypassed.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }
        if tool_declarations:
            config_kwargs["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(**declaration)
                        for declaration in tool_declarations
                    ]
                )
            ]

        try:
            return self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as error:  # noqa: BLE001
            if _is_transient(error):
                log.warning("transient API error, will retry: %s", type(error).__name__)
                raise TransientAPIError(str(error)) from error
            raise


def _function_calls(response: Any) -> list:
    calls = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            call = getattr(part, "function_call", None)
            if call is not None and getattr(call, "name", None):
                calls.append(call)
    return calls


def _text_of(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text
    chunks = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None):
                chunks.append(part.text)
    return "\n".join(chunks)


def _parse_answer(text: str) -> Answer:
    """Pull the JSON object out of the reply, tolerating code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        cleaned = cleaned.removeprefix("json").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        return Answer(reply=text.strip())
    try:
        return Answer.model_validate_json(cleaned[start : end + 1])
    except (ValidationError, json.JSONDecodeError):
        return Answer(reply=text.strip())


def ask(question: str, client: GeminiClient | None = None, max_steps: int | None = None) -> Turn:
    """Answer one question. Runs tools, then the checker, then returns."""
    from google.genai import types

    client = client or GeminiClient()
    max_steps = max_steps or config.MAX_AGENT_STEPS
    started = time.monotonic()
    turn = Turn(question=question)

    contents: list = [types.Content(role="user", parts=[types.Part(text=question)])]

    try:
        for step in range(1, max_steps + 1):
            turn.steps_used = step
            last_step = step == max_steps

            response = client.generate(
                contents,
                # On the final allowed step we withdraw the tools, which forces
                # the model to answer with what it already has instead of
                # looping forever.
                None if last_step else tools.TOOL_DECLARATIONS,
            )
            calls = _function_calls(response)

            if not calls:
                turn.raw_reply = _text_of(response)
                break

            contents.append(response.candidates[0].content)
            for call in calls:
                arguments = dict(call.args or {})
                log.info("step %d calling %s", step, call.name)
                result = tools.call_tool(call.name, arguments)
                turn.tool_calls.append(call.name)
                turn.tool_results.append(result)
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=call.name, response={"result": result}
                            )
                        ],
                    )
                )
        else:
            turn.raw_reply = _text_of(response)

    except Exception as error:  # noqa: BLE001
        turn.error = f"{type(error).__name__}: {error}"
        turn.latency_seconds = time.monotonic() - started
        return turn

    answer = _parse_answer(turn.raw_reply)
    checked = verify.check(answer, turn.tool_results, question=question)
    checked.tool_calls = turn.tool_calls
    checked.steps_used = turn.steps_used
    turn.checked = checked
    turn.latency_seconds = time.monotonic() - started

    if checked.removed:
        for item in checked.removed:
            log.warning(
                "checker removed %s %r from %s: %s",
                item.kind, item.value, item.where, item.reason,
            )
    return turn

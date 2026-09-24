"""Tests for the HTTP API.

The central claim being tested: the form does not depend on the AI. So most
of this file deliberately breaks the AI in various ways and checks that
/recommend carries on regardless.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from copilot import config
from copilot.agent.loop import ERROR_KINDS, classify_error
from copilot.api import app

client = TestClient(app)


# --- The deterministic half -------------------------------------------------


def test_health_reports_status_without_leaking_the_key():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["form_available"] is True
    assert isinstance(body["chat_configured"], bool)
    # The key itself must never appear anywhere in the response.
    assert config.GEMINI_API_KEY == "" or config.GEMINI_API_KEY not in str(body)


def test_recommend_returns_all_three_bands_with_true_totals():
    body = client.post("/recommend", json={
        "rank": 34000, "category": "OC", "gender": "BOYS",
        "local_area": "AU", "branch": ["CSE"], "per_band": 5,
    }).json()
    totals = body["total_options_per_band"]
    assert totals["Safe"] > 5 and totals["Moderate"] > 0 and totals["Reach"] > 0
    assert body["data_year"] == config.RECOMMEND_YEAR
    shown = {o["band"] for o in body["options"]}
    assert shown == {"Safe", "Moderate", "Reach"}


def test_recommend_rejects_a_nonsense_rank():
    assert client.post("/recommend", json={
        "rank": 0, "category": "OC", "gender": "BOYS", "local_area": "AU",
    }).status_code == 422


def test_recommend_flags_sc_categories():
    body = client.post("/recommend", json={
        "rank": 45000, "category": "SC-I", "gender": "BOYS", "local_area": "AU",
    }).json()
    assert body["band_untested_for_category"] is True
    assert "could not be tested" in body["warning"]


def test_meta_always_states_the_year_and_the_caveats():
    body = client.get("/meta").json()
    assert body["data_year"] == config.RECOMMEND_YEAR
    assert body["branch_names_are_unofficial"] is True
    assert "no honest 2025 figure" in body["fee_note"]
    assert "shall in no way reflect" in body["source_disclaimer"]
    assert "could not be tested" in body["sc_warning"]


def test_filters_come_from_the_data_not_a_hardcoded_list():
    body = client.get("/filters").json()
    assert "OC" in body["categories"] and "SC-I" in body["categories"]
    assert set(body["local_areas"]) <= {"AU", "SVU"}, "OU is Telangana"
    assert any(b["code"] == "CSE" for b in body["branches"])
    assert len(body["districts"]) > 5


# --- The form must survive the AI being broken ------------------------------


BREAKAGES = [
    ("quota",      "429 RESOURCE_EXHAUSTED: quota exceeded"),
    ("auth",       "401 UNAUTHENTICATED: API key not valid"),
    ("unavailable", "503 UNAVAILABLE: model overloaded"),
    ("offline",    "ConnectError: failed to connect"),
]


@pytest.mark.parametrize(("kind", "message"), BREAKAGES)
def test_the_form_still_works_when_chat_is_broken(monkeypatch, kind, message):
    """This is the whole point of splitting the two endpoints."""
    def explode(*_args, **_kwargs):
        raise RuntimeError(message)

    monkeypatch.setattr("copilot.agent.loop.ask", explode, raising=False)

    body = client.post("/recommend", json={
        "rank": 34000, "category": "OC", "gender": "BOYS", "local_area": "AU",
    }).json()
    assert body["total_options"] > 0, f"/recommend broke when chat had a {kind} failure"


@pytest.mark.parametrize(("kind", "message"), BREAKAGES)
def test_chat_failures_are_reported_as_data_not_a_500(monkeypatch, kind, message):
    def explode(*_args, **_kwargs):
        raise RuntimeError(message)

    monkeypatch.setattr("copilot.agent.loop.ask", explode)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "present", raising=False)
    monkeypatch.setattr(config, "GEMINI_MODEL", "some-model", raising=False)

    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 200, "a broken AI must not be an HTTP error"
    body = response.json()
    assert body["ok"] is False
    assert body["error_kind"] == kind
    assert body["form_still_works"] is True
    assert body["message"] == ERROR_KINDS[kind]


def test_out_of_quota_and_bad_key_say_different_things():
    """Both read as 'it is broken', but they need different actions."""
    assert ERROR_KINDS["quota"] != ERROR_KINDS["auth"]
    assert "form" in ERROR_KINDS["quota"].lower()
    assert "form" in ERROR_KINDS["auth"].lower()
    assert "limit" in ERROR_KINDS["quota"].lower()
    assert "key" in ERROR_KINDS["auth"].lower()


def test_chat_says_so_when_no_key_is_configured(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "", raising=False)
    body = client.post("/chat", json={"message": "hello"}).json()
    assert body["ok"] is False
    assert body["error_kind"] == "not_configured"
    assert body["form_still_works"] is True


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("429 RESOURCE_EXHAUSTED", "quota"),
        ("quota exceeded for model", "quota"),
        ("401 UNAUTHENTICATED", "auth"),
        ("403 PERMISSION_DENIED", "auth"),
        ("API key not valid", "auth"),
        ("503 UNAVAILABLE", "unavailable"),
        ("500 internal error", "unavailable"),
        ("SSL: CERTIFICATE_VERIFY_FAILED", "offline"),
        ("Missing GEMINI_API_KEY", "not_configured"),
        ("TypeError: something else", "other"),
    ],
)
def test_error_classification(message, expected):
    assert classify_error(message) == expected


# --- Cold start -------------------------------------------------------------


def test_the_ui_waits_and_explains_instead_of_showing_an_error():
    """A free host sleeps after 15 minutes. The first visitor then waits about
    a minute. A blank page for that minute reads as 'this project is broken',
    which is the wrong conclusion about a service that is merely asleep."""
    from copilot import ui

    assert ui.WAKE_TIMEOUT >= 60, "shorter than a cold start, so it would give up too early"

    import inspect
    source = inspect.getsource(ui.wait_for_api)
    assert "Waking up the free server" in source
    assert "takes up to a minute" in source
    # It must poll rather than check once and give up.
    assert "for second in range" in source


def test_the_deploy_blueprint_never_contains_the_key():
    """render.yaml is committed, so the key must be prompted for, not stored."""
    import pathlib
    import re

    text = (pathlib.Path(__file__).resolve().parents[1] / "render.yaml").read_text(
        encoding="utf-8"
    )
    assert "sync: false" in text, "GEMINI_API_KEY must be marked sync:false"
    # No assignment of an actual value to the key anywhere in the file.
    assert not re.search(r"GEMINI_API_KEY[\s\S]{0,40}value:", text)


def test_the_container_entrypoint_starts_both_and_waits():
    import pathlib

    script = (pathlib.Path(__file__).resolve().parents[1] / "docker" / "start.sh").read_text(
        encoding="utf-8"
    )
    assert "uvicorn copilot.api:app" in script
    assert "streamlit run" in script
    # The UI must not start before the API can answer, or the first page load
    # races the API and shows the wake-up screen unnecessarily.
    assert "/health" in script
    # Render assigns the public port; the API port stays internal.
    assert "${PORT:-8501}" in script
    assert "127.0.0.1:${API_PORT}" in script


def test_the_api_root_explains_itself_instead_of_404ing():
    """A bare 404 at the API root reads as "the service is broken" when the
    service is fine and you are simply at the wrong address. This cost real
    confusion twice: once on Render, once locally."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "not the app" in body["note"]
    assert "8501" in body["the_app_is_at"]
    assert "POST /recommend" in body["endpoints"]


def test_health_names_which_llm_setting_is_missing(monkeypatch):
    """"No API key" was misleading when the key was present and the model
    name was not. It sent someone hunting for a problem with a good key."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", "present", raising=False)
    monkeypatch.setattr(config, "GEMINI_MODEL", "", raising=False)

    body = client.get("/health").json()
    assert body["chat_configured"] is False
    assert body["chat_missing_settings"] == ["GEMINI_MODEL"]
    # The key is present, so it must NOT be blamed.
    assert "GEMINI_API_KEY" not in body["chat_missing_settings"]


def test_health_names_a_missing_key_too(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "", raising=False)
    monkeypatch.setattr(config, "GEMINI_MODEL", "some-model", raising=False)
    body = client.get("/health").json()
    assert body["chat_missing_settings"] == ["GEMINI_API_KEY"]


def test_health_names_both_when_both_are_missing(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "", raising=False)
    monkeypatch.setattr(config, "GEMINI_MODEL", "", raising=False)
    body = client.get("/health").json()
    assert body["chat_missing_settings"] == ["GEMINI_API_KEY", "GEMINI_MODEL"]


def test_health_never_reveals_the_value_only_the_name(monkeypatch):
    secret = "super-secret-key-value"
    monkeypatch.setattr(config, "GEMINI_API_KEY", secret, raising=False)
    monkeypatch.setattr(config, "GEMINI_MODEL", "", raising=False)
    body = client.get("/health").json()
    assert secret not in str(body)


# --- A pasted setting with invisible whitespace ------------------------------


def test_settings_are_stripped(monkeypatch):
    """A value pasted into a hosting dashboard can carry a trailing space or
    newline that is invisible in the form field. An untrimmed model name
    produces a 400 from the API with no obvious cause."""
    import importlib

    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite \n")
    monkeypatch.setenv("GEMINI_API_KEY", "  a-key  ")
    importlib.reload(config)
    try:
        assert config.GEMINI_MODEL == "gemini-3.5-flash-lite"
        assert config.GEMINI_API_KEY == "a-key"
    finally:
        importlib.reload(config)


@pytest.mark.parametrize(
    "message",
    [
        "400 INVALID_ARGUMENT. GenerateContentRequest.model: unexpected model name",
        "404 NOT_FOUND. models/gemini-nope is not found for API version v1",
    ],
)
def test_a_rejected_model_name_is_its_own_error_kind(message):
    """It looks like a generic failure, but the cause is one wrong setting."""
    assert classify_error(message) == "bad_model"
    assert "model name" in ERROR_KINDS["bad_model"]
    assert "form" in ERROR_KINDS["bad_model"].lower()


def test_bad_model_is_reported_to_the_ui_as_such(monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("400 INVALID_ARGUMENT GenerateContentRequest.model: bad")

    monkeypatch.setattr("copilot.agent.loop.ask", explode)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "present", raising=False)
    monkeypatch.setattr(config, "GEMINI_MODEL", "bad ", raising=False)

    body = client.post("/chat", json={"message": "hi"}).json()
    assert body["ok"] is False
    assert body["error_kind"] == "bad_model"
    assert body["form_still_works"] is True


def test_the_chat_ui_renders_the_recommendations_it_receives():
    """The agent answers with prose AND a structured list. Rendering only the
    prose meant a reply saying "here are some options" was followed by nothing:
    the list was returned, checked, then dropped by the interface."""
    import inspect

    from copilot import ui

    source = inspect.getsource(ui)
    assert "def render_chat_recommendations" in source
    # Called for a fresh answer and when replaying history.
    assert source.count("render_chat_recommendations(") >= 3


def test_the_agent_is_told_to_honour_a_requested_count():
    from copilot.agent.loop import SYSTEM_PROMPT

    assert "per_band" in SYSTEM_PROMPT
    assert "top 10" in SYSTEM_PROMPT.lower()


def test_the_agent_is_told_not_to_describe_options_only_in_prose():
    from copilot.agent.loop import SYSTEM_PROMPT

    assert "recommendations` array" in SYSTEM_PROMPT or "recommendations array" in SYSTEM_PROMPT

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

"""Live status dashboard: login, privacy, and robot status.

Support conversations must never appear word for word, and nothing is
served without the dashboard password.
"""
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server import dashboard as d


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(d, "STATE", d._State())
    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw-test")
    yield


def _client():
    app = FastAPI()
    app.include_router(d.router)
    return TestClient(app)


def test_state_needs_the_password():
    c = _client()
    assert c.get("/dashboard/api/state").status_code == 401
    assert c.post("/dashboard/api/login", json={"password": "nope"}).status_code == 401
    assert c.post("/dashboard/api/login", json={"password": "pw-test"}).status_code == 200
    assert c.get("/dashboard/api/state").status_code == 200


def test_no_password_configured_serves_nothing(monkeypatch):
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    monkeypatch.delenv("NAO_SHARED_SECRET", raising=False)
    c = _client()
    assert c.get("/dashboard/api/state").status_code == 503
    assert c.post("/dashboard/api/login", json={"password": ""}).status_code == 503


def test_page_and_image_are_served():
    c = _client()
    assert "NAO status" in c.get("/dashboard").text
    assert c.get("/dashboard/nao.jpg").headers["content-type"] == "image/jpeg"


def _turn(agent, transcript="I feel awful about everything", reply="That sounds hard."):
    d.capture(None, "info", {"event": "stt_legacy", "session_id": "s1",
                             "transcript": transcript})
    d.capture(None, "info", {"event": "turn_complete", "session_id": "s1",
                             "outcome": "ok", "active_agent": agent,
                             "reply_preview": reply,
                             "phase_ms": {"e2e_user_to_first_audio": 1500}})
    return d.snapshot()["turns"][0]


@pytest.mark.parametrize("agent", ["therapist", "cbt_coach", "grounding_coach", "mi_coach"])
def test_support_turns_never_show_words(agent):
    t = _turn(agent)
    assert t["private"] is True
    assert t["question"] == "" and t["reply"] == ""


def test_crisis_turns_never_show_words():
    d.capture(None, "info", {"event": "stt_legacy", "session_id": "s1",
                             "transcript": "private words"})
    d.capture(None, "info", {"event": "turn_complete", "session_id": "s1",
                             "outcome": "crisis", "reply_preview": "988"})
    t = d.snapshot()["turns"][0]
    assert t["private"] and t["question"] == ""


def test_ordinary_turn_shows_question_and_answer():
    t = _turn("cs_direct", "Who teaches COSC 220?", "Jin Guo.")
    assert t["question"] == "Who teaches COSC 220?" and t["reply"] == "Jin Guo."
    s = d.snapshot()["today"]
    assert s["answered"] == 1 and s["cs"] == 1


def test_capture_never_raises_or_alters_events():
    ev = {"event": "turn_complete", "phase_ms": "garbage"}
    assert d.capture(None, "info", ev) is ev


def test_robot_state_logic(monkeypatch):
    assert d.snapshot()["robot"]["state"] == "unknown"
    d.capture(None, "info", {"event": "ws_connected", "session_id": "s1"})
    d.robot_connected("172.20.95.121")
    d.robot_status({"battery": 50, "life_state": "disabled"})
    assert d.snapshot()["robot"]["state"] == "online"
    d.capture(None, "info", {"event": "turn_complete", "session_id": "s1",
                             "outcome": "client_dropped"})
    d.STATE.robot_ssh_ok = True
    assert d.snapshot()["robot"]["state"] == "program_down"
    d.STATE.robot_ssh_ok = False
    assert d.snapshot()["robot"]["state"] == "offline"


def test_warnings_become_problems_but_listener_noise_does_not():
    d.capture(None, "warning", {"event": "agent_stream_error", "level": "warning",
                                "error": "API key is invalid"})
    d.capture(None, "debug", {"event": "mute_listener_no_match", "level": "warning"})
    probs = d.snapshot()["problems"]
    assert [p["event"] for p in probs] == ["agent_stream_error"]

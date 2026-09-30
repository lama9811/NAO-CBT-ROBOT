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
    yield


def _client():
    app = FastAPI()
    app.include_router(d.router)
    return TestClient(app)


def test_status_is_open_to_anyone_with_the_link():
    c = _client()
    r = c.get("/dashboard/api/state")
    assert r.status_code == 200 and "robot" in r.json()
    assert c.post("/dashboard/api/login", json={}).status_code in (404, 405)


def test_page_has_no_sign_in():
    page = (d._STATIC / "index.html").read_text()
    for gone in ('id="gate"', 'id="logout"', "/login"):
        assert gone not in page


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


def test_hook_is_live_under_the_default_structlog_chain():
    """The server never calls configure_logging, so the hook has to join the
    default chain; turns logged through structlog must reach the dashboard."""
    import structlog
    structlog.reset_defaults()
    d.install_log_hook()
    d.install_log_hook()  # idempotent
    names = [getattr(p, "__name__", "") for p in structlog.get_config()["processors"]]
    assert names.count("capture") == 1
    log = structlog.get_logger()
    log.info("stt_legacy", session_id="h1", transcript="Who teaches COSC 220?")
    log.info("turn_complete", session_id="h1", outcome="ok",
             active_agent="cs_direct", reply_preview="Jin Guo.")
    assert d.snapshot()["turns"][0]["question"] == "Who teaches COSC 220?"
    structlog.reset_defaults()


# ───────────────────────── hosted (Vercel) copy ──────────────────────────
def test_remote_payload_never_carries_support_words():
    _turn("therapist", "private words", "private reply")
    _turn("cs_direct", "Who teaches COSC 220?", "Jin Guo.")
    turns = d._remote_payload()["turns"]
    private = [t for t in turns if t["private"]]
    assert private and all(t["question"] == "" and t["reply"] == "" for t in private)
    assert any(t["question"] == "Who teaches COSC 220?" for t in turns)


def test_status_only_mode_strips_all_words(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PUSH_CONVERSATION", "0")
    _turn("cs_direct", "Who teaches COSC 220?", "Jin Guo.")
    p = d._remote_payload()
    assert p["conversation_hidden"] is True
    assert all(t["question"] == "" and t["reply"] == "" for t in p["turns"])


def test_push_is_off_until_configured(monkeypatch):
    import asyncio
    monkeypatch.delenv("DASHBOARD_REMOTE_URL", raising=False)
    monkeypatch.delenv("DASHBOARD_INGEST_SECRET", raising=False)
    # Returns immediately instead of looping forever.
    asyncio.run(asyncio.wait_for(d.push_remote_forever(), 1.0))


def test_vercel_copy_matches_the_pi_page():
    """vercel-dashboard/ serves copies; a drift would ship two different UIs."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for name in ("index.html", "nao.jpg"):
        assert (root / "server/dashboard_static" / name).read_bytes() == \
            (root / "vercel-dashboard" / name).read_bytes(), name


def test_page_picks_its_api_by_location():
    page = (d._STATIC / "index.html").read_text()
    assert 'ON_PI ? "/dashboard/api" : "/api"' in page


def test_page_explains_setup_problems_instead_of_blaming_the_pi():
    """A Vercel site without its database used to say "The Pi is not
    answering", sending people to check a Pi that was fine."""
    page = (d._STATIC / "index.html").read_text()
    assert 'error === "no_database"' in page
    assert "connect Upstash Redis" in page
    assert "Root Directory to vercel-dashboard" in page

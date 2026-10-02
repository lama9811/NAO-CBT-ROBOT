"""No student words in logs for support/crisis turns; rotated log file;
dashboard pushes status only by default."""
import json

import pytest
import structlog

from server import privacy


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    privacy._reset_for_tests()
    monkeypatch.delenv("LOG_REDACT", raising=False)
    yield
    privacy._reset_for_tests()


def _run(ev):
    return privacy.redact_processor(None, "info", dict(ev))


def test_therapist_turn_is_redacted():
    out = _run({"event": "turn_complete", "user": "a",
                "active_agent": "therapist", "transcript": "my secret",
                "reply_preview": "I hear you"})
    assert out["transcript"] == privacy.REDACTED
    assert out["reply_preview"] == privacy.REDACTED
    assert out["redacted"] is True


def test_crisis_event_is_redacted():
    out = _run({"event": "crisis_block", "user": "a", "transcript": "x y"})
    assert out["transcript"] == privacy.REDACTED


def test_user_stays_private_after_a_support_turn():
    _run({"event": "turn_complete", "user": "Ann", "active_agent": "cbt_coach",
          "transcript": "t"})
    # Next turn's STT line is written before routing.
    out = _run({"event": "stt_complete", "user": "ann",
                "transcript": "what about tomorrow"})
    assert out["transcript"] == privacy.REDACTED


def test_emotional_text_is_redacted_before_routing():
    out = _run({"event": "stt_complete", "user": "b",
                "transcript": "I feel so lonely"})
    assert out["transcript"] == privacy.REDACTED


def test_ordinary_turn_is_untouched():
    out = _run({"event": "turn_complete", "user": "c",
                "active_agent": "cs_direct",
                "transcript": "who teaches COSC 220"})
    assert out["transcript"] == "who teaches COSC 220"
    assert "redacted" not in out


def test_redaction_can_be_disabled(monkeypatch):
    monkeypatch.setenv("LOG_REDACT", "0")
    out = _run({"event": "crisis_block", "transcript": "x"})
    assert out["transcript"] == "x"


def test_install_puts_redaction_first_and_is_idempotent():
    try:
        privacy.install_log_redaction()
        privacy.install_log_redaction()
        procs = structlog.get_config()["processors"]
        names = [getattr(p, "__name__", "") for p in procs]
        assert names[0] == "redact_processor"
        assert names.count("redact_processor") == 1
    finally:
        structlog.reset_defaults()


def test_rendered_log_line_has_no_words(capsys):
    try:
        structlog.configure(processors=[
            privacy.redact_processor, structlog.processors.JSONRenderer()],
            logger_factory=structlog.PrintLoggerFactory())
        structlog.get_logger().info("turn_complete", user="z",
                                    active_agent="therapist",
                                    transcript="I failed again")
        line = capsys.readouterr().out
        assert "I failed again" not in line
        assert json.loads(line)["transcript"] == privacy.REDACTED
    finally:
        structlog.reset_defaults()


def test_rotating_log_file(tmp_path, monkeypatch):
    from server import logging_setup
    path = tmp_path / "server.jsonl"
    monkeypatch.setenv("LOG_FILE", str(path))
    monkeypatch.setenv("LOG_MAX_BYTES", "200")
    monkeypatch.setenv("LOG_BACKUPS", "2")
    monkeypatch.setattr(logging_setup, "_FILE_INSTALLED", False)
    import logging
    before = list(logging.getLogger().handlers)
    try:
        assert logging_setup.install_rotating_file() == str(path)
        log = structlog.get_logger()
        for i in range(30):
            log.info("tick", n=i)
        assert path.exists()
        assert (tmp_path / "server.jsonl.1").exists()
        assert not (tmp_path / "server.jsonl.3").exists()
    finally:
        for h in list(logging.getLogger().handlers):
            if h not in before:
                logging.getLogger().removeHandler(h)
                h.close()
        structlog.reset_defaults()


def test_rotating_log_file_off_by_default(monkeypatch):
    from server import logging_setup
    monkeypatch.delenv("LOG_FILE", raising=False)
    assert logging_setup.install_rotating_file() is None


def test_dashboard_pushes_status_only_by_default(monkeypatch):
    from server import dashboard as d
    monkeypatch.delenv("DASHBOARD_PUSH_CONVERSATION", raising=False)
    d.STATE.turns.clear()
    d._ingest_event({"event": "turn_complete", "session_id": "s",
                     "active_agent": "cs_direct", "outcome": "ok",
                     "transcript": "Who teaches COSC 220?",
                     "reply_preview": "Jin Guo."})
    p = d._remote_payload()
    assert p["conversation_hidden"] is True
    assert all(t["question"] == "" and t["reply"] == "" for t in p["turns"])


def test_dashboard_never_shows_redacted_turn_words():
    from server import dashboard as d
    d.STATE.turns.clear()
    d._ingest_event({"event": "turn_complete", "session_id": "s2",
                     "active_agent": "router", "outcome": "ok",
                     "transcript": privacy.REDACTED, "redacted": True,
                     "reply_preview": privacy.REDACTED})
    t = d.snapshot()["turns"][0]
    assert t["private"] and t["question"] == "" and t["reply"] == ""

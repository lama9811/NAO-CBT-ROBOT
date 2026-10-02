"""After a crisis hit: a no-text safety_events row, the optional webhook,
and a one-shot check-in on the next turn."""
import asyncio
import sqlite3
from unittest.mock import patch

import pytest

from server import config, safety


@pytest.fixture
def db(tmp_path, monkeypatch):
    from server import session as s
    path = str(tmp_path / "safety.db")
    monkeypatch.setattr(s, "_DB_PATH", path)
    return path


def _rows(path):
    with sqlite3.connect(path) as c:
        return c.execute("SELECT username, turn_index, clause, severity, "
                         "payload, created_at FROM safety_events").fetchall()


def test_record_crisis_event_writes_no_message_text(db):
    safety.record_crisis_event("alice", "keyword", 4)
    (row,) = _rows(db)
    assert row[:5] == ("alice", 4, "crisis_gate", "keyword", "")
    assert row[5]  # timestamp


def test_record_crisis_event_never_raises(monkeypatch):
    from server import session as s
    monkeypatch.setattr(s, "_DB_PATH", "/nonexistent/dir/x.db")
    safety.record_crisis_event("alice", "llm")  # no exception


def test_webhook_off_when_unset(monkeypatch):
    monkeypatch.setattr(config, "CRISIS_ALERT_WEBHOOK_URL", "")
    with patch("httpx.post") as post:
        assert safety.send_crisis_alert("keyword") is None
    assert not post.called


def test_webhook_posts_level_and_time_only(monkeypatch):
    monkeypatch.setattr(config, "CRISIS_ALERT_WEBHOOK_URL", "https://hook.example/x")
    with patch("httpx.post") as post:
        t = safety.send_crisis_alert("llm", ts=123.0)
        t.join(5)
    post.assert_called_once()
    assert post.call_args.kwargs["json"] == {"level": "llm", "ts": 123.0}


def test_webhook_failure_is_swallowed(monkeypatch):
    monkeypatch.setattr(config, "CRISIS_ALERT_WEBHOOK_URL", "https://hook.example/x")
    with patch("httpx.post", side_effect=OSError("down")):
        safety.send_crisis_alert("llm").join(5)


def test_followup_note_applies_once_to_a_string():
    conv = {"recent_user_turns": ["I want to kill"]}
    safety.mark_crisis(conv, "keyword")
    assert "recent_user_turns" not in conv
    first = safety.apply_crisis_followup(conv, "hi")
    assert first.startswith("[CRISIS_FOLLOWUP]") and first.endswith("hi")
    assert safety.apply_crisis_followup(conv, "hi") == "hi"


def test_followup_note_applies_to_multimodal_input():
    conv = {}
    safety.mark_crisis(conv, "llm")
    msg = [{"role": "user", "content": [
        {"type": "input_text", "text": "hello"},
        {"type": "input_image", "image_url": "data:..."}]}]
    out = safety.apply_crisis_followup(conv, msg)
    assert out[0]["content"][0]["text"].startswith("[CRISIS_FOLLOWUP]")


def test_remember_turn_keeps_a_short_window():
    conv = {}
    for i in range(6):
        safety.remember_turn(conv, f"turn {i}")
    assert conv["recent_user_turns"] == [
        f"turn {i}" for i in range(6 - safety.STITCH_TURNS, 6)]


def test_app_ws_after_crisis_hook(db, monkeypatch):
    from server import app_ws, privacy
    privacy._reset_for_tests()
    monkeypatch.setattr(config, "CRISIS_ALERT_WEBHOOK_URL", "")
    sess = app_ws._Session("Alice")
    conv = {}
    app_ws._after_crisis(sess, conv,
                         safety.CrisisResult(True, "keyword", stitched=True))
    assert conv["crisis_followup"] == {"level": "keyword_stitched"}
    assert privacy.is_private_user("Alice")
    (row,) = _rows(db)
    assert row[0] == "Alice" and row[3] == "keyword_stitched"


def test_emit_crisis_speaks_the_counseling_line(monkeypatch):
    from server import app_ws
    spoken = []

    class WS:
        async def send_text(self, t):
            pass

        async def send_json(self, p):
            pass

    def fake_synth(user, text, profile=None):
        spoken.append(text)
        return b"\xff\xfb\x90\x00"

    async def fake_chunk(ws, sess, frame, force=False):
        return True

    monkeypatch.setattr(app_ws, "_synth_for", fake_synth)
    monkeypatch.setattr(app_ws, "_send_audio_chunk", fake_chunk)
    asyncio.run(app_ws._emit_crisis(WS(), app_ws._Session("guest"),
                                    "words", {}))
    assert spoken == [safety.hotline_reply()]
    assert "443-885-3130" in spoken[0]

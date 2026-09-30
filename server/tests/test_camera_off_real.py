# -*- coding: utf-8 -*-
""""Camera off" must actually turn the camera off, and NAO must not do it
to itself.

Incident, 2026-09-30 14:48: NAO spoke its camera heads-up ("...Say 'stop
watching me' anytime."), heard itself, STT returned
"Station. Say stop watching me anytime.", and that matched the
`disable_camera` voice trigger. Three separate faults:

1. Both echo guards tokenized the quoted announcement as "'stop" / "me'",
   so the echo fell under their thresholds.
2. The announcement itself contained a camera-off command.
3. "Camera off" was theatre: the server never saved the choice and the
   robot logged "unknown action: disable_camera", then kept sending a
   photo after every reply.

`pytest-asyncio` is not installed, so coroutines run under asyncio.run().
"""
import asyncio
import json
import sys
import threading
import types

import pytest

from server import app_ws, config, motion_trigger

TODAY_ECHO = "Station. Say stop watching me anytime."


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_text(self, text):
        try:
            self.sent.append(json.loads(text))
        except (TypeError, ValueError):
            self.sent.append(text)

    async def send_json(self, payload):
        self.sent.append(payload)

    def frames(self, subtype):
        return [f for f in self.sent
                if isinstance(f, dict) and f.get("subtype") == subtype]

    def actions(self):
        return [f for f in self.sent
                if isinstance(f, dict) and f.get("type") == "action"]


@pytest.fixture
def db(tmp_path, monkeypatch):
    from server import session as s
    monkeypatch.setattr(s, "_DB_PATH", str(tmp_path / "cam.db"))
    return s


# ─────────────────────────────── 1. echo guards ──────────────────────────────
def test_tokens_strip_surrounding_quotes_but_keep_contractions():
    assert app_ws._echo_tokens("Say 'stop watching me' anytime.") == [
        "say", "stop", "watching", "me", "anytime"]
    assert app_ws._echo_tokens("I don't know") == ["i", "don't", "know"]
    assert "you'd" in app_ws._echo_tokens("If you\u2019d rather")


def test_todays_echo_is_caught_by_the_stateless_guard():
    assert app_ws._is_system_line_echo(TODAY_ECHO) is True


def test_todays_echo_is_caught_by_the_reply_history_guard():
    app_ws._reset_reply_chunks("guest", app_ws._CAMERA_ANNOUNCE_LEGACY)
    assert app_ws._is_substring_or_sentence_echo("guest", TODAY_ECHO) is True


def test_real_camera_commands_still_get_through():
    for cmd in ("stop watching me", "camera off", "don't watch me"):
        assert app_ws._is_system_line_echo(cmd) is False
        assert motion_trigger.detect(cmd).action == "disable_camera"


# ─────────────────────────── 2. announcement wording ─────────────────────────
@pytest.mark.parametrize("text", [
    config.CAMERA_ANNOUNCE_TEXT, app_ws._CAMERA_ANNOUNCE_FALLBACK])
def test_announcement_contains_no_camera_command(text):
    m = motion_trigger.detect(text)
    assert m is None or m.action not in ("disable_camera", "enable_camera"), (
        "NAO hears its own speaker; an announcement that contains a camera "
        "command turns the camera off by itself")


def test_new_announcement_echo_is_rejected():
    echo = "Heads up. My camera is on for this conversation."
    assert app_ws._is_system_line_echo(echo) is True


# ─────────────────────────── 3. camera off is real ───────────────────────────
def _run_motion(ws, sess, action, monkeypatch):
    monkeypatch.setattr(app_ws, "_synth_for", lambda *a, **k: b"")
    match = motion_trigger.MotionMatch(action=action, args={}, ack="Camera.")
    asyncio.run(app_ws._emit_motion(ws, sess, "x", match, {}))


def test_voice_camera_off_persists_and_tells_the_robot(db, monkeypatch):
    ws, sess = FakeWS(), app_ws._Session("guest")
    sess.image_b64 = "stale-frame"
    _run_motion(ws, sess, "disable_camera", monkeypatch)
    assert db.get_camera_consent("guest") is False
    assert sess.image_b64 is None
    assert ws.frames("camera_state")[-1]["data"]["on"] is False


def test_voice_camera_on_restores(db, monkeypatch):
    db.set_camera_consent("guest", False)
    ws, sess = FakeWS(), app_ws._Session("guest")
    _run_motion(ws, sess, "enable_camera", monkeypatch)
    assert db.get_camera_consent("guest") is True
    assert ws.frames("camera_state")[-1]["data"]["on"] is True


def test_server_drops_photos_while_camera_off(db):
    db.set_camera_consent("guest", False)
    ws, sess = FakeWS(), app_ws._Session("guest")
    asyncio.run(app_ws._ingest_frame(ws, sess, {"type": "image", "data": "abc"}))
    assert sess.image_b64 is None


def test_server_keeps_photos_while_camera_on(db):
    ws, sess = FakeWS(), app_ws._Session("guest")
    asyncio.run(app_ws._ingest_frame(ws, sess, {"type": "image", "data": "abc"}))
    assert sess.image_b64 == "abc"


# ───────────────────────────── robot side ────────────────────────────────────
def _load_ws_client():
    for name in ("naoqi", "qi"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.ALProxy = object
            mod.ALModule = object
            sys.modules[name] = mod
    if "nao" not in sys.path:
        sys.path.insert(0, "nao")
    return pytest.importorskip("ws_client")


class _Log:
    def __init__(self):
        self.events = []

    def __getattr__(self, level):
        return lambda event, **kw: self.events.append((event, kw))


def _client(wc):
    c = object.__new__(wc.NaoWsClient)
    c.log = _Log()
    c._camera_on = True
    c.dispatched = []
    c.action_dispatcher = lambda *a, **k: c.dispatched.append(a)
    return c


def test_robot_camera_action_is_handled_not_dispatched():
    wc = _load_ws_client()
    c = _client(wc)
    wc.NaoWsClient._handle_action(c, {"name": "disable_camera", "args": {}})
    assert c._camera_on is False
    wc.NaoWsClient._handle_action(c, {"name": "enable_camera", "args": {}})
    assert c._camera_on is True


def test_robot_takes_no_photo_while_camera_off():
    wc = _load_ws_client()
    c = _client(wc)
    c._camera_on = False
    started = []
    orig = threading.Thread.start
    threading.Thread.start = lambda self: started.append(self)
    try:
        wc.NaoWsClient._snap_and_push_image(c, reason="post_tts")
    finally:
        threading.Thread.start = orig
    assert started == []
    assert ("snap_skipped_camera_off", {"reason": "post_tts"}) in c.log.events

# -*- coding: utf-8 -*-
"""Wiring for the spoken mute command.

`test_mute_words.py` covers the matcher; this covers what the server does
with a match -- session state, the control frame the robot needs to cut
audio it has already queued, and the suppression choke point.

`pytest-asyncio` is not installed in this repo, so every coroutine is
driven with `asyncio.run()` rather than an async test function (an
`@pytest.mark.asyncio` test would silently skip).
"""
import asyncio
import json

from server import app_ws


class FakeWS:
    """Captures frames instead of sending them.

    `_send_json` serializes to a JSON *string* and calls `send_text`, so
    the fake has to decode rather than store dicts -- storing the raw call
    argument makes every frame assertion silently pass on an empty list.
    """

    def __init__(self):
        self.sent = []

    async def send_text(self, text):
        try:
            self.sent.append(json.loads(text))
        except (TypeError, ValueError):
            self.sent.append(text)

    async def send_json(self, payload):
        self.sent.append(payload)

    def frames_of(self, subtype):
        return [f for f in self.sent
                if isinstance(f, dict) and f.get("subtype") == subtype]


import pytest


@pytest.fixture(autouse=True)
def _no_real_tts(monkeypatch):
    """Mute/unmute now speak a confirmation; never hit a real TTS API."""
    monkeypatch.setattr(app_ws, "_synth_for", lambda *a, **k: b"\x01mp3")


def _session(username="tester"):
    return app_ws._Session(username)


def _mp3_frame(text="hello"):
    return app_ws._audio_chunk_frame(1, text, b"\x00\x01\x02")


class TestMuteCommand:
    def test_mute_sets_state_and_notifies_robot(self):
        ws, sess = FakeWS(), _session()
        handled = asyncio.run(app_ws._handle_mute_command(ws, sess, "Nao mute"))
        assert handled is True
        assert sess.muted is True
        # The robot must be told: the reply it is speaking was synthesized
        # before the command and is already queued in its player.
        assert ws.frames_of("mute"), "no mute control frame sent"

    def test_unmute_clears_state(self):
        ws, sess = FakeWS(), _session()
        sess.muted = True
        handled = asyncio.run(app_ws._handle_mute_command(ws, sess, "nao unmute"))
        assert handled is True
        assert sess.muted is False
        assert ws.frames_of("unmute")

    def test_ordinary_speech_is_not_consumed(self):
        ws, sess = FakeWS(), _session()
        handled = asyncio.run(
            app_ws._handle_mute_command(ws, sess, "what classes should I take"))
        assert handled is False
        assert sess.muted is False
        assert ws.sent == []

    def test_self_echo_does_not_mute(self):
        # NAO hears itself through the mic left open during TTS.
        ws, sess = FakeWS(), _session()
        sess.speaking_text = "Okay, I will mute myself now."
        handled = asyncio.run(app_ws._handle_mute_command(ws, sess, "mute"))
        # Consumed (not treated as user speech) but must NOT mute.
        assert handled is True
        assert sess.muted is False


class TestSuppression:
    def test_audio_flows_when_unmuted(self):
        ws, sess = FakeWS(), _session()
        sent = asyncio.run(app_ws._send_audio_chunk(ws, sess, _mp3_frame()))
        assert sent is True
        # One audio frame (plus the `turn_state: speaking` eye cue).
        audio = [f for f in ws.sent
                 if isinstance(f, dict) and f.get("type") == "audio_chunk"]
        assert len(audio) == 1

    def test_audio_suppressed_when_muted(self):
        ws, sess = FakeWS(), _session()
        sess.muted = True
        sent = asyncio.run(app_ws._send_audio_chunk(ws, sess, _mp3_frame()))
        assert sent is False
        assert ws.sent == []

    def test_crisis_reply_speaks_through_mute(self):
        # A spoken "mute" must not be able to silence a 988 referral.
        ws, sess = FakeWS(), _session()
        sess.muted = True
        sent = asyncio.run(
            app_ws._send_audio_chunk(ws, sess, _mp3_frame("988"), force=True))
        assert sent is True
        assert len(ws.sent) == 1

    def test_speaking_text_tracked_for_echo_guard(self):
        ws, sess = FakeWS(), _session()
        asyncio.run(app_ws._send_audio_chunk(ws, sess, _mp3_frame("I am NAO")))
        assert sess.speaking_text == "I am NAO"


class TestMuteListener:
    def test_disabled_when_already_muted(self):
        # While muted NAO is silent, so "unmute" arrives on the ordinary
        # transcript path -- the side channel must not also buffer audio.
        ws, sess = FakeWS(), _session()
        sess.muted = True
        asyncio.run(app_ws._feed_mute_listener(ws, sess, b"\x00" * 1000))
        assert len(sess.mute_buf) == 0

    def test_buffers_until_window_reached(self):
        ws, sess = FakeWS(), _session()
        asyncio.run(app_ws._feed_mute_listener(ws, sess, b"\x00" * 1000))
        assert len(sess.mute_buf) == 1000  # below window, nothing fired

    def test_buffer_is_capped(self):
        ws, sess = FakeWS(), _session()
        sess._mute_check_running = True  # block the flush path
        asyncio.run(app_ws._feed_mute_listener(
            ws, sess, b"\x00" * (app_ws._MUTE_BUF_MAX_BYTES + 5000)))
        assert len(sess.mute_buf) <= app_ws._MUTE_BUF_MAX_BYTES


class TestUsernameRebindKeepsEchoState:
    """A session rename must not orphan what Nao has already said.

    Real incident, 2026-09-04: the camera announcement was stored under
    `guest`, face identity rebound the session to a learned name, and the
    echoed announcement -- which contains the words "stop watching me" --
    then matched the `disable_camera` trigger. Nao turned its own camera
    off by hearing itself.
    """

    ANNOUNCE = ("Heads up, my camera is on for this conversation. "
                "Say stop watching me anytime.")

    def test_echo_history_follows_the_rename(self):
        sess = _session("guest")
        app_ws._reset_reply_chunks("guest", self.ANNOUNCE)
        assert app_ws._is_substring_or_sentence_echo("guest", self.ANNOUNCE)

        app_ws._rebind_username(sess, "Mia")

        assert sess.username == "mia"
        # The echo must still be recognised under the NEW name -- this is
        # the assertion that would have caught the camera incident.
        assert app_ws._is_substring_or_sentence_echo("mia", self.ANNOUNCE)
        assert "guest" not in app_ws._LAST_REPLY_FULL

    def test_rename_to_same_name_is_a_noop(self):
        sess = _session("mia")
        app_ws._reset_reply_chunks("mia", self.ANNOUNCE)
        app_ws._rebind_username(sess, "mia")
        assert sess.username == "mia"
        assert app_ws._is_substring_or_sentence_echo("mia", self.ANNOUNCE)

    def test_empty_rename_is_ignored(self):
        sess = _session("guest")
        app_ws._rebind_username(sess, "")
        assert sess.username == "guest"


class TestSystemLineEchoGuard:
    """Nao's own canned lines must never be treated as user speech.

    Stateless on purpose: the per-turn echo stores can be empty, stale, or
    orphaned by a rename, and on 2026-09-04 all three conspired to let the
    camera announcement through into a motion trigger.
    """

    # Verbatim from logs/server.log, 2026-09-04 16:32:34.
    REAL_ECHO = ("Heads up. My camera is on for this conversation. "
                 "Say stop watching me anytime.")

    def test_camera_announcement_echo_is_rejected(self):
        assert app_ws._is_system_line_echo(self.REAL_ECHO) is True

    def test_rejected_with_no_reply_history_at_all(self):
        # The stores are what failed last time; the guard must not need them.
        app_ws._LAST_REPLY_FULL.pop("nobody", None)
        app_ws._LAST_REPLY_CHUNKS.pop("nobody", None)
        assert app_ws._is_system_line_echo(self.REAL_ECHO) is True

    def test_short_real_command_still_gets_through(self):
        # The announcement tells people to say this. If the guard ate it,
        # the feature it is advertising would stop working.
        assert app_ws._is_system_line_echo("stop watching me") is False
        assert app_ws._is_system_line_echo("camera off") is False
        assert app_ws._is_system_line_echo("turn the camera on") is False

    def test_ordinary_speech_is_untouched(self):
        for t in ("tell me about the computer science program at morgan",
                  "i am feeling really anxious about my exam tomorrow",
                  "what time is my next class on tuesday morning"):
            assert app_ws._is_system_line_echo(t) is False, t

    def test_empty_and_tiny_inputs_are_safe(self):
        for t in ("", "   ", "hi", "mute", "nao mute"):
            assert app_ws._is_system_line_echo(t) is False


class TestListenerDoesNotBlockTheEventLoop:
    """The mute listener must do all blocking work on a worker thread.

    Shipped 2026-09-04 with `_write_pcm_to_wav` and `has_voice` called
    straight from the coroutine, so every listener window stalled the whole
    server -- WebSocket reads included. On an idle Pi (load 0.9 / 4 cores,
    40 C, no throttling) the purely-CPU `vad` phase went from ~11-56 ms to
    p50 230 ms, max 3021 ms.
    """

    def test_all_blocking_work_is_offloaded(self):
        import inspect
        src = inspect.getsource(app_ws._feed_mute_listener)
        check = src[src.index("async def _check"):]
        for blocking in ("_write_pcm_to_wav(", "has_voice("):
            assert blocking not in check, (
                "%s runs on the event loop inside _check; it belongs in the "
                "to_thread helper" % blocking
            )

    def test_helper_does_the_blocking_work(self):
        import inspect
        src = inspect.getsource(app_ws._feed_mute_listener)
        helper = src[src.index("def _transcribe_window"):src.index("async def _check")]
        assert "_write_pcm_to_wav(" in helper
        assert "has_voice(" in helper
        assert "transcribe(" in helper


class TestTempWavLocation:
    """Per-turn WAVs should avoid the SD card where tmpfs exists.

    Both the turn path and the mute listener write a WAV per window; on the
    Pi `/tmp` is the SD card, and the concurrent small writes contended.
    Also spares the card write cycles -- corruption has downed this Pi.
    """

    def test_prefers_tmpfs_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_ws.os.path, "isdir",
                            lambda p: p == "/dev/shm")
        monkeypatch.setattr(app_ws.os, "access", lambda p, m: p == "/dev/shm")
        assert app_ws._pick_tmp_dir() == "/dev/shm"

    def test_falls_back_when_no_tmpfs(self, monkeypatch):
        # The Mac has no /dev/shm; must return None so tempfile uses its
        # own default rather than raising.
        monkeypatch.setattr(app_ws.os.path, "isdir", lambda p: False)
        assert app_ws._pick_tmp_dir() is None

    def test_write_still_produces_a_readable_wav(self):
        import wave
        path = app_ws._write_pcm_to_wav(b"\x00\x01" * 8000)
        try:
            with wave.open(path, "rb") as w:
                assert w.getnchannels() == 1
                assert w.getframerate() == app_ws._WS_AUDIO_SR
                assert w.getnframes() == 8000
        finally:
            app_ws.os.unlink(path)


class TestSpokenConfirmation:
    """Requested 2026-09-30: say "Okay, my lips are closed now." on mute and
    "I'm back." on unmute, so the user knows the command landed."""

    def _spoken(self, ws):
        return [f["text"] for f in ws.sent
                if isinstance(f, dict) and f.get("type") == "audio_chunk"]

    def test_mute_says_lips_closed_even_though_muted(self):
        ws, sess = FakeWS(), _session()
        asyncio.run(app_ws._handle_mute_command(ws, sess, "Nao, be quiet"))
        assert sess.muted is True
        assert self._spoken(ws) == [app_ws.MUTE_ACK]
        assert app_ws.MUTE_ACK == "Okay, my lips are closed now."

    def test_unmute_says_im_back(self):
        ws, sess = FakeWS(), _session()
        sess.muted = True
        asyncio.run(app_ws._handle_mute_command(ws, sess, "you can talk"))
        assert sess.muted is False
        assert self._spoken(ws) == ["I'm back."]

    def test_no_confirmation_when_nothing_changed(self):
        ws, sess = FakeWS(), _session()
        sess.muted = True
        asyncio.run(app_ws._handle_mute_command(ws, sess, "mute"))
        assert self._spoken(ws) == []

    def test_confirmation_is_not_heard_as_user_speech(self):
        ws, sess = FakeWS(), _session()
        asyncio.run(app_ws._handle_mute_command(ws, sess, "be quiet"))
        echo = "Okay, my lips are closed now."
        assert (app_ws._is_system_line_echo(echo)
                or app_ws._is_substring_or_sentence_echo(sess.username, echo))

    def test_mute_state_changes_before_speaking(self):
        """The control frame must reach the robot before the ack audio, so
        the robot's player stop does not cut the confirmation off."""
        ws, sess = FakeWS(), _session()
        asyncio.run(app_ws._handle_mute_command(ws, sess, "be quiet"))
        kinds = [f.get("subtype") or f.get("type") for f in ws.sent
                 if isinstance(f, dict)]
        assert kinds.index("mute") < kinds.index("audio_chunk")

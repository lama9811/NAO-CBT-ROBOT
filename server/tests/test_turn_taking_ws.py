# -*- coding: utf-8 -*-
"""app_ws wiring for the conversation-feel features.

Spoken repair, therapy-lane end-of-utterance, turn-state LEDs, goodbye
close and the idle check-in. `pytest-asyncio` is not installed, so every
coroutine is driven with `asyncio.run()`.
"""
import asyncio
import json
import time

import pytest

from server import app_ws, conversation_state, session, turn_taking


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_text(self, text):
        try:
            self.sent.append(json.loads(text))
        except (TypeError, ValueError):
            self.sent.append(text)

    def frames_of(self, subtype):
        return [f for f in self.sent
                if isinstance(f, dict) and f.get("subtype") == subtype]

    def audio(self):
        return [f for f in self.sent
                if isinstance(f, dict) and f.get("type") == "audio_chunk"]

    def turn_states(self):
        return [f["data"]["state"] for f in self.frames_of("turn_state")]


class FakeSilero:
    def __init__(self, silence_ms=0, speaking=False):
        self.silence_ms = silence_ms
        self.speaking = speaking

    def silence_duration_ms(self):
        return self.silence_ms

    def is_speech_now(self):
        return self.speaking


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(app_ws, "_synth_for", lambda *a, **k: b"\x01mp3")
    monkeypatch.setattr(app_ws, "TURN_LEDS", True)
    app_ws._LAST_REPAIR_MS.clear()
    app_ws._LAST_REPAIR_LINE.clear()
    conversation_state._reset_for_tests()
    yield
    conversation_state._reset_for_tests()


def _session(username="tt_tester"):
    return app_ws._Session(username)


# ───────── spoken repair ─────────

class TestRepair:
    def test_rejected_turn_gets_one_spoken_repair(self):
        ws, sess = FakeWS(), _session()
        spoke = asyncio.run(app_ws._maybe_speak_repair(
            ws, sess, "no_voice", 1500.0, True))
        assert spoke is True
        audio = ws.audio()
        assert len(audio) == 1
        assert audio[0]["text"] in turn_taking.REPAIR_LINES
        # Ends like any reply so the robot reopens its mic.
        assert ws.frames_of("tts_ended")
        assert ws.turn_states() == ["speaking", "listening"]

    def test_not_twice_in_a_row(self):
        ws, sess = FakeWS(), _session()
        asyncio.run(app_ws._maybe_speak_repair(ws, sess, "no_voice", 1500.0))
        sess.tts_active_until_ms = 0.0  # cooldown over
        again = asyncio.run(app_ws._maybe_speak_repair(
            ws, sess, "no_voice", 1500.0))
        assert again is False
        assert len(ws.audio()) == 1

    def test_rate_limit_survives_a_reconnect(self):
        ws = FakeWS()
        first = _session("tt_reconnect")
        asyncio.run(app_ws._maybe_speak_repair(ws, first, "no_voice", 1500.0))
        second = _session("tt_reconnect")  # fresh WS connection, same user
        assert asyncio.run(app_ws._maybe_speak_repair(
            ws, second, "no_voice", 1500.0)) is False

    def test_rearmed_by_an_accepted_turn_but_still_rate_limited(self):
        ws, sess = FakeWS(), _session()
        asyncio.run(app_ws._maybe_speak_repair(ws, sess, "no_voice", 1500.0))
        app_ws._mark_turn_accepted(sess)
        assert sess.repair_armed is True
        sess.tts_active_until_ms = 0.0
        assert asyncio.run(app_ws._maybe_speak_repair(
            ws, sess, "no_voice", 1500.0)) is False  # interval not passed

    def test_never_when_muted(self):
        ws, sess = FakeWS(), _session()
        sess.muted = True
        assert asyncio.run(app_ws._maybe_speak_repair(
            ws, sess, "no_voice", 1500.0)) is False
        assert ws.sent == []

    def test_never_during_nao_speech(self):
        ws, sess = FakeWS(), _session()
        sess.tts_active_until_ms = time.time() * 1000.0 + 5000.0
        assert asyncio.run(app_ws._maybe_speak_repair(
            ws, sess, "no_voice", 1500.0)) is False
        assert ws.audio() == []

    def test_echo_of_the_repair_line_is_dropped_but_a_real_request_is_not(self):
        ws, sess = FakeWS(), _session()
        asyncio.run(app_ws._maybe_speak_repair(ws, sess, "no_voice", 1500.0))
        said = ws.audio()[0]["text"]
        assert app_ws._is_recent_turn_line_echo(sess, said) is True
        # Once the echo window has passed the same words are the user's.
        for line in list(sess.turn_lines_said):
            sess.turn_lines_said[line] -= (
                app_ws.TURN_LINE_ECHO_WINDOW_S * 1000.0 + 1.0)
        assert app_ws._is_recent_turn_line_echo(sess, said) is False


# ───────── therapy end-of-utterance ─────────

class TestTherapyEou:
    def _arbiter(self, sess):
        return asyncio.run(app_ws._should_finalize_turn(
            sess, transcript_so_far=None, now_ms=time.time() * 1000.0))

    def test_normal_lane_finalizes_on_the_robot_hint(self, monkeypatch):
        monkeypatch.setattr(app_ws, "_support_lane", lambda u: None)
        sess = _session()
        sess.silero = FakeSilero(silence_ms=200)
        sess.had_speech = True
        sess.robot_eou_hint = True
        assert self._arbiter(sess) is True

    def test_therapy_lane_waits_for_the_longer_silence(self, monkeypatch):
        monkeypatch.setattr(app_ws, "_support_lane", lambda u: "therapist")
        sess = _session()
        sess.had_speech = True
        sess.robot_eou_hint = True
        sess.silero = FakeSilero(silence_ms=600)
        assert self._arbiter(sess) is False
        sess.silero = FakeSilero(
            silence_ms=turn_taking.THERAPY_EOU_SILENCE_MS + 10)
        assert self._arbiter(sess) is True

    def test_therapy_lane_skips_the_semantic_shortcut(self, monkeypatch):
        monkeypatch.setattr(app_ws, "_support_lane", lambda u: "therapist")
        called = []

        async def _semantic(text):
            called.append(text)
            return True

        monkeypatch.setattr(app_ws, "_maybe_run_semantic_endpoint", _semantic)
        sess = _session()
        sess.had_speech = True
        sess.silero = FakeSilero(silence_ms=300)
        assert asyncio.run(app_ws._should_finalize_turn(
            sess, transcript_so_far="I just feel like",
            now_ms=time.time() * 1000.0)) is False
        assert called == []

    def test_hint_does_not_force_in_therapy_lane(self, monkeypatch):
        monkeypatch.setattr(app_ws, "_support_lane", lambda u: "therapist")
        processed = []

        async def _fake_process(ws, sess):
            processed.append(True)
            sess.reset_turn()

        monkeypatch.setattr(app_ws, "_process_turn", _fake_process)

        async def _go():
            ws, sess = FakeWS(), _session()
            sess.silero = FakeSilero(silence_ms=300)
            sess.had_speech = True
            sess.audio_buf.extend(b"\x00" * 3200)
            await app_ws._ingest_control(ws, sess, {
                "type": "control", "subtype": "end_of_utterance",
                "data": {"robot_eou_hint": True}})
            deferred = list(processed)
            task = sess._eou_fallback_task
            task.cancel()
            return deferred, task

        deferred, task = asyncio.run(_go())
        assert deferred == []
        assert task is not None

    def test_hint_still_forces_in_normal_lane(self, monkeypatch):
        monkeypatch.setattr(app_ws, "_support_lane", lambda u: None)
        processed = []

        async def _fake_process(ws, sess):
            processed.append(True)
            sess.reset_turn()

        monkeypatch.setattr(app_ws, "_process_turn", _fake_process)

        async def _go():
            ws, sess = FakeWS(), _session()
            sess.silero = FakeSilero(silence_ms=0, speaking=True)
            sess.audio_buf.extend(b"\x00" * 3200)
            await app_ws._ingest_control(ws, sess, {
                "type": "control", "subtype": "end_of_utterance",
                "data": {"robot_eou_hint": True}})

        asyncio.run(_go())
        assert processed == [True]

    def test_fallback_forces_when_the_robot_stops_streaming(self, monkeypatch):
        processed = []

        async def _fake_finalize(ws, sess, force=False):
            processed.append(force)
            return True

        monkeypatch.setattr(app_ws, "_finalize_turn_if_ready", _fake_finalize)
        sess = _session()
        sess.audio_buf.extend(b"\x00" * 3200)
        sess.robot_eou_hint = True
        sess.last_chunk_ms = 0.0  # nothing for ages
        sess.silero = FakeSilero(silence_ms=2000)
        asyncio.run(app_ws._therapy_eou_fallback(FakeWS(), sess, 0.0))
        assert processed == [True]


# ───────── turn-state LEDs ─────────

class TestTurnStateLeds:
    def test_dedup_and_disable(self, monkeypatch):
        ws, sess = FakeWS(), _session()
        asyncio.run(app_ws._send_turn_state(ws, sess, "thinking"))
        asyncio.run(app_ws._send_turn_state(ws, sess, "thinking"))
        assert ws.turn_states() == ["thinking"]
        monkeypatch.setattr(app_ws, "TURN_LEDS", False)
        asyncio.run(app_ws._send_turn_state(ws, sess, "speaking"))
        assert ws.turn_states() == ["thinking"]

    def test_first_audio_chunk_sets_speaking(self):
        ws, sess = FakeWS(), _session()
        frame = app_ws._audio_chunk_frame(1, "hi", b"\x00")
        asyncio.run(app_ws._send_audio_chunk(ws, sess, frame))
        asyncio.run(app_ws._send_audio_chunk(ws, sess, frame))
        assert ws.turn_states() == ["speaking"]
        # LED frame lands before the audio it describes.
        assert ws.sent[0]["subtype"] == "turn_state"

    def test_crisis_reply_keeps_its_white_eyes(self):
        ws, sess = FakeWS(), _session()
        frame = app_ws._audio_chunk_frame(1, "988", b"\x00")
        asyncio.run(app_ws._send_audio_chunk(ws, sess, frame, force=True))
        assert ws.turn_states() == []

    def test_listening_after_a_reply_task(self):
        ws, sess = FakeWS(), _session()

        async def _reply():
            await app_ws._send_turn_state(ws, sess, "speaking")

        asyncio.run(app_ws._with_listening_after(ws, sess, _reply()))
        assert ws.turn_states() == ["speaking", "listening"]


# ───────── slow speech ─────────

def test_tts_speed(monkeypatch):
    monkeypatch.setattr(app_ws, "TTS_SLOW_SPEED", 0.85)
    assert app_ws._tts_speed_for(False, "chat") is None
    assert app_ws._tts_speed_for(True, "chat") == 0.85
    assert app_ws._tts_speed_for(False, "grounding_coach") == 0.85
    monkeypatch.setattr(app_ws, "TTS_SLOW_SPEED", 1.0)
    assert app_ws._tts_speed_for(True, "grounding_coach") is None


# ───────── goodbye ─────────

class TestGoodbye:
    def test_goodbye_speaks_and_announces_close(self, monkeypatch):
        monkeypatch.setattr(app_ws, "_support_lane", lambda u: None)

        async def _go():
            ws, sess = FakeWS(), _session("tt_bye")
            await app_ws._emit_goodbye(ws, sess, "okay bye", {})
            await asyncio.gather(*list(app_ws._BACKGROUND_TASKS))
            return ws

        ws = asyncio.run(_go())
        assert [f["text"] for f in ws.audio()] == [turn_taking.GOODBYE_REPLY]
        assert ws.frames_of("conversation_closed")

    def test_support_goodbye_finalizes_recap_and_clears_state(self, monkeypatch):
        calls = []
        monkeypatch.setattr(app_ws._emotion_module, "finalize_session_recap",
                            lambda username, **kw: calls.append(username),
                            raising=False)
        state = conversation_state.state_for("tt_support")
        conversation_state.set_lane(state, "therapist")
        status = asyncio.run(app_ws._close_conversation(
            "tt_support", "therapist"))
        assert status == "ok"
        assert calls == ["tt_support"]
        assert conversation_state.active_lane(
            conversation_state.state_for("tt_support")) is None

    def test_recap_skipped_when_function_absent(self, monkeypatch):
        monkeypatch.delattr(app_ws._emotion_module, "finalize_session_recap",
                            raising=False)
        assert asyncio.run(app_ws._close_conversation(
            "tt_support2", "therapist")) == "unavailable"

    def test_no_recap_outside_support_lane(self, monkeypatch):
        calls = []
        monkeypatch.setattr(app_ws._emotion_module, "finalize_session_recap",
                            lambda username, **kw: calls.append(username),
                            raising=False)
        assert asyncio.run(app_ws._close_conversation(
            "tt_chat", None)) == "skipped"
        assert calls == []

    def test_anonymous_goodbye_retires_the_guest_epoch(self):
        first = session.session_key_for("guest")
        asyncio.run(app_ws._close_conversation("guest", None))
        assert session.live_anonymous_key() is None
        assert session.session_key_for("guest") != first


# ───────── idle check-in ─────────

class TestIdleCheckin:
    def _quiet(self, sess):
        long_ago = time.time() * 1000.0 - (
            turn_taking.IDLE_CHECKIN_S * 1000.0 + 5000.0)
        sess.last_user_speech_ms = long_ago
        sess.tts_active_until_ms = long_ago

    def test_not_before_anyone_has_spoken(self):
        sess = _session()
        self._quiet(sess)
        assert app_ws._claim_idle_checkin(sess) is False

    def test_once_per_silence(self):
        sess = _session()
        app_ws._mark_turn_accepted(sess)
        self._quiet(sess)
        assert app_ws._claim_idle_checkin(sess) is True
        assert app_ws._claim_idle_checkin(sess) is False
        # The user speaks again -> a later silence may check in again.
        app_ws._mark_turn_accepted(sess)
        self._quiet(sess)
        assert app_ws._claim_idle_checkin(sess) is True

    def test_not_while_muted_or_mid_utterance(self):
        sess = _session()
        app_ws._mark_turn_accepted(sess)
        self._quiet(sess)
        sess.muted = True
        assert app_ws._claim_idle_checkin(sess) is False
        sess.muted = False
        sess.had_speech = True
        assert app_ws._claim_idle_checkin(sess) is False

    def test_spoken_line(self):
        ws, sess = FakeWS(), _session()
        app_ws._mark_turn_accepted(sess)
        self._quiet(sess)
        assert asyncio.run(app_ws._maybe_idle_checkin(ws, sess)) is True
        assert [f["text"] for f in ws.audio()] == [
            turn_taking.IDLE_CHECKIN_LINE]

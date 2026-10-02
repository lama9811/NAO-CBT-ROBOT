# -*- coding: utf-8 -*-
"""Pure turn-taking policies (server/turn_taking.py) and breathing LED cues.

No WebSocket here -- `test_turn_taking_ws.py` covers the app_ws wiring.
"""
import pytest

from server import breathing_pacing, turn_taking


# ───────── spoken repair ─────────

def _allowed(**overrides):
    kw = dict(reason="no_voice", clip_ms=1500.0, muted=False, engaged=True,
              armed=True, last_repair_ms=0.0, now_ms=1_000_000.0)
    kw.update(overrides)
    return turn_taking.repair_allowed(**kw)


class TestRepairAllowed:
    def test_plain_rejection_gets_a_repair(self):
        assert _allowed() is True

    @pytest.mark.parametrize("reason", [
        "hallucination_or_noise", "non_english", "silero_no_speech",
        "empty_transcript"])
    def test_lost_speech_reasons(self, reason):
        assert _allowed(reason=reason) is True

    @pytest.mark.parametrize("reason", [
        "self_echo", "robot_named_echo", "system_line_echo", "invalid_audio",
        "mute_command", "wait_more_audio", None])
    def test_echo_and_transport_reasons_never_repair(self, reason):
        assert _allowed(reason=reason) is False

    def test_never_when_muted(self):
        assert _allowed(muted=True) is False

    def test_never_while_nao_is_speaking(self):
        assert _allowed(nao_speaking=True) is False

    def test_never_twice_in_a_row(self):
        assert _allowed(armed=False) is False

    def test_rate_limited(self):
        now = 1_000_000.0
        recent = now - (turn_taking.REPAIR_MIN_INTERVAL_S * 1000.0) + 1000.0
        assert _allowed(last_repair_ms=recent, now_ms=now) is False
        old = now - (turn_taking.REPAIR_MIN_INTERVAL_S * 1000.0) - 1.0
        assert _allowed(last_repair_ms=old, now_ms=now) is True

    def test_short_click_is_not_a_person(self):
        assert _allowed(clip_ms=turn_taking.REPAIR_MIN_CLIP_MS - 1) is False

    def test_noise_transcript_without_detected_speech_is_noise(self):
        assert _allowed(reason="hallucination_or_noise",
                        speech_confirmed=False) is False
        assert _allowed(reason="hallucination_or_noise",
                        speech_confirmed=True) is True
        # VAD unavailable: don't hold it against the user.
        assert _allowed(reason="hallucination_or_noise",
                        speech_confirmed=None) is True

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setattr(turn_taking, "REPAIR_ENABLED", False)
        assert _allowed() is False


def test_repair_lines_vary_and_never_repeat():
    assert len(turn_taking.REPAIR_LINES) >= 2
    prev = None
    for _ in range(30):
        line = turn_taking.pick_repair_line(prev)
        assert line in turn_taking.REPAIR_LINES
        assert line != prev
        prev = line


# ───────── therapy end-of-utterance ─────────

def test_therapy_lane_waits_longer():
    assert turn_taking.THERAPY_EOU_SILENCE_MS >= 1000
    assert turn_taking.eou_silence_ms(500, None) == 500
    assert turn_taking.eou_silence_ms(500, "therapist") == \
        turn_taking.THERAPY_EOU_SILENCE_MS
    # Never shortens a longer default.
    assert turn_taking.eou_silence_ms(5000, "cbt_coach") == 5000


# ───────── goodbye ─────────

@pytest.mark.parametrize("text", [
    "Bye!", "Goodbye.", "bye bye", "Okay, thank you. Bye!",
    "See you later", "Thanks, see you tomorrow.", "I have to go now.",
    "I gotta go", "Take care, NAO.", "Have a good day!",
    "That's all, thanks.", "Alright, I'm heading out now",
    "I am leaving now", "Good night",
])
def test_goodbye_recognised(text):
    assert turn_taking.is_goodbye(text) is True


@pytest.mark.parametrize("text", [
    "", "hello", "I never got to say goodbye to my grandmother.",
    "How do I say goodbye to someone I love?",
    "That's all", "that's all I can think of",
    "I want to go to the library", "Can you take care of my plant",
    "see what I mean", "bye the way I failed my exam",
])
def test_goodbye_not_recognised(text):
    assert turn_taking.is_goodbye(text) is False


def test_goodbye_reply_mentions_support_in_therapy_lane():
    assert "988" in turn_taking.goodbye_reply("therapist")
    assert "988" not in turn_taking.goodbye_reply(None)


# ───────── idle check-in ─────────

def _due(**overrides):
    kw = dict(now_ms=100_000.0, last_activity_ms=0.0, muted=False,
              already_done=False, busy=False)
    kw.update(overrides)
    return turn_taking.idle_checkin_due(**kw)


def test_idle_checkin_after_long_silence_only():
    assert _due() is True
    assert _due(last_activity_ms=100_000.0 - 1000.0) is False


@pytest.mark.parametrize("flag", ["muted", "already_done", "busy"])
def test_idle_checkin_blocked(flag):
    assert _due(**{flag: True}) is False


# ───────── breathing → LED cues ─────────

def test_paced_breath_gets_phase_cues():
    chunks = breathing_pacing.expand_tts_pacing(
        'Breathe in slowly one<break time="800ms"/>two'
        '<break time="800ms"/>three<break time="800ms"/>four'
        '<break time="800ms"/>and hold.')
    cues = breathing_pacing.breath_phase_cues(chunks)
    assert len(cues) == len(chunks)
    assert cues[0]["phase"] == "inhale"
    # The inhale lasts the whole count: four beats of speech + pauses.
    assert cues[0]["seconds"] >= 3.0
    assert cues[-1]["phase"] == "hold"
    assert all(c is None for c in cues[1:-1])


def test_exhale_cue():
    chunks = breathing_pacing.expand_tts_pacing("Breathe out: 1 2 3 4 5 6.")
    cues = breathing_pacing.breath_phase_cues(chunks)
    assert cues[0]["phase"] == "exhale"
    assert breathing_pacing.MIN_PHASE_S <= cues[0]["seconds"] \
        <= breathing_pacing.MAX_PHASE_S


def test_ordinary_sentence_never_drives_the_eyes():
    chunks = breathing_pacing.expand_tts_pacing(
        "Hold on, let me think about that for a second.")
    assert breathing_pacing.is_paced(chunks) is False
    assert breathing_pacing.breath_phase_cues(chunks) == [None]

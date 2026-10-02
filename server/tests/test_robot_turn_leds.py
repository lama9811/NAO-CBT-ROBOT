"""Robot side of the turn-taking eye colours (``nao/ws_client.py``).

The server sends ``turn_state`` (listening / thinking / speaking) and
``led_breath`` controls; the robot maps them onto ``leds.LedDriver``.
Runs the Python 2.7 robot module under Python 3 with fakes for the LED
driver and TTS player -- no naoqi needed.
"""
from __future__ import annotations

import sys
import time

import pytest


def _load():
    if "nao" not in sys.path:
        sys.path.insert(0, "nao")
    return pytest.importorskip("ws_client")


class FakeLeds:
    def __init__(self):
        self.calls = []

    def set_thinking(self):
        self.calls.append(("set_thinking",))

    def set_speaking(self):
        self.calls.append(("set_speaking",))

    def set_listening(self):
        self.calls.append(("set_listening",))

    def breathe(self, phase, seconds):
        self.calls.append(("breathe", phase, seconds))


class FakePlayer:
    def __init__(self):
        self.playing = False
        self.jobs = []

    def is_playing(self):
        return self.playing

    def enqueue(self, text, mp3_bytes, pause_after_ms=0, on_start=None):
        self.jobs.append((text, pause_after_ms, on_start))


class OldPlayer(FakePlayer):
    def enqueue(self, text, mp3_bytes, pause_after_ms=0):
        self.jobs.append((text, pause_after_ms, None))


def _client(player=None):
    ws_client = _load()
    cli = ws_client.NaoWsClient(
        server_url="ws://127.0.0.1:1/ws/guest", username="guest",
        shared_secret="", audio_streamer=None,
        tts_player=player if player is not None else FakePlayer(),
        action_dispatcher=None, brain_cache=None)
    cli.leds = FakeLeds()
    # Body language is irrelevant here and touches naoqi.
    cli._kick_stand_up = lambda: None
    cli._start_speaking_gestures = lambda: None
    cli._close_mic_gate_for_tts = lambda: None
    return cli


def _control(sub, **data):
    return {"type": "control", "subtype": sub, "data": data}


def _wait_for(pred, timeout=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_thinking_and_speaking_colours():
    cli = _client()
    cli._handle_control(_control("turn_state", state="thinking"))
    cli._handle_control(_control("turn_state", state="speaking"))
    assert cli.leds.calls == [("set_thinking",), ("set_speaking",)]


def test_listening_waits_for_playback_to_drain():
    player = FakePlayer()
    player.playing = True
    cli = _client(player)
    cli._handle_control(_control("turn_state", state="listening"))
    time.sleep(0.3)
    assert ("set_listening",) not in cli.leds.calls
    player.playing = False
    assert _wait_for(lambda: ("set_listening",) in cli.leds.calls)


def test_stale_listening_is_superseded():
    player = FakePlayer()
    player.playing = True
    cli = _client(player)
    cli._handle_control(_control("turn_state", state="listening"))
    cli._handle_control(_control("turn_state", state="thinking"))
    player.playing = False
    time.sleep(0.4)
    assert ("set_listening",) not in cli.leds.calls


def test_unknown_controls_are_harmless():
    cli = _client()
    cli._handle_control(_control("turn_state", state="dancing"))
    cli._handle_control(_control("conversation_closed", reason="goodbye"))
    cli._handle_control(_control("something_new"))
    assert cli.leds.calls == []


def test_no_leds_attached_is_harmless():
    cli = _client()
    cli.leds = None
    cli._handle_control(_control("turn_state", state="thinking"))
    cli._handle_control(_control("led_breath", seq=3, phase="inhale",
                                 seconds=4.0))


def test_turn_leds_can_be_switched_off():
    cli = _client()
    cli._turn_leds_enabled = False
    cli._handle_control(_control("turn_state", state="speaking"))
    assert cli.leds.calls == []


def test_breath_cue_starts_with_its_chunk():
    cli = _client()
    cli._handle_control(_control("led_breath", seq=7, phase="inhale",
                                 seconds=4.2))
    assert cli.leds.calls == []  # not on receipt
    cli._handle_audio_chunk({"type": "audio_chunk", "seq": 7,
                             "text": "Breathe in slowly one",
                             "pause_after_ms": 800, "data": "AAEC"})
    text, pause, on_start = cli.tts_player.jobs[-1]
    assert pause == 800 and on_start is not None
    on_start()  # what the player does when the chunk starts
    assert cli.leds.calls == [("breathe", "inhale", 4.2)]
    # A chunk without a cue gets no callback.
    cli._handle_audio_chunk({"type": "audio_chunk", "seq": 8, "text": "two",
                             "data": "AAEC"})
    assert cli.tts_player.jobs[-1][2] is None


def test_breath_cue_with_an_older_player():
    cli = _client(OldPlayer())
    cli._handle_control(_control("led_breath", seq=1, phase="exhale",
                                 seconds=6.0))
    cli._handle_audio_chunk({"type": "audio_chunk", "seq": 1, "text": "out",
                             "data": "AAEC"})
    assert cli.leds.calls == [("breathe", "exhale", 6.0)]
    assert len(cli.tts_player.jobs) == 1


def test_led_driver_breathe_and_thinking_exist():
    if "nao" not in sys.path:
        sys.path.insert(0, "nao")
    leds = pytest.importorskip("leds")
    drv = leds.LedDriver("127.0.0.1")  # disabled off-robot: all no-ops
    drv.set_thinking()
    drv.breathe("inhale", 4)
    drv.breathe("hold", 4)
    drv.breathe("exhale", 99)
    drv.breathe("nonsense", "x")

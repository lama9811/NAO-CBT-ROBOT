"""Boot greeting — NAO introduces itself once, then goes quiet again.

`nao/main.py:_announce_ready` speaks one line through NAOqi's own voice when
the robot finishes booting, and waves while it does. Native TTS is deliberately
pinned at volume 0.0 (launcher + `main.py:120`) because the built-in kid voice
otherwise leaks into every ElevenLabs reply as a second speaker — the dual-voice
bug. So the greeting has to unmute, speak, and re-mute, and the re-mute must
survive `say()` raising. That invariant is what most of this file guards.

`nao/` is Python 2.7 robot code and naoqi is not importable off-robot, so the
tests inject fake proxies through the `proxy_factory` seam, the same way
test_gesture.py fakes the motion stack.
"""
from __future__ import annotations

import os
import sys

import pytest

# `nao/` is a sibling package that imports its own modules flat (`import
# config`, `from utils import ...`), so it needs to be on sys.path directly.
_NAO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "nao",
)
if _NAO_DIR not in sys.path:
    sys.path.insert(0, _NAO_DIR)

main = pytest.importorskip("main", reason="nao/main.py not importable here")


class _FakeTTS:
    def __init__(self, raise_on_say: bool = False) -> None:
        self.said: list[str] = []
        self.volumes: list[float] = []
        self._raise = raise_on_say

    def setVolume(self, level):  # noqa: N802 — naoqi's casing
        self.volumes.append(float(level))

    def say(self, text):
        self.said.append(text)
        if self._raise:
            raise RuntimeError("naoqi say blew up")


class _FakeBehaviorManager:
    def __init__(self) -> None:
        self.started: list[str] = []

    def startBehavior(self, name):  # noqa: N802
        self.started.append(name)


class _FakePosture:
    def __init__(self) -> None:
        self.postures: list[str] = []

    def goToPosture(self, name, speed):  # noqa: N802
        self.postures.append(name)
        return True


class _Recorder:
    """Hands out fake proxies by naoqi service name."""

    def __init__(self, raise_on_say: bool = False) -> None:
        self.tts = _FakeTTS(raise_on_say=raise_on_say)
        self.behavior = _FakeBehaviorManager()
        self.posture = _FakePosture()
        self.requested: list[str] = []

    def __call__(self, service, ip, port):
        self.requested.append(service)
        return {
            "ALTextToSpeech": self.tts,
            "ALBehaviorManager": self.behavior,
            "ALRobotPosture": self.posture,
        }.get(service, _FakeBehaviorManager())


class _NullLog:
    def info(self, *a, **kw): pass
    def warn(self, *a, **kw): pass
    def debug(self, *a, **kw): pass
    def exception(self, *a, **kw): pass


@pytest.fixture(autouse=True)
def _reset_once_flag():
    """The greeting fires once per process; unlatch it between tests."""
    main._boot_greeting_done = False
    yield
    main._boot_greeting_done = False


def _announce(rec, monkeypatch, **env):
    for key in ("BOOT_GREETING", "BOOT_GREETING_TEXT"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return main._announce_ready("127.0.0.1", 9559, _NullLog(), proxy_factory=rec)


def test_speaks_the_default_line(monkeypatch):
    rec = _Recorder()
    _announce(rec, monkeypatch)

    assert rec.tts.said == ["Hello, I'm NAO. How can I help you?"]


def test_waves_while_greeting(monkeypatch):
    rec = _Recorder()
    _announce(rec, monkeypatch)

    assert any("Hey_" in b for b in rec.behavior.started), rec.behavior.started


def test_volume_is_restored_to_zero(monkeypatch):
    """Unmute to speak, then back to 0.0 — anything else leaks a second voice."""
    rec = _Recorder()
    _announce(rec, monkeypatch)

    assert rec.tts.volumes, "greeting never touched the volume"
    assert rec.tts.volumes[-1] == 0.0
    assert max(rec.tts.volumes) > 0.0, "never actually unmuted, so nothing was audible"


def test_volume_is_restored_even_when_say_raises(monkeypatch):
    """The re-mute lives in a finally — a naoqi blow-up must not leave it hot."""
    rec = _Recorder(raise_on_say=True)
    _announce(rec, monkeypatch)

    assert rec.tts.volumes[-1] == 0.0


def test_custom_text_is_honoured(monkeypatch):
    rec = _Recorder()
    _announce(rec, monkeypatch, BOOT_GREETING_TEXT="Good morning, I am ready.")

    assert rec.tts.said == ["Good morning, I am ready."]


def test_disabled_by_env(monkeypatch):
    rec = _Recorder()
    _announce(rec, monkeypatch, BOOT_GREETING="0")

    assert rec.tts.said == []
    assert rec.behavior.started == []
    assert rec.tts.volumes == [], "disabled greeting must not touch volume at all"


def test_fires_only_once_per_process(monkeypatch):
    """The call site sits inside main()'s crash-retry loop."""
    rec = _Recorder()
    _announce(rec, monkeypatch)
    _announce(rec, monkeypatch)
    _announce(rec, monkeypatch)

    assert rec.tts.said == ["Hello, I'm NAO. How can I help you?"]


def test_never_raises_when_naoqi_is_broken(monkeypatch):
    """A greeting must never stop the robot from booting."""
    def _explode(service, ip, port):
        raise RuntimeError("no broker in Python's world")

    # Should swallow and return, not propagate.
    main._announce_ready("127.0.0.1", 9559, _NullLog(), proxy_factory=_explode)

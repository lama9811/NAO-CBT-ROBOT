"""NAO should start a conversation on power-up without a tap.

On 2026-09-30 only 2 of 8 wakes came from a face; the rest needed a head
tap, after every restart. main.py now calls ``force_engage("boot")`` once
after start-up, so NAO's first words are the camera line and greeting.
"""
import sys
import types

import pytest


def _ws():
    for name in ("naoqi", "qi"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.ALProxy = object
            mod.ALModule = object
            sys.modules[name] = mod
    if "nao" not in sys.path:
        sys.path.insert(0, "nao")
    return pytest.importorskip("wake_state")


def _wsm(ws):
    return ws._build_wsm(ws._ScriptedFaceReader(), ws._FakeLeds())


def test_force_engage_opens_a_session_from_idle():
    ws = _ws()
    wsm, transitions = _wsm(ws)
    assert wsm.current_state() == ws.STATE_IDLE
    assert wsm.force_engage("boot") is True
    assert wsm.current_state() == ws.STATE_ENGAGED
    engaged = [t for t in transitions if t[0] == "engaged"]
    assert engaged and engaged[0][2] == "boot"


def test_force_engage_never_reopens_a_live_session():
    ws = _ws()
    wsm, transitions = _wsm(ws)
    wsm.force_engage("boot")
    assert wsm.force_engage("boot") is False
    assert len([t for t in transitions if t[0] == "engaged"]) == 1

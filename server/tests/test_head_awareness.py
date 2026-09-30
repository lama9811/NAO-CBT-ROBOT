"""NAO must look at whoever is talking, not lock onto the first face.

Covers ``nao/awareness.py``:

* ``LifeGuard`` keeps NAOqi Autonomous Life disabled. On 2026-09-30 Life
  re-enabled itself two seconds after our boot-time disable and then put
  ALBasicAwareness into FullyEngaged on one person, which is exactly the
  "only answers the first person" symptom.
* ``SpeakingGate`` / ``HeadAwareness`` pause head awareness while NAO's own
  MP3 reply plays, so the robot does not turn toward its own speaker.
"""
from __future__ import annotations

import sys

import pytest


def _load():
    if "nao" not in sys.path:
        sys.path.insert(0, "nao")
    return pytest.importorskip("awareness")


class FakeLife:
    def __init__(self, states):
        self.states = list(states)
        self.set_calls = []

    def getState(self):
        return self.states[0] if len(self.states) == 1 else self.states.pop(0)

    def setState(self, s):
        self.set_calls.append(s)


class FakeBA:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _f(*args):
            self.calls.append((name,) + args)
        return _f

    def names(self):
        return [c[0] for c in self.calls]


# ---------------------------------------------------------------- LifeGuard
def test_life_guard_redisables_when_life_escaped():
    aw = _load()
    life = FakeLife(["solitary"])
    fired = []
    g = aw.LifeGuard(life, on_redisabled=lambda was, ctx: fired.append(was))
    assert g.check_once() is True
    assert life.set_calls == ["disabled"]
    assert fired == ["solitary"]


def test_life_guard_leaves_disabled_life_alone():
    aw = _load()
    life = FakeLife(["disabled"])
    g = aw.LifeGuard(life)
    assert g.check_once() is False
    assert life.set_calls == []


def test_life_guard_tolerates_missing_proxy():
    aw = _load()
    g = aw.LifeGuard(None)
    assert g.check_once() is False
    assert g.hold_disabled_at_boot() is False


def test_boot_hold_catches_life_reenabling_itself():
    """The measured failure: disabled, then Life's own boot flips it on."""
    aw = _load()
    life = FakeLife(["disabled", "disabled", "solitary"] + ["disabled"] * 20)
    t = {"now": 0.0}
    g = aw.LifeGuard(life)
    ok = g.hold_disabled_at_boot(
        stable_s=5.0, timeout_s=60.0, poll_s=1.0,
        clock=lambda: t["now"],
        sleep=lambda s: t.__setitem__("now", t["now"] + s),
    )
    assert ok is True
    assert life.set_calls == ["disabled"]
    assert g.redisable_count == 1


def test_boot_hold_times_out_if_life_never_settles():
    aw = _load()
    life = FakeLife(["interactive"])
    t = {"now": 0.0}
    g = aw.LifeGuard(life)
    ok = g.hold_disabled_at_boot(
        stable_s=5.0, timeout_s=10.0, poll_s=1.0,
        clock=lambda: t["now"],
        sleep=lambda s: t.__setitem__("now", t["now"] + s),
    )
    assert ok is False
    assert len(life.set_calls) == 10


# ------------------------------------------------------------- SpeakingGate
def test_gate_pauses_on_speech_and_resumes_after_tail():
    aw = _load()
    g = aw.SpeakingGate(tail_s=0.8)
    assert g.step(0.0, False) is None
    assert g.step(1.0, True) == "pause"
    assert g.step(1.1, True) is None
    assert g.step(1.5, False) is None          # 0.4s after last speech
    assert g.step(1.8, False) is None          # 0.7s, tail not done
    assert g.step(1.95, False) == "resume"     # 0.85s >= 0.8 tail
    assert g.step(3.0, False) is None


def test_gate_tail_restarts_if_speech_resumes():
    """Back-to-back sentences must not let the head swing between them."""
    aw = _load()
    g = aw.SpeakingGate(tail_s=0.8)
    g.step(0.0, True)
    assert g.step(0.5, False) is None
    assert g.step(0.7, True) is None           # next sentence, still paused
    assert g.step(1.2, False) is None
    assert g.step(1.6, False) == "resume"


# ------------------------------------------------------- sound event parsing
# Captured verbatim from the robot (NAOqi 2.8.7.4), 2026-09-30.
REAL_28_EVENT = [
    [5289, 373060],
    [-2.063405752182007, 0.18170706927776337, 0.27152183651924133,
     0.05943328142166138],
    [0.0, 0.0, 0.1264999955892563, -2.9103830456733704e-11,
     0.009161950089037418, -0.03225589171051979],
    [0.278374582529068, -0.10425599664449692, 0.24873465299606323,
     0.032250769436359406, -0.32340458035469055, 0.290566623210907],
]


def test_parse_real_naoqi_28_event():
    """The old parser read azimuth as confidence and a head position (0.0)
    as the direction, so voice-turning never had a real direction."""
    aw = _load()
    e = aw.parse_sound_event(REAL_28_EVENT)
    assert e["ts"] == (5289, 373060)
    assert e["azimuth"] == pytest.approx(-2.0634, abs=1e-3)
    assert e["confidence"] == pytest.approx(0.2715, abs=1e-3)
    assert e["energy"] == pytest.approx(0.0594, abs=1e-3)
    assert e["head_yaw"] == pytest.approx(-0.0323, abs=1e-3)


def test_parse_legacy_layout_still_works():
    aw = _load()
    e = aw.parse_sound_event([[1, 2], [0.8, 0.1], [0.5, 0.1, 0, 0]])
    assert e["confidence"] == 0.8 and e["azimuth"] == 0.5


def test_parse_garbage_is_none():
    aw = _load()
    assert aw.parse_sound_event(None) is None
    assert aw.parse_sound_event([]) is None


# --------------------------------------------------------------- VoiceTurner
def _ev(ts, az_deg, conf=0.8, head_deg=0.0):
    import math
    return {"ts": (ts, 0), "azimuth": math.radians(az_deg),
            "elevation": 0.0, "confidence": conf, "energy": 0.1,
            "head_yaw": math.radians(head_deg)}


def test_turns_toward_a_confident_voice():
    import math
    aw = _load()
    t = aw.VoiceTurner(min_conf=0.4)
    target = t.decide(_ev(1, 40), now=0.0, current_yaw=0.0)
    assert math.degrees(target) == pytest.approx(40)


def test_target_is_relative_to_where_the_head_was():
    import math
    aw = _load()
    t = aw.VoiceTurner(min_conf=0.4)
    target = t.decide(_ev(1, 30, head_deg=20), now=0.0, current_yaw=0.35)
    assert math.degrees(target) == pytest.approx(50)


def test_ignores_low_confidence_repeat_and_self_speech():
    aw = _load()
    t = aw.VoiceTurner(min_conf=0.4)
    assert t.decide(_ev(1, 40, conf=0.2), now=0.0, current_yaw=0.0) is None
    assert t.decide(_ev(2, 40), now=0.0, current_yaw=0.0, ignore=True) is None
    assert t.decide(_ev(3, 40), now=0.0, current_yaw=0.0) is not None
    # Same timestamp again: localizer has not fired, no new turn.
    assert t.decide(_ev(3, -40), now=5.0, current_yaw=0.7) is None


def test_no_turn_when_already_facing_the_voice():
    import math
    aw = _load()
    t = aw.VoiceTurner(min_conf=0.4, min_delta_deg=12)
    assert t.decide(_ev(1, 5), now=0.0, current_yaw=0.0) is None
    assert t.decide(_ev(2, 45), now=0.0,
                    current_yaw=math.radians(40)) is None


def test_turns_are_rate_limited():
    aw = _load()
    t = aw.VoiceTurner(min_conf=0.4, min_interval_s=1.0, max_yaw_deg=80)
    assert t.decide(_ev(1, 40), now=0.0, current_yaw=0.0) is not None
    assert t.decide(_ev(2, -40), now=0.5, current_yaw=0.7) is None


def test_sounds_from_behind_are_ignored_not_clamped():
    """2026-09-30: a wall echo at about -118 deg was clamped to -100 and NAO
    stared at the wall for minutes."""
    aw = _load()
    t = aw.VoiceTurner(min_conf=0.1, max_yaw_deg=80)
    assert t.decide(_ev(1, -118), now=0.0, current_yaw=0.0) is None
    assert t.decide(_ev(2, 200), now=5.0, current_yaw=0.0) is None
    assert t.decide(_ev(3, 60), now=10.0, current_yaw=0.0) is not None


def test_faint_real_voice_is_accepted_by_default(monkeypatch):
    aw = _load()
    monkeypatch.delenv("VOICE_TURN_MIN_CONF", raising=False)
    t = aw.VoiceTurner()
    assert t.decide(_ev(1, 40, conf=0.2), now=0.0, current_yaw=0.0) is not None


# ------------------------------------------------------------ HeadAwareness
class FakeMemory:
    def __init__(self):
        self.event = None

    def getData(self, key):
        return self.event


class FakeMotion:
    def __init__(self):
        self.yaw = 0.0
        self.moves = []

    def getAngles(self, name, use_sensors):
        return [self.yaw]

    def angleInterpolationWithSpeed(self, names, angles, speed):
        self.moves.append((names, angles))
        self.yaw = angles[0]


def _ha(aw, speaking=None):
    mem, mot, trk, ba = FakeMemory(), FakeMotion(), FakeBA(), FakeBA()
    ha = aw.HeadAwareness(mem, mot, trk, basic_awareness=ba,
                          is_speaking=speaking,
                          turner=aw.VoiceTurner(min_conf=0.4),
                          tts_tail_s=0.5)
    return ha, mem, mot, trk, ba


def _raw(ts, az_rad, conf=0.8):
    return [[ts, 0], [az_rad, 0.0, conf, 0.1], [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]]


def test_tick_turns_head_and_rearms_face_tracking():
    aw = _load()
    ha, mem, mot, trk, _ = _ha(aw)
    mem.event = _raw(1, 0.7)
    assert ha.tick(0.0) == pytest.approx(0.7)
    assert mot.moves == [(["HeadYaw"], [0.7])]
    names = trk.names()
    assert names.index("stopTracker") < names.index("track")


def test_no_turn_toward_own_voice():
    aw = _load()
    speaking = {"v": True}
    ha, mem, mot, _, _ = _ha(aw, speaking=lambda: speaking["v"])
    mem.event = _raw(1, 0.7)
    assert ha.tick(0.0) is None
    speaking["v"] = False
    mem.event = _raw(2, 0.7)
    assert ha.tick(0.2) is None      # inside the 0.5 s tail
    mem.event = _raw(3, 0.7)
    assert ha.tick(0.9) is not None  # tail over, real voice
    assert len(mot.moves) == 1


def test_start_disables_naoqi_awareness_and_tracks_faces():
    aw = _load()
    ha, _, _, trk, ba = _ha(aw)
    assert ha.start() is True
    try:
        assert ("setEnabled", False) in ba.calls
        assert ("track", "Face") in trk.calls
    finally:
        ha.stop()
    assert "stopTracker" in trk.names()


def test_start_knocks_life_down_first():
    aw = _load()
    life = FakeLife(["interactive", "disabled"])
    ha, *_ = _ha(aw)
    ha._life_guard = aw.LifeGuard(life)
    ha.start()
    ha.stop()
    assert life.set_calls[:1] == ["disabled"]


def test_start_without_proxies_is_a_noop():
    aw = _load()
    assert aw.HeadAwareness(None, None, None).start() is False


def test_turn_calls_back_so_caller_can_reidentify():
    aw = _load()
    ha, mem, mot, trk, _ = _ha(aw)
    seen = []
    ha.on_turn = seen.append
    mem.event = _raw(1, 0.7)
    ha.tick(0.0)
    assert seen == [pytest.approx(0.7)]


class LostTracker(FakeBA):
    def __init__(self, lost=True):
        FakeBA.__init__(self)
        self.lost = lost

    def isTargetLost(self):
        return self.lost


def test_recenters_when_nobody_is_found():
    aw = _load()
    mem, mot, trk = FakeMemory(), FakeMotion(), LostTracker(lost=True)
    ha = aw.HeadAwareness(mem, mot, trk, turner=aw.VoiceTurner(min_conf=0.4),
                          tts_tail_s=0.5, recenter_s=4.0)
    mot.yaw = -1.7
    ha.tick(0.0)
    ha.tick(2.0)
    assert mot.moves == []
    ha.tick(4.5)
    assert mot.moves == [(["HeadYaw"], [0.0])]


def test_holds_while_a_face_is_tracked():
    aw = _load()
    mem, mot, trk = FakeMemory(), FakeMotion(), LostTracker(lost=False)
    ha = aw.HeadAwareness(mem, mot, trk, turner=aw.VoiceTurner(min_conf=0.4),
                          tts_tail_s=0.5, recenter_s=4.0)
    mot.yaw = 0.9
    for t in (0.0, 3.0, 6.0, 9.0):
        ha.tick(t)
    assert mot.moves == []


# ------------------------------------------------ idle posture: activity
def _ws_client():
    import types
    for name in ("naoqi", "qi"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.ALProxy = object
            mod.ALModule = object
            sys.modules[name] = mod
    return pytest.importorskip("ws_client")


class _Log:
    def __getattr__(self, level):
        return lambda *a, **k: None


def test_real_transcript_counts_as_activity_but_rejects_do_not():
    """NAO sits after a quiet spell; noise and echoes must not keep it up,
    and a real sentence must stand it back up."""
    wc = _ws_client()
    c = object.__new__(wc.NaoWsClient)
    c.log = _Log()
    c.last_activity_ts = 0.0
    c._on_echo_reject = lambda data: None
    c._track_transcript_health = lambda tx, reason: None
    c._announcer = None
    wc.NaoWsClient._handle_control(c, {"subtype": "transcript", "data": {
        "transcript": "", "reject_reason": "no_voice"}})
    assert c.last_activity_ts == 0.0
    wc.NaoWsClient._handle_control(c, {"subtype": "transcript", "data": {
        "transcript": "Technical.", "reject_reason": "self_echo"}})
    assert c.last_activity_ts == 0.0
    wc.NaoWsClient._handle_control(c, {"subtype": "transcript", "data": {
        "transcript": "What courses need COSC 112?", "reject_reason": ""}})
    assert c.last_activity_ts > 0.0

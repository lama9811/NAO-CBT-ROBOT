# -*- coding: utf-8 -*-
"""awareness.py - make NAO look at whoever is talking.

Two pieces, both Python 2.7 and both no-ops off-robot.

LifeGuard
---------
NAOqi's built-in Autonomous Life must stay disabled while our stack runs.
``main._disable_autonomous`` disables it at boot, but on NAOqi 2.8 Life
finishes its *own* boot a few seconds later ("Starting life after boot
config") and moves itself back to ``solitary``. From there it flips to
``interactive`` whenever someone walks up, and in interactive mode it sets
ALBasicAwareness to *FullyEngaged* on that one person: the robot locks onto
the first face and ignores every other voice in the room. Measured
2026-09-30: our disable landed at 18:16:05, Life re-enabled itself at
18:16:07, then cycled solitary/interactive 17 times in 12 minutes.

``LifeGuard.check_once()`` reads the state and forces it back to
``disabled``. ``hold_disabled_at_boot()`` repeats that until the state has
stayed disabled long enough to trust. Note that entering ``disabled`` makes
Life call ``ALMotion.rest()``, so a re-disable mid-session needs the caller
to re-stand the robot (``on_redisabled``).

HeadAwareness
-------------
Turns the head toward whoever speaks and holds it there, then lets
ALTracker lock onto a face in the new view. Sound direction comes from
``ALSoundLocalization``, parsed with the NAOqi 2.8 layout (see
``parse_sound_event``). NAOqi's own ALBasicAwareness is switched off: it
looked back after one second whenever its person detector found nobody,
which on this robot was always.

NAO's replies are MP3s played through ALAudioPlayer, so NAOqi does not know
the robot is speaking and would localize its own loudspeaker. Sound events
are therefore ignored while ``is_speaking()`` is true and for ``tts_tail_s``
after. The same poll thread runs the LifeGuard watchdog.
"""
from __future__ import print_function

import math
import os
import threading
import time

try:
    from naoqi import ALProxy  # noqa: F401
    _HAS_NAOQI = True
except Exception:  # pragma: no cover - dev box
    ALProxy = None
    _HAS_NAOQI = False


LIFE_DISABLED = "disabled"

_DEFAULT_STIMULI = {
    "Sound": True,
    "People": True,
    # Movement makes the head chase anyone walking past in the background.
    "Movement": False,
    # Touch already means barge-in / stop in our stack.
    "Touch": False,
}


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


class _NullLog(object):
    def _noop(self, *a, **kw):
        pass
    info = warn = debug = error = exception = _noop


def _proxy(name, ip, port):
    if not _HAS_NAOQI or ALProxy is None:
        return None
    try:
        return ALProxy(name, ip, port)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# LifeGuard
# ---------------------------------------------------------------------------
class LifeGuard(object):
    """Keep ALAutonomousLife in the ``disabled`` state."""

    def __init__(self, life_proxy, log=None, on_redisabled=None):
        self._life = life_proxy
        self._log = log or _NullLog()
        self.on_redisabled = on_redisabled
        self.redisable_count = 0

    def state(self):
        if self._life is None:
            return None
        try:
            return str(self._life.getState())
        except Exception:
            return None

    def check_once(self, context="watchdog"):
        """Force Life back to disabled if it escaped. True if we had to."""
        state = self.state()
        if state is None or state == LIFE_DISABLED:
            return False
        try:
            self._life.setState(LIFE_DISABLED)
        except Exception as exc:
            self._log.warn("autonomous_life_redisable_failed",
                           state=state, context=context, error=str(exc))
            return False
        self.redisable_count += 1
        self._log.warn("autonomous_life_redisabled",
                       was=state, context=context,
                       count=self.redisable_count)
        cb = self.on_redisabled
        if cb is not None:
            try:
                cb(state, context)
            except Exception as exc:
                self._log.debug("autonomous_life_callback_failed",
                                error=str(exc))
        return True

    def hold_disabled_at_boot(self, stable_s=10.0, timeout_s=90.0,
                              poll_s=1.0, clock=time.time, sleep=time.sleep,
                              stop_event=None):
        """Re-disable until Life has stayed disabled for ``stable_s``.

        Returns True once stable, False on timeout or stop.
        """
        if self._life is None:
            return False
        start = clock()
        stable_since = None
        while clock() - start < timeout_s:
            if stop_event is not None and stop_event.is_set():
                return False
            if self.check_once(context="boot"):
                stable_since = None
            else:
                now = clock()
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= stable_s:
                    self._log.info("autonomous_life_stable_disabled",
                                   redisabled=self.redisable_count,
                                   waited_s=round(now - start, 1))
                    return True
            sleep(poll_s)
        self._log.warn("autonomous_life_boot_guard_timeout",
                       redisabled=self.redisable_count)
        return False

    def start_boot_thread(self, **kw):
        t = threading.Thread(target=self.hold_disabled_at_boot, kwargs=kw,
                             name="nao-life-guard")
        t.daemon = True
        t.start()
        return t


# ---------------------------------------------------------------------------
# Speaking gate (pure logic, unit-tested)
# ---------------------------------------------------------------------------
class SpeakingGate(object):
    """Decide when to pause/resume awareness around NAO's own speech.

    ``step(now, speaking)`` returns "pause", "resume", or None.
    """

    def __init__(self, tail_s=0.8):
        self.tail_s = float(tail_s)
        self.paused = False
        self._last_speaking_at = None

    def step(self, now, speaking):
        if speaking:
            self._last_speaking_at = now
            if not self.paused:
                self.paused = True
                return "pause"
            return None
        if self.paused:
            last = self._last_speaking_at
            if last is None or now - last >= self.tail_s:
                self.paused = False
                return "resume"
        return None


# ---------------------------------------------------------------------------
# Sound events (NAOqi 2.8 layout)
# ---------------------------------------------------------------------------
def parse_sound_event(event):
    """Parse ``ALSoundLocalization/SoundLocated`` into a dict, or None.

    NAOqi 2.8 (measured on this robot, 2026-09-30) stores::

        [[sec, usec],
         [azimuth, elevation, confidence, energy],
         [head 6D in FRAME_TORSO: x, y, z, wx, wy, wz],
         [head 6D in FRAME_ROBOT]]

    Azimuth is relative to where the head pointed when the sound arrived,
    so the absolute head yaw to face the source is ``head_wz + azimuth``.

    ``sound_localize.py`` assumed an older ``[[ts], [conf, energy],
    [az, el, ...]]`` layout. On this firmware that read the head's x
    position (always 0.0) as the direction and the azimuth as the
    confidence, which is why voice-turning never worked here.
    """
    try:
        ts = event[0]
        a = event[1]
        if len(a) >= 4:
            head = event[2] if len(event) > 2 else []
            head_yaw = float(head[5]) if len(head) >= 6 else 0.0
            return {
                "ts": (ts[0], ts[1]),
                "azimuth": float(a[0]),
                "elevation": float(a[1]),
                "confidence": float(a[2]),
                "energy": float(a[3]),
                "head_yaw": head_yaw,
            }
        # Legacy layout, kept so older firmware still parses.
        geom = event[2]
        return {
            "ts": (ts[0], ts[1]),
            "azimuth": float(geom[0]),
            "elevation": float(geom[1]),
            "confidence": float(a[0]),
            "energy": float(a[1]) if len(a) > 1 else 0.0,
            "head_yaw": 0.0,
        }
    except (IndexError, TypeError, ValueError):
        return None


class VoiceTurner(object):
    """Decide whether a sound event should turn the head, and where to.

    Pure logic so it can be unit-tested; ``decide`` returns a target head
    yaw in radians or None.
    """

    def __init__(self, min_conf=None, min_delta_deg=12.0,
                 min_interval_s=1.0, max_yaw_deg=None):
        # Measured 2026-09-30: real speech from in front of NAO localizes at
        # confidence 0.1-0.5, so a 0.4 floor dropped most of it.
        if min_conf is None:
            min_conf = _env_float("VOICE_TURN_MIN_CONF", 0.15)
        # Sources further round than this are ignored, not clamped. NAO sits
        # with a wall behind it; on 2026-09-30 an echo off that wall turned
        # the head fully right and it stayed staring at the wall.
        if max_yaw_deg is None:
            max_yaw_deg = _env_float("VOICE_TURN_MAX_DEG", 80.0)
        self.min_conf = float(min_conf)
        self.min_delta = math.radians(float(min_delta_deg))
        self.min_interval_s = float(min_interval_s)
        self.max_yaw = math.radians(float(max_yaw_deg))
        self._last_ts = None
        self._last_turn_at = None

    def decide(self, event, now, current_yaw, ignore=False):
        if event is None:
            return None
        if event["ts"] == self._last_ts:
            return None  # localizer has not fired since last poll
        self._last_ts = event["ts"]
        if ignore:
            return None  # NAO itself is talking
        if event["confidence"] < self.min_conf:
            return None
        if (self._last_turn_at is not None
                and now - self._last_turn_at < self.min_interval_s):
            return None
        target = event["head_yaw"] + event["azimuth"]
        # Normalise to [-pi, pi] so "behind" is recognised either way round.
        while target > math.pi:
            target -= 2 * math.pi
        while target < -math.pi:
            target += 2 * math.pi
        if abs(target) > self.max_yaw:
            return None  # behind NAO: wall echo or noise, not a speaker
        if abs(target - current_yaw) < self.min_delta:
            return None  # already facing them
        self._last_turn_at = now
        return target


# ---------------------------------------------------------------------------
# HeadAwareness
# ---------------------------------------------------------------------------
class HeadAwareness(object):
    """Turn toward whoever speaks, hold there, and lock onto their face.

    * A confident sound event turns the head toward the voice and the head
      stays there (NAOqi's ALBasicAwareness looked back after 1 s whenever
      its person detector found nobody, which on this robot was always).
    * After each turn ALTracker is restarted in face mode, so if a face is
      in view the head settles on it and follows it.
    * Sounds are ignored while NAO's own reply plays and for
      ``tts_tail_s`` after, so it does not turn toward its own speaker.
    """

    def __init__(self, memory, motion, tracker, basic_awareness=None,
                 is_speaking=None, life_guard=None, log=None,
                 turner=None, tts_tail_s=None, poll_s=0.1,
                 life_check_s=2.0, turn_speed=0.3, on_turn=None,
                 recenter_s=None):
        self._memory = memory
        self._motion = motion
        self._tracker = tracker
        self._ba = basic_awareness
        self._is_speaking = is_speaking
        self._life_guard = life_guard
        self._log = log or _NullLog()
        self.turner = turner or VoiceTurner()
        if tts_tail_s is None:
            tts_tail_s = _env_float("AWARENESS_TTS_TAIL_S", 0.8)
        self.gate = SpeakingGate(tail_s=tts_tail_s)
        self.poll_s = float(poll_s)
        self.life_check_s = float(life_check_s)
        self.turn_speed = float(turn_speed)
        # Called after each voice turn so the caller can re-identify who is
        # now in front of the camera.
        self.on_turn = on_turn
        # If the face tracker has nobody for this long, glide back to the
        # front, where people actually stand.
        if recenter_s is None:
            recenter_s = _env_float("VOICE_TURN_RECENTER_S", 4.0)
        self.recenter_s = float(recenter_s)
        self._lost_since = None
        self._stop = threading.Event()
        self._thread = None
        self.running = False
        self.turns = 0

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _try(obj, method, *args):
        fn = getattr(obj, method, None) if obj is not None else None
        if fn is None:
            return False
        try:
            fn(*args)
            return True
        except Exception:
            return False

    def _speaking(self):
        cb = self._is_speaking
        if cb is None:
            return False
        try:
            return bool(cb())
        except Exception:
            return False

    def _current_yaw(self):
        try:
            return float(self._motion.getAngles("HeadYaw", True)[0])
        except Exception:
            return 0.0

    def _read_event(self):
        try:
            return parse_sound_event(
                self._memory.getData("ALSoundLocalization/SoundLocated"))
        except Exception:
            return None

    def _start_face_tracking(self):
        t = self._tracker
        if t is None:
            return
        self._try(t, "setMode", "Head")
        self._try(t, "registerTarget", "Face", 0.15)
        self._try(t, "track", "Face")

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        if self._memory is None or self._motion is None or self.running:
            return False
        if self._life_guard is not None:
            self._life_guard.check_once(context="engage")
        # NAOqi's own awareness would fight us for the head and look away.
        if not self._try(self._ba, "setEnabled", False):
            self._try(self._ba, "stopAwareness")
        # ALSoundLocalization only fires while something subscribes to it.
        sl = self._sound_loc_proxy()
        self._try(sl, "setParameter", "Sensitivity",
                  _env_float("VOICE_TURN_SENSITIVITY", 0.9))
        self._try(sl, "subscribe", "nao_head_awareness")
        self._start_face_tracking()
        self.running = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop,
                                        name="nao-head-awareness")
        self._thread.daemon = True
        self._thread.start()
        self._log.info("head_awareness_started", mode="voice_turn",
                       min_conf=self.turner.min_conf,
                       tts_tail_s=self.gate.tail_s)
        return True

    def _sound_loc_proxy(self):
        return getattr(self, "_sound_loc", None)

    def stop(self):
        if not self.running:
            return
        self.running = False
        self._stop.set()
        t = self._thread
        self._thread = None
        if t is not None and t is not threading.current_thread():
            try:
                t.join(timeout=1.5)
            except Exception:
                pass
        self._try(self._tracker, "stopTracker")
        self._try(self._tracker, "unregisterAllTargets")
        self._try(self._sound_loc_proxy(), "unsubscribe",
                  "nao_head_awareness")
        self._log.info("head_awareness_stopped", turns=self.turns)

    def tick(self, now):
        """One poll step. Returns the target yaw (rad) if it turned."""
        self.gate.step(now, self._speaking())
        event = self._read_event()
        current = self._current_yaw()
        target = self.turner.decide(event, now, current,
                                    ignore=self.gate.paused)
        if target is None:
            self._maybe_recenter(now, current)
            return None
        self._turn_to(target, event)
        return target

    def _face_lost(self):
        t = self._tracker
        if t is None:
            return True
        try:
            return bool(t.isTargetLost())
        except Exception:
            return False

    def _maybe_recenter(self, now, current_yaw):
        if abs(current_yaw) < math.radians(15):
            self._lost_since = None
            return
        if not self._face_lost():
            self._lost_since = None
            return
        if self._lost_since is None:
            self._lost_since = now
            return
        if now - self._lost_since < self.recenter_s:
            return
        self._lost_since = None
        self._try(self._tracker, "stopTracker")
        try:
            self._motion.angleInterpolationWithSpeed(
                ["HeadYaw"], [0.0], self.turn_speed * 0.6)
        except Exception:
            pass
        self._log.info("voice_turn_recenter",
                       from_deg=round(math.degrees(current_yaw)))
        self._start_face_tracking()

    def _turn_to(self, yaw, event):
        # Stop the face tracker first or it drags the head straight back
        # to the face it was already following.
        self._try(self._tracker, "stopTracker")
        try:
            self._motion.angleInterpolationWithSpeed(
                ["HeadYaw"], [yaw], self.turn_speed)
        except Exception as exc:
            self._log.debug("voice_turn_failed", error=str(exc))
        self.turns += 1
        self._log.info("voice_turn",
                       yaw_deg=round(math.degrees(yaw)),
                       conf=round(event.get("confidence", 0.0), 2))
        # Re-arm face tracking so a face in the new view takes over.
        self._start_face_tracking()
        self._lost_since = None
        cb = self.on_turn
        if cb is not None:
            try:
                cb(yaw)
            except Exception as exc:
                self._log.debug("voice_turn_callback_failed", error=str(exc))

    def _loop(self):
        next_life_check = time.time() + self.life_check_s
        while not self._stop.is_set():
            now = time.time()
            try:
                self.tick(now)
            except Exception as exc:
                self._log.debug("awareness_tick_failed", error=str(exc))
            if self._life_guard is not None and now >= next_life_check:
                next_life_check = now + self.life_check_s
                try:
                    self._life_guard.check_once(context="session")
                except Exception:
                    pass
            self._stop.wait(self.poll_s)


def build_life_guard(ip, port, log=None, on_redisabled=None):
    return LifeGuard(_proxy("ALAutonomousLife", ip, port), log=log,
                     on_redisabled=on_redisabled)


def build_head_awareness(ip, port, is_speaking=None, life_guard=None,
                         log=None, on_turn=None):
    memory = _proxy("ALMemory", ip, port)
    motion = _proxy("ALMotion", ip, port)
    if memory is None or motion is None:
        return None
    ha = HeadAwareness(
        memory, motion, _proxy("ALTracker", ip, port),
        basic_awareness=_proxy("ALBasicAwareness", ip, port),
        is_speaking=is_speaking, life_guard=life_guard, log=log,
        on_turn=on_turn)
    ha._sound_loc = _proxy("ALSoundLocalization", ip, port)
    return ha

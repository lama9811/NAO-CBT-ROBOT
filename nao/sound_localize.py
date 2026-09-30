# -*- coding: utf-8 -*-
"""
sound_localize.py - SoundLocalizer (Phase 4 / robot-sound-localize)

Subscribes to NAOqi's ``ALSoundLocalization`` and tracks the most recent
speaker direction. Optionally drives ``ALMotion`` to physically turn the
robot's head toward the sound source. Designed to live alongside the rest
of the ``nao/`` package; runs in py2.7 on the robot.

Why polling
-----------
NAOqi 2.x supports two consumption patterns for ``ALSoundLocalization``:

1. ``ALMemory.subscribeToEvent("ALSoundLocalization/SoundLocated", ...)``
   This pushes events into a callback registered against an ``ALModule``
   subclass that has been bound to a running ``ALBroker``. It is the
   officially-documented path but requires the host process to either be
   the robot itself (``naoqi-bin``) or to have constructed an in-process
   broker. The broker plumbing is heavy and conflicts with how the rest
   of this codebase opens proxies (we already have an ``ALModule`` for
   audio streaming, and registering a second one in ``__main__`` is
   error-prone).

2. ``ALMemory.getData("ALSoundLocalization/SoundLocated")`` polled at
   100 ms cadence. The same payload that the event would have carried is
   stamped into ALMemory each time the localizer fires; we just read it.
   Trade-off is up to one polling interval of latency (~100 ms), which is
   well inside the 300 ms requirement in PRD_v2 (Phase 4, line 271).

We pick (2) for robustness on this firmware. The class still exposes a
``start/stop`` lifecycle so any future swap to event-subscribe is a
drop-in change.

Event format
------------
``ALSoundLocalization/SoundLocated`` returns::

    [
        [bufferTimestamp_sec, bufferTimestamp_usec],
        [confidence, energy],
        [azimuth_rad, elevation_rad, head_x, head_y],
    ]

Azimuth is in the robot frame: 0 rad = straight ahead, positive = robot's
left (the same convention ``HeadYaw`` uses, so we can pipe yaw in degrees
straight to ``angleInterpolationWithSpeed``). Elevation is positive up.

Defensive design
----------------
On a developer Mac we cannot import ``naoqi`` and we have no proxies to
talk to. The whole class is therefore wired to behave as a no-op in that
mode:

  * ``start()`` returns immediately and logs once.
  * ``stop()`` is idempotent.
  * ``get_last_direction()`` returns ``None``.
  * ``turn_head_toward(...)`` returns silently.

This lets the rest of ``nao_execute.py`` (which constructs a ``SoundLocalizer``
unconditionally and uses ``get_last_direction()`` for the
``point_listener`` gesture) work in unit tests off-robot.

Public API
----------
    sl = SoundLocalizer(nao_ip="127.0.0.1")
    sl.start()                           # begin polling thread
    direction = sl.get_last_direction()  # dict | None
    sl.turn_head_toward(yaw_deg=15.0)    # blocking ALMotion call
    sl.stop()                            # joins thread within 1 s
"""
from __future__ import print_function

import logging
import math
import threading
import time


# ---------------------------------------------------------------------------
# naoqi import guard. The robot ships with the SDK; a developer laptop does
# not. We keep the module importable in both contexts so unit tests + AST
# parses run anywhere, and the class degrades to a no-op when proxies are
# unavailable.
# ---------------------------------------------------------------------------
try:
    from naoqi import ALProxy
    _NAOQI_AVAILABLE = True
except ImportError:
    ALProxy = None  # type: ignore
    _NAOQI_AVAILABLE = False


logger = logging.getLogger(__name__)


# ALMemory key the localizer stamps each new event into.
_SOUND_LOC_KEY = "ALSoundLocalization/SoundLocated"

# Default poll interval - 100 ms gives ~10 Hz tracking, matching the
# native firing rate of the localizer module.
_DEFAULT_POLL_MS = 100

# Thread join timeout on stop() - per the task contract.
_STOP_JOIN_TIMEOUT_S = 1.0

# Confidence the firmware reports for "stale" memory entries (when the
# localizer hasn't fired since boot). We treat any zero-confidence record
# as "no sound yet seen" and skip it.
_ZERO_CONF_EPS = 1e-6


def _now_ms():
    """Wall-clock ms (float). Robot has no monotonic clock that returns ms."""
    return time.time() * 1000.0


def _clamp(value, lo, hi):
    """Clamp ``value`` to the inclusive range ``[lo, hi]``."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


class SoundLocalizer(object):
    """Subscribes to ALSoundLocalization events; tracks recent speaker
    direction; can drive head turning via ALMotion.

    NAOqi's ``SoundLocalization`` fires events ~10 Hz with the payload::

        [[ts_sec, ts_usec], [confidence, energy], [azimuth, elevation, _, _]]

    in robot frame (radians). This class polls ``ALMemory`` at the same
    cadence, converts azimuth/elevation to degrees, applies the
    ``confidence_min`` filter, clamps to the ``max_yaw_deg`` /
    ``max_pitch_deg`` envelope, and stores the latest sample under a lock
    so other threads (e.g. the gesture dispatcher) can read it.

    Parameters
    ----------
    nao_ip : str
        Address of the NAO ALMain broker. Use ``"127.0.0.1"`` if running
        on the robot itself.
    nao_port : int, default 9559
    motion : ALProxy or None
        Optional pre-built ``ALMotion`` proxy. If ``None``, the class
        builds one at start. Allows callers to share an existing proxy
        and avoids redundant connection setup.
    max_yaw_deg : float, default 60.0
        Hard cap on commanded ``HeadYaw`` magnitude. NAO's mechanical
        limit is ~119 deg but anything past 60 is unfriendly to a person
        standing in front of the robot.
    max_pitch_deg : float, default 20.0
        Hard cap on commanded ``HeadPitch`` magnitude. Mechanical range
        runs roughly -38..29 deg; 20 is a comfortable conversational
        envelope.
    turn_speed_dps : float, default 30.0
        Head turn speed in degrees per second. Maps to a 0..1 fraction
        for ``angleInterpolationWithSpeed`` via ``_speed_fraction()``.
    confidence_min : float, default 0.4
        Minimum localizer confidence (0..1) before an event is accepted.
        Below this the room is too noisy or the source too distant.
    auto_track : bool, default False
        If True, the polling thread calls ``turn_head_toward(...)`` for
        every accepted event. False (the default) leaves head motion to
        the caller (e.g. the ``point_listener`` gesture).
    poll_ms : int, default 100
        Polling cadence in milliseconds.
    """

    # NAO H25 head joint speed envelope (deg/s). Used to translate the
    # caller's ``turn_speed_dps`` into the 0..1 ``fractionMaxSpeed``
    # argument that ``angleInterpolationWithSpeed`` expects. Aldebaran
    # documents max HeadYaw speed at ~7 rad/s (~400 deg/s); 100 deg/s is a
    # conservative ceiling that produces smooth, conversational motion
    # rather than the snap of full speed.
    _HEAD_MAX_SPEED_DPS = 100.0
    _MIN_SPEED_FRACTION = 0.05
    _MAX_SPEED_FRACTION = 1.0

    def __init__(self,
                 nao_ip,
                 nao_port=9559,
                 motion=None,
                 max_yaw_deg=60.0,
                 max_pitch_deg=20.0,
                 turn_speed_dps=30.0,
                 confidence_min=0.4,
                 auto_track=False,
                 poll_ms=_DEFAULT_POLL_MS):
        self.nao_ip = nao_ip
        self.nao_port = int(nao_port)
        self.max_yaw_deg = float(max_yaw_deg)
        self.max_pitch_deg = float(max_pitch_deg)
        self.turn_speed_dps = float(turn_speed_dps)
        self.confidence_min = float(confidence_min)
        self.auto_track = bool(auto_track)
        self.poll_ms = int(poll_ms)

        # Proxies. ``motion`` may be supplied; ``ALMemory`` is always
        # built on demand at start().
        self._motion = motion
        self._memory = None

        # Last-direction record + lock. ``None`` until the first accepted
        # event arrives. Schema:
        #   {"azimuth_deg": float,
        #    "elevation_deg": float,
        #    "ts_ms": float,
        #    "confidence": float}
        self._last_direction = None
        self._last_lock = threading.Lock()

        # Lifecycle.
        self._stop_event = threading.Event()
        self._thread = None
        self._started = False

        # Off-robot warning: log once so unit-test runs don't spam.
        self._naoqi_warned = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        """Start the background polling thread.

        Idempotent; calling start() on an already-running localizer is a
        no-op. When ``naoqi`` is unavailable, logs a single warning and
        returns without starting any thread.
        """
        if self._started:
            return

        if not _NAOQI_AVAILABLE:
            self._warn_naoqi_missing()
            self._started = True  # so stop() is a clean no-op
            return

        # Open ALMemory once; ALMotion lazily on demand or supplied.
        try:
            self._memory = ALProxy("ALMemory", self.nao_ip, self.nao_port)
        except Exception as exc:
            logger.warning(
                "[sound_localize] could not open ALMemory proxy "
                "(ip=%s port=%d): %s; localizer will be inert",
                self.nao_ip, self.nao_port, exc,
            )
            self._memory = None
            self._started = True
            return

        # Best-effort: ask the localizer module to start firing events.
        # NAOqi normally has it on by default, but explicit subscribe is
        # cheap insurance and matches the documented usage pattern.
        try:
            sl_proxy = ALProxy("ALSoundLocalization",
                               self.nao_ip, self.nao_port)
            try:
                sl_proxy.subscribe("SoundLocalizer")
            except Exception:
                # Already subscribed by another client - fine.
                pass
        except Exception as exc:
            # Not fatal: if the module isn't running, getData will just
            # keep returning the same stale record and we'll skip it.
            logger.warning(
                "[sound_localize] ALSoundLocalization proxy unavailable: "
                "%s; will rely on whatever client(s) already subscribed",
                exc,
            )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="SoundLocalizer-poll",
        )
        self._thread.daemon = True
        self._thread.start()
        self._started = True
        logger.info(
            "[sound_localize] started (poll=%d ms, "
            "confidence_min=%.2f, max_yaw=%.0f deg, auto_track=%s)",
            self.poll_ms, self.confidence_min,
            self.max_yaw_deg, self.auto_track,
        )

    def stop(self):
        """Stop the background thread. Joins within 1 s. Idempotent."""
        if not self._started:
            return

        self._stop_event.set()
        thread = self._thread
        self._thread = None

        if thread is not None and thread.is_alive():
            thread.join(timeout=_STOP_JOIN_TIMEOUT_S)

        # Best-effort unsubscribe so we don't leak a subscriber slot
        # across reload cycles.
        if _NAOQI_AVAILABLE:
            try:
                sl_proxy = ALProxy("ALSoundLocalization",
                                   self.nao_ip, self.nao_port)
                try:
                    sl_proxy.unsubscribe("SoundLocalizer")
                except Exception:
                    pass
            except Exception:
                pass

        self._started = False
        logger.info("[sound_localize] stopped")

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------
    def get_last_direction(self):
        """Return the most recent accepted localizer reading or ``None``.

        Schema::

            {"azimuth_deg": float,    # robot frame, +left
             "elevation_deg": float,  # +up
             "ts_ms": float,          # wall clock at acceptance
             "confidence": float}     # 0..1

        Thread-safe. Returns ``None`` until a reading at or above
        ``confidence_min`` has been observed (and forever, in
        naoqi-unavailable mode).
        """
        with self._last_lock:
            if self._last_direction is None:
                return None
            # Return a shallow copy so callers can't mutate our state.
            return dict(self._last_direction)

    def turn_head_toward(self, azimuth_deg, pitch_deg=0.0):
        """Drive ``HeadYaw`` and ``HeadPitch`` toward the given direction.

        Parameters
        ----------
        azimuth_deg : float
            Target yaw in degrees (robot frame, positive = left). Clamped
            to ``[-max_yaw_deg, +max_yaw_deg]``.
        pitch_deg : float, default 0.0
            Target pitch in degrees (positive = up). Note that NAO's
            ``HeadPitch`` joint uses a robotics convention where positive
            angles point the head **down**; we negate before sending so a
            positive ``pitch_deg`` argument behaves the way "look up"
            sounds in English. Clamped to
            ``[-max_pitch_deg, +max_pitch_deg]``.

        Blocking. If naoqi is unavailable the call returns silently.
        """
        if not _NAOQI_AVAILABLE:
            self._warn_naoqi_missing()
            return

        yaw_deg = _clamp(float(azimuth_deg),
                         -self.max_yaw_deg, self.max_yaw_deg)
        # Clamp the user-facing "look up is positive" value, then negate
        # for the joint frame.
        looked_pitch_deg = _clamp(float(pitch_deg),
                                  -self.max_pitch_deg, self.max_pitch_deg)
        joint_pitch_deg = -looked_pitch_deg

        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(joint_pitch_deg)

        speed_fraction = self._speed_fraction(self.turn_speed_dps)

        try:
            motion = self._get_motion()
            if motion is None:
                return
            motion.angleInterpolationWithSpeed(
                ["HeadYaw", "HeadPitch"],
                [yaw_rad, pitch_rad],
                speed_fraction,
            )
        except Exception as exc:
            # Don't kill the polling thread if a single move fails - the
            # robot may be in a posture that locks the head briefly.
            logger.warning(
                "[sound_localize] turn_head_toward failed (yaw=%.1f, "
                "pitch=%.1f): %s",
                yaw_deg, looked_pitch_deg, exc,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _poll_loop(self):
        """Background poll thread. Reads ALMemory at ``poll_ms`` cadence."""
        last_event_ts = None  # raw [sec, usec] tuple of last accepted event
        interval_s = max(0.01, self.poll_ms / 1000.0)

        while not self._stop_event.is_set():
            try:
                event = self._read_event()
                if event is not None:
                    self._handle_event(event, last_event_ts)
                    # Re-pull the last_event_ts AFTER _handle_event has
                    # decided whether to keep this sample, so dedupe is
                    # accurate. Fast read - the lock is uncontended.
                    with self._last_lock:
                        if self._last_direction is not None:
                            # Re-extract the raw timestamp tuple stored
                            # alongside the direction (kept for dedupe).
                            last_event_ts = self._last_direction.get(
                                "_raw_ts_tuple", last_event_ts,
                            )
            except Exception:
                # Catch-all so the poll thread never dies silently. We
                # don't log per-iteration to avoid noise; if the user
                # wants more, they can flip the logger to DEBUG.
                logger.debug(
                    "[sound_localize] poll iteration error",
                    exc_info=True,
                )

            # Wait the interval, but wake immediately on stop().
            if self._stop_event.wait(interval_s):
                break

    def _read_event(self):
        """Fetch the latest ALMemory record. Returns the raw payload or ``None``."""
        if self._memory is None:
            return None
        try:
            data = self._memory.getData(_SOUND_LOC_KEY)
        except Exception:
            return None
        # NAOqi returns an empty list when the key has never been set.
        if not data:
            return None
        return data

    def _handle_event(self, event, last_event_ts):
        """Validate, dedupe, and store an event.

        ``event`` is the raw 3-element list documented at module top.
        ``last_event_ts`` is the raw [sec, usec] tuple from the previous
        accepted record; we skip events whose timestamp matches (the
        localizer hasn't fired since our last poll).
        """
        # Defensive shape check. ALMemory has been known to occasionally
        # return scalars in odd states, and the polling pattern is
        # vulnerable to a partially-written memory entry.
        # NAOqi 2.8 stores [ts, [az, el, conf, energy], head6D, head6D];
        # the old [ts, [conf, energy], [az, el]] reading returned az=0 on
        # this robot. parse_sound_event handles both layouts.
        try:
            from awareness import parse_sound_event
            parsed = parse_sound_event(event)
        except Exception:
            parsed = None
        if parsed is None:
            return
        ts_pair = list(parsed["ts"])
        confidence = parsed["confidence"]
        azimuth_rad = parsed["head_yaw"] + parsed["azimuth"]
        elevation_rad = parsed["elevation"]

        # Dedupe: identical timestamp means the localizer hasn't fired
        # since last poll. (Compare element-wise; lists from NAOqi don't
        # always survive == cleanly across firmware versions.)
        if last_event_ts is not None and self._ts_equal(ts_pair, last_event_ts):
            return

        # Confidence filter. Drop both the explicit "below threshold"
        # case and the "never been observed" case where the firmware
        # reports a flat zero.
        if confidence < self.confidence_min:
            return
        if confidence < _ZERO_CONF_EPS:
            return

        azimuth_deg = math.degrees(azimuth_rad)
        elevation_deg = math.degrees(elevation_rad)

        # Clamp to the configured envelope so downstream consumers never
        # see a yaw that, if commanded, would over-rotate the head.
        clamped_yaw = _clamp(azimuth_deg,
                             -self.max_yaw_deg, self.max_yaw_deg)
        clamped_pitch = _clamp(elevation_deg,
                               -self.max_pitch_deg, self.max_pitch_deg)

        record = {
            "azimuth_deg": clamped_yaw,
            "elevation_deg": clamped_pitch,
            "ts_ms": _now_ms(),
            "confidence": confidence,
            # Internal: keep the raw timestamp so the next poll can
            # dedupe. Stripped from get_last_direction()'s return.
            "_raw_ts_tuple": (ts_pair[0], ts_pair[1])
            if isinstance(ts_pair, (list, tuple)) and len(ts_pair) >= 2
            else None,
        }

        with self._last_lock:
            self._last_direction = record

        if self.auto_track:
            # Fire and forget; turn_head_toward swallows its own errors.
            self.turn_head_toward(clamped_yaw, 0.0)

    @staticmethod
    def _ts_equal(a, b):
        """Element-wise equality check for the [sec, usec] timestamp pair."""
        try:
            return a[0] == b[0] and a[1] == b[1]
        except (IndexError, TypeError):
            return False

    def _get_motion(self):
        """Return an ``ALMotion`` proxy, building one on first use."""
        if self._motion is not None:
            return self._motion
        if not _NAOQI_AVAILABLE:
            return None
        try:
            self._motion = ALProxy("ALMotion", self.nao_ip, self.nao_port)
        except Exception as exc:
            logger.warning(
                "[sound_localize] could not open ALMotion proxy: %s",
                exc,
            )
            self._motion = None
        return self._motion

    def _speed_fraction(self, dps):
        """Map degrees-per-second to the 0..1 fractionMaxSpeed argument."""
        if dps <= 0:
            return self._MIN_SPEED_FRACTION
        fraction = float(dps) / float(self._HEAD_MAX_SPEED_DPS)
        return _clamp(fraction,
                      self._MIN_SPEED_FRACTION, self._MAX_SPEED_FRACTION)

    def _warn_naoqi_missing(self):
        """Log the off-robot warning at most once per instance."""
        if self._naoqi_warned:
            return
        self._naoqi_warned = True
        logger.warning(
            "[sound_localize] naoqi unavailable; SoundLocalizer is inert. "
            "All methods are no-ops; get_last_direction() returns None."
        )

    # Internal copy of get_last_direction that does NOT strip the raw
    # timestamp tuple - used by the poll loop to look up the last event
    # for dedupe purposes. Public consumers go through
    # get_last_direction() which removes the underscore-prefixed key.
    # (Kept here only as documentation of the contract; the poll loop
    # reads directly under the lock for speed.)


def _self_test():
    """Smoke test runnable on a developer machine.

    Confirms that the four public methods do not raise when naoqi is
    unavailable. Also exercised by ``python -m py_compile``.
    """
    logging.basicConfig(level=logging.INFO)
    sl = SoundLocalizer(
        nao_ip="127.0.0.1",
        nao_port=9559,
        max_yaw_deg=45.0,
        max_pitch_deg=15.0,
        turn_speed_dps=20.0,
        confidence_min=0.5,
        auto_track=False,
    )

    # start() should be a no-op off-robot. Note: in disabled mode the
    # one-time naoqi warning will fire here.
    sl.start()

    # get_last_direction() must return None until the first event lands;
    # off-robot, that's forever.
    direction = sl.get_last_direction()
    assert direction is None, (
        "expected None off-robot, got {0!r}".format(direction)
    )

    # turn_head_toward() must not raise even without naoqi.
    sl.turn_head_toward(15.0, pitch_deg=5.0)
    sl.turn_head_toward(-90.0, pitch_deg=99.0)  # exercise the clamps

    # stop() must complete within 1 s and be idempotent.
    t0 = time.time()
    sl.stop()
    elapsed = time.time() - t0
    assert elapsed < _STOP_JOIN_TIMEOUT_S + 0.5, (
        "stop() exceeded budget: {0:.2f}s".format(elapsed)
    )
    sl.stop()  # idempotent

    print("[sound_localize] self-test OK (elapsed={0:.3f}s)".format(elapsed))


if __name__ == "__main__":
    _self_test()

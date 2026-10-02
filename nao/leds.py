# -*- coding: utf-8 -*-
"""LedDriver -- ALLeds + ALAudioPlayer wrapper for the Phase 3 wake state machine.

The wake state machine (``nao/wake_state.py``) drives one of five named states:
IDLE, AWARE, ENGAGED, LISTENING, SPEAKING. Each maps to a different eye
colour + transition speed. ``LedDriver`` exposes a single, flat API the
state machine calls so the colour vocabulary lives in one place.

Two firmware-quirk things are worth knowing:

1. ``ALLeds.fadeRGB`` exists in two signatures. NAO V6 firmware accepts
   ``fadeRGB(name, packed_int_rgb, duration)`` -- the same form the existing
   ``utils/nao_execute.py:change_eye_color`` uses, so we know the robot
   honours it. Some firmware revisions also expose
   ``fadeRGB(name, r, g, b, duration)`` with floats. We *try* the packed-int
   form first (it's the one verified working on this fleet); on a
   ``RuntimeError`` we fall back to the float-tuple call. Either way the LED
   eventually lands on the requested colour.

2. ``ALAudioPlayer.playFile`` requires a path on disk -- it cannot play
   in-memory bytes. ``chime()`` writes a 220 Hz sine, 200 ms WAV to
   ``/tmp/nao_chime.wav`` once on first call (or after the file is missing)
   and reuses it on subsequent calls. The disk write is ~6 KB so the cost is
   negligible, and ``/tmp`` is tmpfs on the robot so we don't wear flash.

Off-robot guard
---------------
``naoqi`` only exists on the robot. On a developer Mac running
``py_compile`` / ``ast.parse`` we still want this module importable, so the
constructor catches ``ImportError`` / ``RuntimeError`` from ``ALProxy`` and
flips ``_disabled = True``. Every public method then no-ops + emits a single
warning. This lets ``python -m py_compile nao/leds.py`` pass and lets the
``__main__`` self-test exercise every preset without a robot connection.

Threading
---------
Multiple state-machine threads (and the conversation thread that toggles
``set_listening`` / ``set_speaking``) may call into ``LedDriver`` at the
same time. Each ``pulse()`` invocation runs on its own daemon thread; we
serialise proxy access with ``self._lock`` so a fade and a pulse on the
same group don't issue overlapping ALLeds calls. Pulses are tracked by
group name in ``self._pulses`` so ``stop_pulse(group)`` reliably halts the
prior worker before a new one starts.
"""
from __future__ import print_function

import logging
import math
import os
import struct
import threading
import time
import wave


# naoqi only ships on the robot. Mirror the pattern in ``audio_module.py`` so
# this module imports cleanly off-robot for syntax / smoke checks.
try:
    from naoqi import ALProxy  # noqa: F401
    _NAOQI_AVAILABLE = True
except ImportError:
    _NAOQI_AVAILABLE = False

    def ALProxy(*args, **kwargs):  # noqa: D401
        raise RuntimeError("ALProxy unavailable: naoqi not importable here")


logger = logging.getLogger(__name__)

# Where chime() writes its cached WAV. /tmp is tmpfs on NAO V6 so this
# survives across reboots fine and never touches flash.
_CHIME_PATH = "/tmp/nao_chime.wav"
_CHIME_FREQ_HZ = 220
_CHIME_DURATION_S = 0.20
_CHIME_SAMPLE_RATE = 16000
# The chime spec is "80 dB". ALAudioPlayer scales to a master volume in the
# 0..1 range; the actual SPL depends on the speaker setting. Writing the
# WAV at full-scale (~32700 amplitude on int16) and bumping the player
# master volume gets us close to 80 dB on the H25 speaker. Subtle distortion
# at full-scale is acceptable for a 200 ms tone.
_CHIME_AMPLITUDE = 32000

# Pulse uses 30% intensity of the base RGB for the "low" half of the cycle.
_PULSE_LOW_INTENSITY = 0.30


def _rgb_to_packed_int(rgb):
    """Convert ``(r, g, b)`` floats in 0..1 to ALLeds packed 0xRRGGBB.

    ALLeds.fadeRGB(name, color, duration) wants color as 0xRRGGBB. Float
    values outside 0..1 are clamped (not raised) so the state machine can
    keep nudging towards a colour without us having to validate every call.
    """
    r, g, b = rgb
    r = max(0.0, min(1.0, float(r)))
    g = max(0.0, min(1.0, float(g)))
    b = max(0.0, min(1.0, float(b)))
    ri = int(round(r * 255.0))
    gi = int(round(g * 255.0))
    bi = int(round(b * 255.0))
    return (ri << 16) | (gi << 8) | bi


def _scale_rgb(rgb, factor):
    """Multiply each channel of ``rgb`` by ``factor``, clamping to 0..1."""
    factor = max(0.0, float(factor))
    return tuple(max(0.0, min(1.0, float(c) * factor)) for c in rgb)


def _ensure_chime_wav(path=_CHIME_PATH,
                      freq_hz=_CHIME_FREQ_HZ,
                      duration_s=_CHIME_DURATION_S,
                      sample_rate=_CHIME_SAMPLE_RATE,
                      amplitude=_CHIME_AMPLITUDE):
    """Write the cached 220 Hz sine WAV if it isn't already on disk.

    Returns the path. Idempotent -- if the file already exists and is
    non-empty we trust it. ``chime()`` always calls through here so a
    deleted /tmp won't break the second chime of a session.
    """
    try:
        if os.path.isfile(path) and os.path.getsize(path) > 44:
            return path
    except OSError:
        # stat failed -- regenerate.
        pass

    nsamples = int(sample_rate * duration_s)
    # Sine table built sample-by-sample. With 16 kHz x 0.2 s = 3200 samples
    # this is fast enough that doing it eagerly is fine; no numpy on robot.
    frames = bytearray()
    two_pi_f_over_sr = 2.0 * math.pi * freq_hz / float(sample_rate)
    for n in range(nsamples):
        sample = int(amplitude * math.sin(two_pi_f_over_sr * n))
        # Apply a short linear fade-in / fade-out (5 ms each side) so the
        # tone doesn't click at start/stop.
        fade_n = int(0.005 * sample_rate)
        if n < fade_n:
            sample = int(sample * (n / float(fade_n)))
        elif n > nsamples - fade_n:
            sample = int(sample * ((nsamples - n) / float(fade_n)))
        frames.extend(struct.pack("<h", sample))

    try:
        wf = wave.open(path, "wb")
    except (IOError, OSError) as exc:
        logger.warning("[leds] could not open %s for writing: %s", path, exc)
        return None
    try:
        wf.setnchannels(1)
        wf.setsampwidth(2)        # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))
    finally:
        wf.close()
    return path


class LedDriver(object):
    """ALLeds + ALAudioPlayer wrapper used by ``WakeStateMachine`` for state
    visualisation.

    Construct once per process; pass into ``WakeStateMachine`` and into any
    other code that wants to flash the eyes (e.g. main loop on shutdown).

    Off-robot, instantiation succeeds with ``self._disabled = True`` and
    every method becomes a no-op. The class never raises out of a public
    method -- it logs a single warning per failed proxy call so the wake
    state machine never crashes because the LEDs hiccup.
    """

    # Group names match ALLeds documented groups so they can be passed
    # directly into proxy calls.
    EYES_GROUP = "FaceLeds"
    CHEST_GROUP = "ChestLeds"
    EAR_LEFT_GROUP = "EarLeds"

    # Phase 3 state -> colour matrix. Floats 0..1 each channel. The packed
    # int representation is computed lazily inside ``fade``.
    COLOR_GRAY      = (0.10, 0.10, 0.12)   # IDLE
    COLOR_SOFT_BLUE = (0.10, 0.30, 0.70)   # AWARE
    COLOR_SOLID_BLUE = (0.20, 0.50, 1.00)  # ENGAGED
    COLOR_CYAN      = (0.10, 0.80, 0.95)   # LISTENING
    COLOR_YELLOW    = (1.00, 0.80, 0.10)   # SPEAKING
    COLOR_GREEN     = (0.10, 0.90, 0.30)   # camera-active (Phase 6)

    # Default fade durations per state preset. Pulled from the Phase 3 task
    # map so callers don't sprinkle magic numbers across the state machine.
    DUR_IDLE = 0.6
    DUR_AWARE = 0.4
    DUR_ENGAGED = 0.2
    DUR_LISTENING = 0.3
    DUR_SPEAKING = 0.2

    # Turn-taking "thinking" cue: the ENGAGED blue, pulsing slowly while the
    # server works out a reply. Not purple -- purple eyes mean Autonomous
    # Life's sleep mode on this robot.
    THINKING_PERIOD_S = 1.4

    # Paced breathing: eyes brighten on the inhale and dim on the exhale.
    COLOR_BREATH = (0.10, 0.80, 0.95)      # same cyan as LISTENING
    BREATH_LOW_INTENSITY = 0.12
    BREATH_MAX_S = 10.0

    # Master volume to set on ALAudioPlayer before chime playback. 1.0 is
    # the documented max. The H25 chassis speaker pegs ~80 dB at this volume
    # for a full-scale 220 Hz tone.
    CHIME_VOLUME = 1.0

    def __init__(self, nao_ip, nao_port=9559):
        self._nao_ip = nao_ip
        self._nao_port = int(nao_port)
        self._leds = None
        self._player = None
        self._disabled = False
        # One lock for proxy access -- both fade() and pulse() workers call
        # into ALLeds, and overlapping calls to setIntensity sometimes throw
        # on naoqi. Serialising is cheap (a fade is < 1 ms of Python work).
        self._lock = threading.Lock()
        # Active pulse threads keyed by group name. ``stop_pulse(group)``
        # signals the corresponding event to halt cleanly.
        self._pulses = {}
        self._pulses_lock = threading.Lock()
        self._chime_path = None
        self._warned_disabled = False

        if not _NAOQI_AVAILABLE:
            self._disabled = True
            self._warn_once(
                "naoqi unavailable -- LedDriver running in disabled mode (all calls no-op)"
            )
            return

        try:
            self._leds = ALProxy("ALLeds", self._nao_ip, self._nao_port)
        except Exception as exc:  # naoqi raises RuntimeError on connect fail
            self._disabled = True
            self._warn_once(
                "ALLeds proxy failed ({0}) -- LedDriver disabled".format(exc)
            )
            return

        try:
            self._player = ALProxy("ALAudioPlayer", self._nao_ip, self._nao_port)
        except Exception as exc:
            # Player is only used for chime(); LEDs still work.
            logger.warning(
                "[leds] ALAudioPlayer proxy failed (%s); chime() will be a no-op",
                exc,
            )
            self._player = None

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _warn_once(self, msg):
        """Log ``msg`` once per process when this driver enters disabled mode.

        The state machine flips presets dozens of times per minute; if we
        warned on every call the log would drown in identical lines. Once
        is enough -- the operator sees the cause at startup.
        """
        if not self._warned_disabled:
            logger.warning("[leds] %s", msg)
            self._warned_disabled = True

    def _safe_call(self, fn, *args, **kwargs):
        """Run ``fn(*args, **kwargs)`` with the proxy lock and swallow errors.

        We never want a flaky LED call to bubble out and break the wake
        state machine. Failures are logged at WARNING.
        """
        if self._disabled or self._leds is None:
            return None
        with self._lock:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                logger.warning("[leds] proxy call failed: %s", exc)
                return None

    def _fade_proxy(self, group, rgb, duration_s):
        """Issue ``ALLeds.fadeRGB``. Tries packed-int form first, then
        the float-tuple form on RuntimeError.

        See the module docstring for why both signatures matter.
        """
        if self._leds is None:
            return
        packed = _rgb_to_packed_int(rgb)
        try:
            self._leds.fadeRGB(group, packed, float(duration_s))
            return
        except Exception as primary:
            # Some firmware exposes only the (name, r, g, b, duration)
            # signature with floats. Try that before giving up.
            try:
                r, g, b = rgb
                self._leds.fadeRGB(
                    group,
                    float(r), float(g), float(b),
                    float(duration_s),
                )
                return
            except Exception as fallback:
                logger.warning(
                    "[leds] fadeRGB(%s) failed both forms: packed=%s float=%s",
                    group, primary, fallback,
                )

    # ------------------------------------------------------------------
    # public: fade + pulse
    # ------------------------------------------------------------------
    def fade(self, group, rgb, duration_s=0.4):
        """Smoothly transition ``group`` to ``rgb`` over ``duration_s`` seconds.

        ``group`` is one of the ``*_GROUP`` constants (or any ALLeds group
        name); ``rgb`` is a 3-tuple of floats in 0..1. ``duration_s`` is the
        fade ramp; 0 means an instantaneous switch (ALLeds clamps internally
        but accepts 0).

        No-op if the driver is disabled. Always returns ``None``.
        """
        if self._disabled:
            return
        # If a pulse is running on this group, fade overrides it -- stop the
        # pulse first so it doesn't keep slamming the colour back and forth.
        self.stop_pulse(group)
        # Run fadeRGB on the calling thread; the call is non-blocking from
        # ALLeds' point of view (it schedules the ramp internally).
        self._safe_call(self._fade_proxy, group, rgb, float(duration_s))

    def pulse(self, group, rgb, period_s=1.0, count=None):
        """Pulse ``group`` between ``rgb`` and 30% intensity of ``rgb``.

        Runs on a background daemon thread. ``period_s`` is the full
        on/off cycle (so each half is ``period_s / 2``). ``count`` is the
        number of full cycles; ``None`` means pulse forever until
        ``stop_pulse(group)`` is called or the driver is disposed.

        Calling ``pulse`` again on the same group cancels the prior pulse
        first -- the API is "one pulse per group at a time".

        No-op if disabled. Always returns ``None`` (the worker thread is
        managed internally; callers don't need a handle).
        """
        if self._disabled:
            return
        # Sanitise inputs; the state machine constructs callers from
        # untrusted config files in some setups.
        try:
            period_s = max(0.05, float(period_s))
        except (TypeError, ValueError):
            period_s = 1.0
        if count is not None:
            try:
                count = max(1, int(count))
            except (TypeError, ValueError):
                count = None

        # Cancel any prior pulse on this group, then register the new one.
        self.stop_pulse(group)
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._pulse_worker,
            args=(group, tuple(rgb), period_s, count, stop_event),
            name="LedPulse-{0}".format(group),
        )
        thread.daemon = True
        with self._pulses_lock:
            self._pulses[group] = (thread, stop_event)
        thread.start()

    def stop_pulse(self, group):
        """Halt the pulse running on ``group``, if any.

        Idempotent -- safe to call when no pulse is active.
        """
        with self._pulses_lock:
            entry = self._pulses.pop(group, None)
        if entry is None:
            return
        thread, stop_event = entry
        stop_event.set()
        # Don't join here from the lock-holder; the pulse thread is daemon
        # so leaving it is harmless and joining could deadlock if pulse
        # itself is holding ``self._lock``.

    def _pulse_worker(self, group, rgb, period_s, count, stop_event):
        """Background loop that alternates the LED group between high/low.

        Sleeps the half-period via ``stop_event.wait`` so a ``stop_pulse``
        cancels promptly (rather than sleeping out the full half-period).
        """
        half = period_s / 2.0
        # Each fade is half the half-period so the LED is at the target
        # colour for the second half. Tighter ramps look more like a pulse;
        # softer ramps look like a slow breathe. The Phase 3 spec doesn't
        # nail the curve so we go with snappy: 1/3 of the half.
        ramp = max(0.05, half / 3.0)
        low_rgb = _scale_rgb(rgb, _PULSE_LOW_INTENSITY)
        cycles = 0
        try:
            while not stop_event.is_set():
                # high half
                self._safe_call(self._fade_proxy, group, rgb, ramp)
                if stop_event.wait(half):
                    break
                # low half
                self._safe_call(self._fade_proxy, group, low_rgb, ramp)
                if stop_event.wait(half):
                    break
                cycles += 1
                if count is not None and cycles >= count:
                    break
        finally:
            # Clean up the registry entry if we exited because count hit
            # zero (rather than via stop_pulse which already pop()ped us).
            with self._pulses_lock:
                cur = self._pulses.get(group)
                if cur is not None and cur[1] is stop_event:
                    self._pulses.pop(group, None)

    # ------------------------------------------------------------------
    # public: state presets
    # ------------------------------------------------------------------
    def set_idle(self):
        """Eyes dim grey -- IDLE state."""
        self.fade(self.EYES_GROUP, self.COLOR_GRAY, self.DUR_IDLE)

    def set_aware(self):
        """Eyes soft blue -- AWARE state (face detected, not yet engaged)."""
        self.fade(self.EYES_GROUP, self.COLOR_SOFT_BLUE, self.DUR_AWARE)

    def set_engaged(self):
        """Eyes solid blue -- ENGAGED state (engagement gate fired)."""
        self.fade(self.EYES_GROUP, self.COLOR_SOLID_BLUE, self.DUR_ENGAGED)

    def set_listening(self):
        """Eyes cyan -- LISTENING state."""
        self.fade(self.EYES_GROUP, self.COLOR_CYAN, self.DUR_LISTENING)

    def set_speaking(self):
        """Eyes warm yellow -- SPEAKING state."""
        self.fade(self.EYES_GROUP, self.COLOR_YELLOW, self.DUR_SPEAKING)

    def set_thinking(self):
        """Eyes pulse the ENGAGED blue -- NAO is working out a reply."""
        self.pulse(self.EYES_GROUP, self.COLOR_SOLID_BLUE,
                   period_s=self.THINKING_PERIOD_S)

    def breathe(self, phase, seconds):
        """Follow one breathing phase with the eyes.

        ``inhale`` fades up to full brightness over ``seconds``, ``exhale``
        fades down to a dim glow, ``hold`` keeps the current level. Runs
        on a daemon thread: ``ALLeds.fadeRGB`` can block for the whole
        ramp, and the caller is the TTS worker, which must not stall.
        """
        if self._disabled:
            return
        try:
            seconds = max(0.2, min(self.BREATH_MAX_S, float(seconds)))
        except (TypeError, ValueError):
            seconds = 2.0
        phase = str(phase or "").lower()
        if phase == "inhale":
            target = self.COLOR_BREATH
        elif phase == "exhale":
            target = _scale_rgb(self.COLOR_BREATH, self.BREATH_LOW_INTENSITY)
        elif phase == "hold":
            # Stop any pulse so the level we reached simply stays.
            self.stop_pulse(self.EYES_GROUP)
            return
        else:
            return
        t = threading.Thread(target=self.fade,
                             args=(self.EYES_GROUP, target, seconds),
                             name="LedBreath")
        t.daemon = True
        t.start()

    # ------------------------------------------------------------------
    # public: chime
    # ------------------------------------------------------------------
    def chime(self):
        """Play the wake chime: 220 Hz sine, ~200 ms, ~80 dB.

        The WAV is generated on first call and cached at
        ``/tmp/nao_chime.wav``. ``ALAudioPlayer`` cannot play in-memory
        bytes so the disk path is required. Best-effort; logs and returns
        on any failure rather than raising.
        """
        if self._disabled or self._player is None:
            return
        if self._chime_path is None:
            self._chime_path = _ensure_chime_wav()
        if self._chime_path is None:
            return
        # Set master volume just before play so a previous TTS call that
        # turned the volume down doesn't make the chime inaudible. The
        # existing stream_tts.py uses the same belt-and-braces pattern.
        try:
            self._player.setMasterVolume(self.CHIME_VOLUME)
        except Exception as exc:
            logger.warning("[leds] setMasterVolume failed: %s", exc)
        try:
            # post.playFile is non-blocking on this firmware; the chime is
            # 200 ms and we don't want to stall the wake transition while
            # it plays. Returns a task id we don't track.
            self._player.post.playFile(self._chime_path)
        except Exception as primary:
            # Some firmware exposes only the synchronous form. Fall back.
            try:
                self._player.playFile(self._chime_path)
            except Exception as fallback:
                logger.warning(
                    "[leds] chime playFile failed: post=%s sync=%s",
                    primary, fallback,
                )

    # ------------------------------------------------------------------
    # public: shutdown
    # ------------------------------------------------------------------
    def close(self):
        """Stop all running pulses; called on graceful shutdown."""
        with self._pulses_lock:
            entries = list(self._pulses.items())
            self._pulses.clear()
        for _group, (_thread, stop_event) in entries:
            stop_event.set()


# ----------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Hand-run smoke test. Off-robot this enters disabled mode and exercises
    # every public preset + pulse + chime to prove no path raises. On the
    # robot the same script flips real eye colours, so it doubles as a
    # post-deploy sanity check.
    print("[leds self-test] starting")
    driver = LedDriver(nao_ip="127.0.0.1", nao_port=9559)
    print("[leds self-test] disabled =", driver._disabled)
    # Color presets
    driver.set_idle()
    driver.set_aware()
    driver.set_engaged()
    driver.set_listening()
    driver.set_speaking()
    # Generic fade
    driver.fade(LedDriver.CHEST_GROUP, LedDriver.COLOR_GREEN, duration_s=0.3)
    # Pulse: kick one off, then immediately cancel it.
    driver.pulse(LedDriver.EYES_GROUP, LedDriver.COLOR_CYAN, period_s=0.4, count=2)
    time.sleep(0.05)
    driver.stop_pulse(LedDriver.EYES_GROUP)
    # Chime (no-op off-robot, but should not raise)
    driver.chime()
    driver.close()
    print("[leds self-test] OK")

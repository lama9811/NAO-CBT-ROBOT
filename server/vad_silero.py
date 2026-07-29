"""Silero VAD wrapper — server-side voice activity detection.

NAO captures audio with loose energy gating; this module performs the final
endpoint check. Silero is a tiny (~1.5MB) torch model that runs comfortably
on CPU. We load it once at module import and reuse for every call.

This module exposes two layers:

1. **File / utterance API** — `has_voice(wav_path)`, `trim_silence(wav_path)`
   Used by the legacy Flask `/turn` flow (file-based).

2. **Streaming API** — `StreamingSilero`
   Used by the Phase 2 WebSocket EoU arbiter. Feed it PCM bytes as they
   arrive; ask `is_speech_now()` and `silence_duration_ms()` for live
   endpoint decisions. Optionally adaptive threshold tuned to ambient
   conditions (bimodal valley between speech / non-speech distributions).

Industry-standard params (Silero v4/v5 defaults):
- Sample rate: 16000 Hz (we resample if needed for file API)
- Streaming inference frame: 512 samples (32 ms at 16 kHz). Silero v5
  requires *exactly* 512 samples at 16 kHz per inference call — shorter
  frames raise ``ValueError("Input audio chunk is too short")`` at the
  TorchScript layer. The streaming API can therefore *accept* arbitrary
  PCM byte chunks (e.g. 30 ms / 480 samples per WS frame, per the
  Phase 2 contract) but always *consumes* in 512-sample steps from an
  internal ring buffer.
- Threshold: 0.5 default; 0.3 = more sensitive (fewer false rejects).
  We pick 0.4 — slightly more sensitive than default, since NAO already
  pre-filters by energy and we'd rather over-accept than cut users off.
- min_silence_duration_ms: 500 — pauses shorter than this stay inside one
  speech segment (handles natural breath pauses).
- speech_pad_ms: 100 — pad each side of detected segments so we don't clip
  the leading consonant or trailing fricative.

Public API:
    has_voice(wav_path)   -> bool      True if file contains any speech.
    trim_silence(wav_path) -> str      Path to trimmed WAV (head/tail silence
                                       stripped). Falls back to original on
                                       any error.
    compute_adaptive_threshold(history) -> float
                                       Pure helper; bimodal valley or 0.4.
    StreamingSilero(...)              Streaming VAD class.
"""
from __future__ import annotations

import logging
import os
import struct
import tempfile
import threading
import wave
from collections import deque
from typing import Iterable

log = logging.getLogger("sage.vad_silero")

# Silero defaults documented above.
SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # 32 ms @ 16 kHz — Silero v5 inference window
THRESHOLD = float(os.environ.get("SILERO_THRESHOLD", "0.15"))
MIN_SILENCE_MS = 500
SPEECH_PAD_MS = 100

# Adaptive threshold bounds and cadence.
ADAPTIVE_MIN = float(os.environ.get("SILERO_ADAPTIVE_MIN", "0.05"))
ADAPTIVE_MAX = 0.7
ADAPTIVE_FALLBACK = float(os.environ.get("SILERO_ADAPTIVE_FALLBACK", "0.15"))
ADAPTIVE_HISTOGRAM_BINS = 50
ADAPTIVE_RECOMPUTE_MS = 5_000  # recompute every 5 s of processed audio
ADAPTIVE_MIN_SAMPLES = 50      # need at least this much history before tuning

# Streaming inference window in milliseconds, derived from the model
# constraint above. Used for ms-bookkeeping (silence duration, history len).
INFERENCE_FRAME_MS = int(round(1000.0 * FRAME_SAMPLES / SAMPLE_RATE))  # 32

# Process-wide singleton cache for the loaded model. Loading + first-frame
# warmup costs ~200-400 ms; do it lazily on first `feed()` (or first
# `has_voice()`) so server startup stays snappy.
_model = None
_get_speech_timestamps = None
_torch = None
_load_error: str | None = None
_load_lock = threading.Lock()


def _try_load() -> bool:
    """Lazy load Silero. Returns True if model is ready.

    Thread-safe: multiple concurrent WS handlers may race on first call.
    """
    global _model, _get_speech_timestamps, _torch, _load_error
    if _model is not None:
        return True
    if _load_error is not None:
        return False
    with _load_lock:
        # double-checked locking
        if _model is not None:
            return True
        if _load_error is not None:
            return False
        try:
            import torch  # type: ignore
            _torch = torch
            # Use the standalone `silero-vad` pip package, period. The previous
            # torch.hub.load fallback pulled `snakers4/silero-vad` HEAD with
            # trust_repo=True at runtime — supply-chain risk if that repo is
            # ever compromised. The pip package is pinned in requirements.txt
            # and ships the same model weights, so dropping the fallback
            # reduces attack surface without losing capability.
            from silero_vad import load_silero_vad, get_speech_timestamps  # type: ignore
            _model = load_silero_vad()
            _get_speech_timestamps = get_speech_timestamps
            return True
        except Exception as e:  # noqa: BLE001
            _load_error = repr(e)
            log.warning(
                "silero VAD failed to load: %s. Install `silero-vad` from "
                "requirements.txt; the unsafe torch.hub fallback was removed.",
                _load_error,
            )
            return False


def _new_stream_model():
    """Return a fresh, independent Silero model for ONE streaming session.

    Each ``StreamingSilero`` owns its own model instance so concurrent sessions
    — and the reconnect churn that accumulates over a long-running server —
    cannot corrupt each other's RNN hidden state. That shared-state corruption
    was the root cause of Silero confidence collapsing to ~0 and the arbiter
    rejecting clearly-audible speech as ``no_voice`` (see CLAUDE.md). The model
    is ~1-2 MB, so a copy per session is cheap.

    Falls back to the shared singleton only if a fresh load fails, which at
    worst restores the old (buggy-but-functional) shared behavior rather than
    dropping VAD entirely.
    """
    if not _try_load():
        return None
    try:
        from silero_vad import load_silero_vad  # type: ignore
        return load_silero_vad()
    except Exception as e:  # noqa: BLE001
        log.warning(
            "vad_silero: per-session model load failed (%s); falling back to "
            "the shared model (concurrent sessions may interfere)", e,
        )
        return _model


def _read_wav_mono16k(path: str):
    """Read a WAV and return a float32 torch tensor at 16 kHz, mono.

    Returns None on unreadable / unsupported files.
    """
    try:
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            sw = w.getsampwidth()
            n = w.getnframes()
            raw = w.readframes(n)
        if sw != 2 or ch != 1 or n == 0:
            return None
        import torch  # type: ignore
        samples = struct.unpack("<{}h".format(n), raw)
        t = torch.tensor(samples, dtype=torch.float32) / 32768.0
        if sr != SAMPLE_RATE:
            # Cheap linear resample. Silero is robust to this for VAD.
            try:
                import torchaudio.functional as F  # type: ignore
                t = F.resample(t, sr, SAMPLE_RATE)
            except Exception:
                # Fallback: skip files at unexpected SR rather than crash.
                ratio = float(SAMPLE_RATE) / float(sr)
                new_len = int(len(t) * ratio)
                if new_len <= 0:
                    return None
                idx = torch.linspace(0, len(t) - 1, new_len)
                idx = idx.long().clamp(0, len(t) - 1)
                t = t[idx]
        return t
    except Exception as e:  # noqa: BLE001
        log.warning("vad_silero: read_wav failed: %s", e)
        return None


def has_voice(wav_path: str) -> bool:
    """Return True if `wav_path` contains at least one speech segment.

    Permissive on errors: if Silero can't be loaded or audio can't be parsed,
    return True so we don't drop legitimate clips.
    """
    if not _try_load():
        return True  # don't block traffic — energy gate already passed
    audio = _read_wav_mono16k(wav_path)
    if audio is None:
        return True
    try:
        ts = _get_speech_timestamps(
            audio, _model,
            sampling_rate=SAMPLE_RATE,
            threshold=THRESHOLD,
            min_silence_duration_ms=MIN_SILENCE_MS,
            speech_pad_ms=SPEECH_PAD_MS,
        )
        return bool(ts)
    except Exception as e:  # noqa: BLE001
        log.warning("vad_silero: get_speech_timestamps failed: %s", e)
        return True


def trim_silence(wav_path: str) -> str:
    """Trim leading/trailing silence from `wav_path`. Returns path to a new
    WAV. On any error, returns the original path unchanged.
    """
    if not _try_load():
        return wav_path
    audio = _read_wav_mono16k(wav_path)
    if audio is None:
        return wav_path
    try:
        ts = _get_speech_timestamps(
            audio, _model,
            sampling_rate=SAMPLE_RATE,
            threshold=THRESHOLD,
            min_silence_duration_ms=MIN_SILENCE_MS,
            speech_pad_ms=SPEECH_PAD_MS,
        )
        if not ts:
            return wav_path
        start = int(ts[0]["start"])
        end = int(ts[-1]["end"])
        if end <= start:
            return wav_path
        clipped = audio[start:end]
        # Write out as 16-bit mono @ 16 kHz.
        pcm = (clipped.clamp(-1.0, 1.0) * 32767.0).to(_torch.int16).tolist()
        out_path = tempfile.NamedTemporaryFile(
            suffix="_silero.wav", delete=False
        ).name
        with wave.open(out_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(struct.pack("<{}h".format(len(pcm)), *pcm))
        return out_path
    except Exception as e:  # noqa: BLE001
        log.warning("vad_silero: trim_silence failed: %s", e)
        return wav_path


# ---------------------------------------------------------------------------
# Adaptive threshold helper (pure, unit-testable).
# ---------------------------------------------------------------------------

def compute_adaptive_threshold(
    history: Iterable[float],
    *,
    bins: int = ADAPTIVE_HISTOGRAM_BINS,
    fallback: float = ADAPTIVE_FALLBACK,
    lo: float = ADAPTIVE_MIN,
    hi: float = ADAPTIVE_MAX,
) -> float:
    """Pick a threshold at the valley between the two highest peaks in
    a 50-bin histogram of recent VAD confidences on ``[0, 1]``.

    Bimodal speech/non-speech distributions look like::

        non-speech peak                 speech peak
              |                              |
              v                              v
        ##### ##### ##                  ##### ##### #####
              |                              |
              +--------- valley ------------+
                            ^
                          chosen threshold

    If the histogram doesn't have two clear peaks (single-peak or empty
    history), fall back to ``fallback`` (default 0.4). Result is clamped
    to ``[lo, hi]`` (default ``[0.2, 0.7]``) to avoid pathological auto-tunes
    when ambient is unusual.

    Pure function — no I/O, no model calls — so unit tests can pin behavior
    without loading torch.
    """
    # Coerce iterable -> list once; reject obviously degenerate inputs.
    samples = [float(x) for x in history if 0.0 <= float(x) <= 1.0]
    if len(samples) < ADAPTIVE_MIN_SAMPLES:
        return _clamp(fallback, lo, hi)

    # Histogram — 50 bins on [0, 1]. Use simple integer-bin bookkeeping
    # rather than numpy to avoid pulling another dep just for this helper.
    if bins < 4:
        return _clamp(fallback, lo, hi)
    counts = [0] * bins
    for v in samples:
        # bin index in [0, bins-1]
        idx = int(v * bins)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1

    # Find local-maximum bins: count strictly greater than both neighbours
    # (treat off-edge neighbours as -infinity so edge bins can qualify).
    # A "flat top" in one mode (multiple equal adjacent bins) won't trigger
    # the strict inequality, which is what we want — we don't want flat
    # tops to count as multiple peaks within the same mode.
    peaks: list[tuple[int, int]] = []  # (count, bin_idx)
    for i in range(bins):
        left = counts[i - 1] if i > 0 else -1
        right = counts[i + 1] if i + 1 < bins else -1
        if counts[i] > left and counts[i] > right and counts[i] > 0:
            peaks.append((counts[i], i))
    if len(peaks) < 2:
        return _clamp(fallback, lo, hi)

    # Pick the top-1 peak, then find the highest *distant* peak.
    # "Distant" = at least bins/8 apart (~6 bins on the default 50-bin
    # histogram), so a multi-bin plateau within a single mode can't
    # masquerade as the second peak.
    peaks.sort(reverse=True)
    p1_count, p1_bin = peaks[0]
    min_separation = max(2, bins // 8)
    p2: tuple[int, int] | None = None
    for cnt, idx in peaks[1:]:
        if abs(idx - p1_bin) >= min_separation:
            p2 = (cnt, idx)
            break
    if p2 is None:
        return _clamp(fallback, lo, hi)
    p2_count, p2_bin = p2

    # Require some peak prominence. If the second peak is too tiny relative
    # to the first, the distribution is effectively unimodal — fall back.
    if p2_count * 4 < p1_count:
        return _clamp(fallback, lo, hi)

    # Valley = bin with smallest count strictly between the two peaks.
    lo_bin, hi_bin = sorted((p1_bin, p2_bin))
    valley_bin = lo_bin + 1
    valley_count = counts[valley_bin]
    for j in range(lo_bin + 1, hi_bin):
        if counts[j] < valley_count:
            valley_bin = j
            valley_count = counts[j]

    # Convert valley bin centre to a threshold in [0, 1].
    threshold = (valley_bin + 0.5) / bins
    return _clamp(threshold, lo, hi)


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


# ---------------------------------------------------------------------------
# Streaming Silero — incremental VAD for the WS EoU arbiter.
# ---------------------------------------------------------------------------

class StreamingSilero:
    """Streaming Silero VAD. Feed it 30 ms PCM @ 16 kHz; ask is_speech_now().

    The class accepts arbitrary 16-bit little-endian PCM byte chunks
    (so the WS handler doesn't have to align to model frame size). It
    buffers internally and runs Silero inference in 512-sample windows
    (~32 ms at 16 kHz — the v5 model's required minimum). Per-frame
    inference cost on CPU is ~0.2-0.5 ms, comfortably under the 5 ms
    budget for a 30 ms input frame.

    Usage::

        vad = StreamingSilero()
        vad.feed(pcm_bytes_30ms)        # repeat as frames arrive
        if not vad.is_speech_now():
            silence_ms = vad.silence_duration_ms()

    All public methods are safe to call from a single async task. The
    class is not thread-safe across tasks — instantiate one per session.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = 30,
        threshold: float = THRESHOLD,
        adaptive: bool = True,
        history_seconds: float = 60.0,
    ) -> None:
        if sample_rate != SAMPLE_RATE:
            # Silero v5 supports 8 kHz and 16 kHz; we only run 16 kHz here
            # because every NAO/WS path resamples to 16 kHz upstream.
            raise ValueError(
                "StreamingSilero only supports 16 kHz; got {}".format(sample_rate)
            )
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.threshold = threshold
        self.adaptive = adaptive
        self.history_seconds = history_seconds

        # Confidence history sized by *inference* frames, not input frames,
        # because that's what the histogram bins. INFERENCE_FRAME_MS is
        # ~32 ms (512 samples @ 16 kHz).
        history_len = max(
            ADAPTIVE_MIN_SAMPLES * 2,
            int(history_seconds * 1000.0 / INFERENCE_FRAME_MS),
        )
        self._confidence_history: deque[float] = deque(maxlen=history_len)

        # PCM ring buffer (raw int16 samples). bytearray gives us cheap
        # append + slice semantics; we drain in 512-sample (1024-byte) chunks.
        self._pcm_buffer = bytearray()

        # Bookkeeping. All time accounting is monotonic and audio-driven
        # (frames * INFERENCE_FRAME_MS), so unit tests are deterministic.
        self._frames_processed = 0
        self._frames_dropped = 0  # decoded chunks that errored out
        self._last_speech_at_ms: int | None = None
        self._last_silence_at_ms: int | None = None
        self._adaptive_threshold = threshold
        self._last_adaptive_recompute_ms = 0
        self._is_speech_now = False

        # Per-session Silero model. Lazily created on first feed() so
        # construction stays cheap and load failures degrade gracefully.
        # Owning our own instance (rather than the module singleton) is what
        # keeps concurrent sessions from corrupting each other's RNN state.
        self._model = None

    # ---- public API --------------------------------------------------

    def feed(self, pcm_bytes: bytes) -> None:
        """Append raw 16-bit little-endian PCM to the internal buffer
        and drain as many 512-sample inference windows as fit.

        The caller is free to pass any chunk size (30 ms / 480 samples
        per the Phase 2 contract is typical). We accumulate and run
        inference at 512-sample boundaries.
        """
        if not pcm_bytes:
            return
        if not _try_load():
            # Lazy-load failure — silently skip so callers aren't broken.
            # The adaptive threshold will stay at its default value.
            return

        if self._model is None:
            # First real audio for this session — bind our own model instance.
            self._model = _new_stream_model()

        self._pcm_buffer.extend(pcm_bytes)
        bytes_per_frame = FRAME_SAMPLES * 2  # int16 little-endian

        while len(self._pcm_buffer) >= bytes_per_frame:
            chunk = bytes(self._pcm_buffer[:bytes_per_frame])
            del self._pcm_buffer[:bytes_per_frame]
            self._infer_one_frame(chunk)

        # Adaptive threshold recompute on a 5 s audio cadence (driven by
        # frames_processed * INFERENCE_FRAME_MS, not wall clock).
        now_ms = self._frames_processed * INFERENCE_FRAME_MS
        if (
            self.adaptive
            and now_ms - self._last_adaptive_recompute_ms >= ADAPTIVE_RECOMPUTE_MS
        ):
            self._adaptive_threshold = compute_adaptive_threshold(
                list(self._confidence_history),
                fallback=self.threshold,
            )
            self._last_adaptive_recompute_ms = now_ms

    def is_speech_now(self) -> bool:
        """True if the most recent inference frame's confidence exceeded
        the current threshold.

        Returns False until at least one inference frame has been processed.
        """
        return self._is_speech_now

    def silence_duration_ms(self) -> int:
        """Milliseconds since the most recent frame whose confidence
        crossed the current threshold.

        Returns 0 when no frame has been processed yet (no observation
        means we can't report silence). After the first speech frame, the
        counter monotonically grows until the next speech frame resets it.
        """
        if self._frames_processed == 0:
            return 0
        now_ms = self._frames_processed * INFERENCE_FRAME_MS
        if self._last_speech_at_ms is None:
            # Never seen speech in this session — silence is everything we've
            # processed.
            return now_ms
        return max(0, now_ms - self._last_speech_at_ms)

    def reset(self) -> None:
        """Clear buffers, history, and per-stream state. Keep the loaded
        model and reset its internal RNN state so the next stream starts
        cleanly."""
        self._confidence_history.clear()
        self._pcm_buffer = bytearray()
        self._frames_processed = 0
        self._frames_dropped = 0
        self._last_speech_at_ms = None
        self._last_silence_at_ms = None
        self._adaptive_threshold = self.threshold
        self._last_adaptive_recompute_ms = 0
        self._is_speech_now = False
        # Reset THIS session's own model, never the shared singleton — that
        # cross-session wipe was the root cause of the confidence-collapse bug.
        m = self._model
        if m is not None:
            try:
                m.reset_states()  # Silero v5 RNN reset
            except Exception as e:  # noqa: BLE001
                log.debug("silero reset_states failed (non-fatal): %s", e)

    def current_threshold(self) -> float:
        """Return the threshold currently used by `is_speech_now`.

        When `adaptive=True`, this is the bimodal-valley estimate from the
        last 60 s of confidences (recomputed every 5 s of audio); otherwise
        the static `threshold` constructor argument.
        """
        return self._adaptive_threshold if self.adaptive else self.threshold

    def stats(self) -> dict:
        """Lightweight diagnostics for /metrics or DEBUG logs."""
        last5s = self._mean_last_n_seconds(5.0)
        return {
            "current_threshold": round(self.current_threshold(), 4),
            "last5s_mean_confidence": (
                round(last5s, 4) if last5s is not None else None
            ),
            "frames_processed": self._frames_processed,
            "frames_dropped": self._frames_dropped,
            "history_len": len(self._confidence_history),
            "buffer_bytes": len(self._pcm_buffer),
            "adaptive": self.adaptive,
        }

    # ---- internals ---------------------------------------------------

    def _infer_one_frame(self, chunk: bytes) -> None:
        """Run one Silero step on exactly FRAME_SAMPLES of int16 PCM."""
        try:
            import torch  # type: ignore
            # Use this session's own model; fall back to the shared singleton
            # only if per-session load failed (degraded, not broken).
            model = self._model if self._model is not None else _model
            samples = struct.unpack("<{}h".format(FRAME_SAMPLES), chunk)
            tensor = torch.tensor(samples, dtype=torch.float32) / 32768.0
            with torch.no_grad():
                conf = float(model(tensor, SAMPLE_RATE))
        except Exception as e:  # noqa: BLE001
            self._frames_dropped += 1
            log.debug("vad_silero: streaming inference failed: %s", e)
            return

        self._frames_processed += 1
        now_ms = self._frames_processed * INFERENCE_FRAME_MS
        self._confidence_history.append(conf)

        thr = self.current_threshold()
        speaking = conf > thr
        self._is_speech_now = speaking
        if speaking:
            self._last_speech_at_ms = now_ms
        else:
            self._last_silence_at_ms = now_ms

        # Diagnostic: every ~32 frames (~1 s of audio) log peak/mean conf.
        # Lets us see whether Silero is reading audio at all and what
        # confidence values it's producing — invaluable when the arbiter
        # keeps rejecting turns as no_voice despite clearly-audible PCM.
        if self._frames_processed % 32 == 0:
            window = list(self._confidence_history)[-32:]
            if window:
                peak = max(window)
                mean = sum(window) / len(window)
                print(
                    "[silero_trace] frames={0} peak={1:.3f} mean={2:.3f} "
                    "thr={3:.3f} speaking={4}".format(
                        self._frames_processed, peak, mean, thr, speaking
                    ),
                    flush=True,
                )

    def _mean_last_n_seconds(self, n: float) -> float | None:
        if not self._confidence_history:
            return None
        n_frames = max(1, int(n * 1000.0 / INFERENCE_FRAME_MS))
        if n_frames >= len(self._confidence_history):
            window = list(self._confidence_history)
        else:
            # deque has no slice; pull the tail efficiently.
            window = list(self._confidence_history)[-n_frames:]
        if not window:
            return None
        return sum(window) / len(window)


# ---------------------------------------------------------------------------
# Self-test entrypoint. `python server/vad_silero.py` exercises the adaptive
# helper on synthetic histograms and runs a 1 s zero-PCM smoke through the
# streaming class.
# ---------------------------------------------------------------------------

def _self_test() -> int:
    print("== compute_adaptive_threshold ==")

    # Single-peak distribution -> fallback.
    single_peak = [0.1] * 200 + [0.11] * 200 + [0.09] * 200
    t = compute_adaptive_threshold(single_peak)
    print("single-peak ->", round(t, 3), "(expect fallback 0.4)")
    assert abs(t - ADAPTIVE_FALLBACK) < 1e-9

    # Clear bimodal at 0.2 / 0.7 -> valley somewhere between them.
    bimodal = []
    for _ in range(400):
        bimodal.append(0.18)
        bimodal.append(0.22)
    for _ in range(400):
        bimodal.append(0.68)
        bimodal.append(0.72)
    t = compute_adaptive_threshold(bimodal)
    print("bimodal 0.2/0.7 ->", round(t, 3), "(expect roughly mid)")
    assert 0.25 <= t <= 0.65, "valley should lie strictly between peaks"

    # All-zero history -> too few samples / single peak -> fallback.
    t = compute_adaptive_threshold([])
    print("empty ->", round(t, 3), "(expect fallback)")
    assert t == ADAPTIVE_FALLBACK
    t = compute_adaptive_threshold([0.0] * 1000)
    print("all-zero ->", round(t, 3), "(expect fallback)")
    assert t == ADAPTIVE_FALLBACK

    # Pathological tuned-out distribution should clamp to [0.2, 0.7].
    extreme = []
    for _ in range(400):
        extreme.append(0.01)
    for _ in range(400):
        extreme.append(0.99)
    t = compute_adaptive_threshold(extreme)
    print("extreme bimodal 0.01/0.99 ->", round(t, 3), "(expect clamped)")
    assert ADAPTIVE_MIN <= t <= ADAPTIVE_MAX

    print("\n== StreamingSilero smoke ==")
    if not _try_load():
        print("silero unavailable; skipping streaming smoke")
        return 0

    vad = StreamingSilero()
    # 1 s of zero PCM, fed in 30 ms (480 sample = 960 byte) chunks.
    chunk_30ms = b"\x00\x00" * 480
    for _ in range(int(1000 / 30)):
        vad.feed(chunk_30ms)
    s = vad.silence_duration_ms()
    print("silence after 1 s of zeros:", s, "ms")
    print("stats:", vad.stats())
    assert s > 950, "expected silence_duration_ms > 950 after zero PCM"

    print("\nself-test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

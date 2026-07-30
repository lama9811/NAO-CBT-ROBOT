"""The voice gate must not drop real speech just because the user paused first.

``sess.audio_buf`` accumulates every PCM chunk from the moment the previous
turn was finalized, so the WAV handed to ``has_voice`` is *thinking pause +
utterance*. The webrtcvad half of the gate scored a **ratio** of voiced frames
over that whole window, so the same sentence flipped from accepted to
``no_voice`` purely as a function of how long the user took to start talking.
Measured on the Pi with its own webrtcvad build (0.5 s of voiced audio):

    pad=1 s → ratio 0.420  ACCEPT
    pad=2 s → ratio 0.253  ACCEPT
    pad=4 s → ratio 0.140  REJECT
    pad=8 s → ratio 0.074  REJECT

A rejected turn produces no reply at all, which is what "NAO goes quiet on me"
looks like from the other side of the room — and it bites hardest in exactly
the slow, reflective conversations the therapist lane is for.

webrtcvad does not import on the dev Mac (Python 3.13 / pkg_resources), so the
tests inject a stub detector: frames of digital silence are unvoiced, anything
else is voiced. That is the same shape as the Pi measurement above.
"""
from __future__ import annotations

import struct
import wave


SR = 16000


def _write_wav(path, silence_s: float, voiced_s: float) -> str:
    """WAV of `silence_s` digital silence followed by `voiced_s` of tone."""
    n_sil = int(SR * silence_s)
    n_voiced = int(SR * voiced_s)
    frames = [0] * n_sil + [(6000 if (i // 8) % 2 else -6000)
                            for i in range(n_voiced)]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(struct.pack("<{}h".format(len(frames)), *frames))
    return str(path)


class _StubVad:
    """Stand-in for webrtcvad.Vad: non-silent frame == speech."""

    def __init__(self, aggressiveness: int = 2) -> None:
        self.aggressiveness = aggressiveness

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        return any(frame)


class _StubWebrtcvad:
    Vad = _StubVad


def _install_stub_vad(monkeypatch):
    from server import _legacy_helpers as L
    from server import vad_silero

    monkeypatch.setattr(L, "webrtcvad", _StubWebrtcvad, raising=False)
    monkeypatch.setattr(L, "_VAD_AVAILABLE", True)
    # Silero is the other half of the gate; keep it out of the way — these
    # tests are about the frame-ratio arithmetic, not the model.
    monkeypatch.setattr(vad_silero, "has_voice", lambda path: True)
    return L


def test_speech_after_a_long_pause_is_accepted(tmp_path, monkeypatch):
    """A one-second answer inside a 7.5 s buffer is still an answer.

    Ratio here is 1.0/7.5 = 0.133, under the old 0.18 floor — the shape of the
    turns the live Pi logged as `no_voice` with ~7.5-8.5 s buffers.
    """
    L = _install_stub_vad(monkeypatch)
    wav = _write_wav(tmp_path / "late_speech.wav", silence_s=6.5, voiced_s=1.0)

    assert L.has_voice(wav) is True


def test_speech_right_away_is_still_accepted(tmp_path, monkeypatch):
    """The short-buffer case that already worked must keep working."""
    L = _install_stub_vad(monkeypatch)
    wav = _write_wav(tmp_path / "prompt_speech.wav", silence_s=0.2, voiced_s=1.5)

    assert L.has_voice(wav) is True


def test_pure_silence_is_still_rejected(tmp_path, monkeypatch):
    """Loosening the gate must not let an empty room through."""
    L = _install_stub_vad(monkeypatch)
    wav = _write_wav(tmp_path / "empty_room.wav", silence_s=8.0, voiced_s=0.0)

    assert L.has_voice(wav) is False


def test_a_single_click_is_still_rejected(tmp_path, monkeypatch):
    """A 60 ms transient is not an utterance — below the absolute floor."""
    L = _install_stub_vad(monkeypatch)
    wav = _write_wav(tmp_path / "click.wav", silence_s=8.0, voiced_s=0.06)

    assert L.has_voice(wav) is False

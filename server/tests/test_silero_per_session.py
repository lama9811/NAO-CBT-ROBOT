"""Per-session Silero VAD isolation — regression for the shared-model bug.

Root cause (documented in CLAUDE.md): ``vad_silero._model`` was a process-wide
stateful RNN shared by every ``StreamingSilero``. Concurrent sessions interleaved
audio into one hidden state, and one session's ``reset()`` called
``_model.reset_states()`` and wiped another's mid-turn. Silero confidence
collapsed to ~0 and the arbiter rejected clearly-audible speech as ``no_voice``.
It got worse as WebSocket reconnects accumulated.

The fix gives each ``StreamingSilero`` its own model instance (~1-2 MB, cheap).
These tests pin that isolation so the shared-model regression can't return.

Batch helpers (``has_voice`` / ``trim_silence``) intentionally still share the
module singleton — they are stateless one-shot calls, so they are unaffected.
"""
from __future__ import annotations

import struct

import pytest

pytest.importorskip("torch")
pytest.importorskip("silero_vad")

from server import vad_silero as V  # noqa: E402


def _pcm(nsamples: int, value: int = 1000) -> bytes:
    """Non-zero 16-bit LE PCM so ``feed`` runs real inference frames."""
    return struct.pack("<{}h".format(nsamples), *([value] * nsamples))


def _feed_a_bit(vad: "V.StreamingSilero") -> None:
    # several 512-sample inference windows so the per-session model loads
    vad.feed(_pcm(V.FRAME_SAMPLES * 4))


def test_each_stream_has_its_own_model():
    a = V.StreamingSilero()
    b = V.StreamingSilero()
    _feed_a_bit(a)
    _feed_a_bit(b)
    assert a._model is not None and b._model is not None, (
        "per-session model was not loaded on feed()"
    )
    assert a._model is not b._model, (
        "StreamingSilero instances share one model — concurrent sessions "
        "will corrupt each other's RNN state (the no_voice bug)"
    )


def test_reset_on_one_does_not_touch_another():
    a = V.StreamingSilero()
    b = V.StreamingSilero()
    _feed_a_bit(a)
    _feed_a_bit(b)
    a_model, b_model = a._model, b._model
    b.reset()  # must reset ONLY b's model, never rebind or touch a's
    assert a._model is a_model, "reset() on b disturbed a's model"
    assert b._model is b_model, "reset() rebound b's own model object"
    assert a._model is not b._model, "models still must be independent after reset"

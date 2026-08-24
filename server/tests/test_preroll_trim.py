"""Pre-speech silence must not accumulate in the turn buffer.

Root cause of "NAO takes a minute to answer": ``_ingest`` appended every
inbound PCM chunk to ``sess.audio_buf`` from the moment the mic opened, and
stamped ``utterance_start_ms`` on the *first chunk* rather than on speech
onset. A user who sat quietly for 50 s before speaking had all 50 s of silence
prepended to their one sentence, so:

  * the clip shipped to Deepgram was ~53 s of mostly silence — nova-3 returned
    empty and the nova-2 retry latched onto faint background speech
    (observed live: transcript='Alright. You can sign in over here.');
  * the 60 s hard ceiling measured from mic-open, so a slow starter got cut
    off mid-sentence instead of 60 s into actual speech.

The fix keeps only a short pre-roll while no speech has been detected, so the
onset is never clipped but the silence never piles up.
"""
from __future__ import annotations

import importlib


A = importlib.import_module("server.app_ws")


BYTES_PER_MS = 32  # 16 kHz, 16-bit, mono


def _silence(ms: int) -> bytes:
    return b"\x00\x00" * (16 * ms)


def test_trim_preroll_caps_buffer_at_preroll_window():
    """A long silent lead-in collapses to the pre-roll window."""
    buf = bytearray(_silence(50_000))          # 50 s of silence
    A._trim_preroll(buf)
    assert len(buf) <= A.EOU_PREROLL_MS * BYTES_PER_MS
    # And it keeps the *tail*, so the moment before speech survives.
    assert len(buf) == A.EOU_PREROLL_MS * BYTES_PER_MS


def test_trim_preroll_leaves_short_buffers_untouched():
    """Nothing is dropped before the buffer exceeds the pre-roll window."""
    short = _silence(100)
    buf = bytearray(short)
    A._trim_preroll(buf)
    assert bytes(buf) == short


def test_trim_preroll_keeps_the_most_recent_audio():
    """The retained bytes are the tail, not the head."""
    head = b"\x01\x01" * (16 * 5_000)          # 5 s marked 0x0101
    tail = b"\x02\x02" * (16 * 1_000)          # 1 s marked 0x0202
    buf = bytearray(head + tail)
    A._trim_preroll(buf)
    assert bytes(buf).endswith(tail[-64:])
    assert b"\x01\x01" not in bytes(buf)[-len(tail):]

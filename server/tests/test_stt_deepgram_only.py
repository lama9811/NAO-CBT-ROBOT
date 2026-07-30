"""Deepgram owns STT end to end — including the retry when the first pass is empty.

Two behaviours are pinned here:

1. `deepgram_asr.transcribe` retries on a second Deepgram model when the primary
   returns an empty transcript, so the fallback is still Deepgram rather than a
   different vendor.
2. `_legacy_helpers.transcribe` no longer falls through to OpenAI Whisper by
   default. That fallback is why NAO answered sentences the user never said:
   Whisper never returns empty on weak audio — it invents words ("Hallo", and a
   Japanese sentence out of pure room noise) — while Deepgram honestly reports
   silence. A turn with no transcript is rejected and NAO stays quiet, which is
   the better failure. Set STT_ALLOW_OPENAI_FALLBACK=1 to restore the old chain
   if Deepgram ever has an outage.
"""
from __future__ import annotations

import pytest


class _Resp:
    def __init__(self, transcript: str, status: int = 200) -> None:
        self.status_code = status
        self._transcript = transcript
        self.text = ""

    def json(self) -> dict:
        return {"results": {"channels": [
            {"alternatives": [{"transcript": self._transcript}]}]}}


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "turn.wav"
    p.write_bytes(b"RIFF____WAVEfmt ")
    return str(p)


def _stub_deepgram(monkeypatch, transcripts_by_model):
    """Route each Deepgram model to a canned transcript; record call order."""
    from server import config, deepgram_asr

    monkeypatch.setattr(config, "DEEPGRAM_API_KEY", "dg-test-key")
    monkeypatch.setattr(config, "DEEPGRAM_MODEL", "nova-3")
    monkeypatch.setattr(config, "DEEPGRAM_LANGUAGE", "en-US")
    calls: list[str] = []

    def _fake_post(url, headers=None, data=None, timeout=None):
        model = url.split("model=")[1].split("&")[0]
        calls.append(model)
        return _Resp(transcripts_by_model.get(model, ""))

    monkeypatch.setattr(deepgram_asr.requests, "post", _fake_post)
    return calls


def test_retries_on_second_deepgram_model_when_primary_is_empty(monkeypatch, wav):
    from server import deepgram_asr

    calls = _stub_deepgram(monkeypatch, {"nova-3": "", "nova-2": "hello there"})

    assert deepgram_asr.transcribe(wav) == "hello there"
    assert calls == ["nova-3", "nova-2"], calls


def test_no_retry_when_the_primary_model_answers(monkeypatch, wav):
    """The retry costs a whole extra round-trip — only pay it on an empty result."""
    from server import deepgram_asr

    calls = _stub_deepgram(monkeypatch, {"nova-3": "hello there", "nova-2": "wrong"})

    assert deepgram_asr.transcribe(wav) == "hello there"
    assert calls == ["nova-3"], calls


def test_silence_stays_silence(monkeypatch, wav):
    from server import deepgram_asr

    _stub_deepgram(monkeypatch, {"nova-3": "", "nova-2": ""})

    assert deepgram_asr.transcribe(wav) == ""


def test_pipeline_does_not_reach_openai_by_default(monkeypatch, wav):
    """An empty Deepgram result must not become a Whisper hallucination."""
    from server import _legacy_helpers as L
    from server import config

    monkeypatch.setattr(config, "USE_DEEPGRAM", True)
    monkeypatch.setattr(config, "USE_ELEVENLABS_STT", False)
    monkeypatch.setattr(config, "STT_ALLOW_OPENAI_FALLBACK", False, raising=False)
    monkeypatch.setattr(L, "_deepgram_transcribe", lambda path: "")

    def _boom(*a, **kw):  # pragma: no cover - must never run
        raise AssertionError("OpenAI transcription was called despite the flag")

    monkeypatch.setattr(L._client.audio.transcriptions, "create", _boom)

    assert L.transcribe(wav) == ""


def test_openai_fallback_restorable_by_flag(monkeypatch, wav):
    """One env var brings the old chain back if Deepgram has an outage."""
    from server import _legacy_helpers as L
    from server import config

    monkeypatch.setattr(config, "USE_DEEPGRAM", True)
    monkeypatch.setattr(config, "USE_ELEVENLABS_STT", False)
    monkeypatch.setattr(config, "STT_ALLOW_OPENAI_FALLBACK", True, raising=False)
    monkeypatch.setattr(L, "_deepgram_transcribe", lambda path: "")

    class _Whisper:
        text = "whisper heard this"

    monkeypatch.setattr(
        L._client.audio.transcriptions, "create", lambda **kw: _Whisper()
    )

    assert L.transcribe(wav) == "whisper heard this"


def test_unheard_turn_is_saved_when_dump_dir_is_set(monkeypatch, tmp_path, wav):
    """The failing audio is the one artifact that settles 'NAO didn't hear me'."""
    from server import _legacy_helpers as L
    from server import config

    dump = tmp_path / "unheard"
    monkeypatch.setenv("STT_DEBUG_DUMP_DIR", str(dump))
    monkeypatch.setattr(config, "USE_DEEPGRAM", True)
    monkeypatch.setattr(config, "USE_ELEVENLABS_STT", False)
    monkeypatch.setattr(config, "STT_ALLOW_OPENAI_FALLBACK", False, raising=False)
    monkeypatch.setattr(L, "_deepgram_transcribe", lambda path: "")

    assert L.transcribe(wav) == ""
    assert list(dump.glob("unheard-*.wav")), "no audio kept for the failing turn"


def test_nothing_saved_when_dump_dir_unset(monkeypatch, tmp_path, wav):
    from server import _legacy_helpers as L
    from server import config

    monkeypatch.delenv("STT_DEBUG_DUMP_DIR", raising=False)
    monkeypatch.setattr(config, "USE_DEEPGRAM", True)
    monkeypatch.setattr(config, "USE_ELEVENLABS_STT", False)
    monkeypatch.setattr(config, "STT_ALLOW_OPENAI_FALLBACK", False, raising=False)
    monkeypatch.setattr(L, "_deepgram_transcribe", lambda path: "")

    assert L.transcribe(wav) == ""
    assert not list(tmp_path.glob("**/unheard-*.wav"))

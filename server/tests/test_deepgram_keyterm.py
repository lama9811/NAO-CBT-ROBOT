"""Nova-3 renamed the term-boosting parameter, and the old name is a hard 400.

`deepgram_asr` sent `keywords`, which Nova-3 rejects outright:

    400 INVALID_QUERY_PARAMETER
    "Keywords are not supported for Nova-3. Please use `keyterm` instead."

The adapter swallows non-200s and returns "", and `transcribe()` treats "" as
"provider had nothing" and falls through to the next one — so the failure was
completely silent: every turn paid a wasted Deepgram round-trip and was then
transcribed by Whisper, while the logs showed Deepgram enabled and healthy.

These tests pin the parameter name to the model family so the next model bump
fails loudly in CI instead of quietly on the robot.
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest


class _FakeResponse:
    status_code = 200

    @staticmethod
    def json() -> dict:
        return {"results": {"channels": [{"alternatives": [
            {"transcript": "hello there"}]}]}}


def _capture_url(monkeypatch, tmp_path, model: str) -> str:
    """Run transcribe() against a stubbed POST and return the URL it built."""
    from server import config, deepgram_asr

    monkeypatch.setattr(config, "DEEPGRAM_API_KEY", "dg-test-key")
    monkeypatch.setattr(config, "DEEPGRAM_MODEL", model)
    monkeypatch.setattr(config, "DEEPGRAM_LANGUAGE", "en-US")

    seen: dict[str, str] = {}

    def _fake_post(url, headers=None, data=None, timeout=None):
        seen["url"] = url
        return _FakeResponse()

    monkeypatch.setattr(deepgram_asr.requests, "post", _fake_post)

    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF____WAVEfmt ")

    assert deepgram_asr.transcribe(str(wav)) == "hello there"
    return seen["url"]


def test_nova3_boosts_terms_with_keyterm(monkeypatch, tmp_path):
    """Nova-3 must use `keyterm`; `keywords` is a 400 on every request."""
    qs = parse_qs(urlparse(_capture_url(monkeypatch, tmp_path, "nova-3")).query)

    assert "keywords" not in qs
    assert qs.get("keyterm"), "nova-3 must send keyterm"
    assert "Morgan" in " ".join(qs["keyterm"])


def test_nova2_keeps_keywords(monkeypatch, tmp_path):
    """Older models never learned `keyterm` — don't break them on the way past."""
    qs = parse_qs(urlparse(_capture_url(monkeypatch, tmp_path, "nova-2")).query)

    assert "keyterm" not in qs
    assert qs.get("keywords"), "nova-2 must still send keywords"


@pytest.mark.parametrize("model", ["nova-3", "nova-2"])
def test_keyterm_values_carry_no_weight_suffix(monkeypatch, tmp_path, model):
    """`keyterm` takes bare phrases; the `word:weight` syntax is keywords-only."""
    qs = parse_qs(urlparse(_capture_url(monkeypatch, tmp_path, model)).query)

    for value in qs.get("keyterm", []):
        assert ":" not in value, f"keyterm must not carry a weight suffix: {value}"

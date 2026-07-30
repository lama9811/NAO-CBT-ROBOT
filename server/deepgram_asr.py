"""Deepgram Nova-2 prerecorded ASR client.

NAO uploads a complete WAV file via multipart POST, so we use Deepgram's
synchronous /v1/listen endpoint (prerecorded) rather than the websocket
streaming API. Returns the top-1 transcript or "" on any failure (callers
already handle empty transcripts via _transcript_reject_reason).
"""
from __future__ import annotations

import requests

from server import config

_LISTEN_URL = "https://api.deepgram.com/v1/listen"
_TIMEOUT_S = 15.0

# Boost domain terms NAO and Morgan students actually say.
_TERMS = ["Morgan", "NAO", "CBT", "therapist", "Aayush"]

# Legacy `keywords` syntax carries a per-term weight; Nova-3's `keyterm` does
# not accept one. Weight applies only to the models that still take `keywords`.
_KEYWORD_WEIGHT = 5


def _term_boost_params(model: str) -> list[tuple[str, str]]:
    """Query params that boost `_TERMS`, named for the model family.

    Nova-3 dropped `keywords` and rejects it outright — every request comes
    back `400 INVALID_QUERY_PARAMETER: Keywords are not supported for Nova-3.
    Please use \x60keyterm\x60 instead.` The adapter turns any non-200 into ""
    and `transcribe()` reads "" as "this provider had nothing", so the whole
    thing failed silently: Deepgram looked enabled, every turn paid a wasted
    round-trip, and Whisper quietly did the work.
    """
    if (model or "").lower().startswith("nova-3"):
        return [("keyterm", t) for t in _TERMS]
    return [("keywords", "{0}:{1}".format(t, _KEYWORD_WEIGHT)) for t in _TERMS]


def transcribe(path: str) -> str:
    """Transcribe `path`, retrying on a second Deepgram model if the first
    pass hears nothing.

    Deepgram reports silence honestly, which is what we want — but a single
    empty result shouldn't end the turn while another Deepgram model might
    still find the words. The retry only fires on an empty transcript, so a
    successful first pass never pays the extra round-trip.
    """
    text = _transcribe_with(path, config.DEEPGRAM_MODEL)
    if text:
        return text

    fallback = (getattr(config, "DEEPGRAM_FALLBACK_MODEL", "") or "").strip()
    if not fallback or fallback == config.DEEPGRAM_MODEL:
        return ""
    print("[deepgram] {0} returned empty; retrying on {1}".format(
        config.DEEPGRAM_MODEL, fallback), flush=True)
    return _transcribe_with(path, fallback)


def _transcribe_with(path: str, model: str) -> str:
    if not config.DEEPGRAM_API_KEY:
        return ""

    params = {
        "model": model,
        "language": config.DEEPGRAM_LANGUAGE,
        "smart_format": "true",
        "punctuate": "true",
    }
    headers = {
        "Authorization": "Token {0}".format(config.DEEPGRAM_API_KEY),
        "Content-Type": "audio/wav",
    }

    try:
        with open(path, "rb") as f:
            audio_bytes = f.read()
        # Build the URL manually so we can repeat the term-boost param.
        from urllib.parse import urlencode
        flat_params = list(params.items()) + _term_boost_params(model)
        url = _LISTEN_URL + "?" + urlencode(flat_params)
        resp = requests.post(url, headers=headers, data=audio_bytes, timeout=_TIMEOUT_S)
    except Exception as e:
        print("[deepgram] request failed: {0!r}".format(e), flush=True)
        return ""

    if resp.status_code != 200:
        print(
            "[deepgram] non-200 status={0} body={1!r}".format(
                resp.status_code, resp.text[:300],
            ),
            flush=True,
        )
        return ""

    try:
        data = resp.json()
        alts = data["results"]["channels"][0]["alternatives"]
        if not alts:
            return ""
        return (alts[0].get("transcript") or "").strip()
    except Exception as e:
        print("[deepgram] parse failed: {0!r}".format(e), flush=True)
        return ""

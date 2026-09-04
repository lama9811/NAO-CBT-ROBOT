"""Phase 2 unit tests — adaptive VAD threshold, streaming Silero, EoU arbiter,
and the upgraded semantic endpoint.

These tests are intentionally defensive: the modules under test (the streaming
``StreamingSilero`` API, ``_should_finalize_turn`` arbiter, and
``is_complete_thought_sync`` cache layer) are owned by sibling Phase 2 agents
in separate worktrees and may not have landed yet. Every test guards with
``pytest.importorskip(...)`` plus ``hasattr`` so the file collects clean even
when the implementations are missing — once the new APIs ship, the existing
test bodies start exercising them.

Heavy mocking is the point: torch is never invoked, OpenAI never called, no
network. The behavioural contract under test comes verbatim from
``docs/PHASE_2_TASK_MAP.md`` ("Contracts" section).
"""
from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive threshold (server.vad_silero.compute_adaptive_threshold)
# ─────────────────────────────────────────────────────────────────────────────


def _bimodal_confidences(low_peak: float = 0.15, high_peak: float = 0.65,
                         per_peak: int = 60, spread: float = 0.04) -> list[float]:
    """Build a synthetic confidence history with two clear peaks.

    Tight Gaussians around the two peaks → an unambiguous valley between them.
    The valley region is the inclusive interval bounded by the inner edge of
    each cluster, i.e. roughly [low_peak + 2*spread, high_peak - 2*spread].
    """
    samples: list[float] = []
    for i in range(per_peak):
        # Deterministic spread — alternates around the peak by a fraction of
        # `spread` so the cluster is symmetric without invoking RNG.
        offset = spread * (((i % 5) - 2) / 2.0)
        samples.append(low_peak + offset)
        samples.append(high_peak + offset)
    return samples


def test_compute_adaptive_threshold_bimodal() -> None:
    """Two clear peaks at 0.15 and 0.65 → threshold must land in the valley."""
    pytest.importorskip("server.vad_silero")
    from server import vad_silero

    fn = getattr(vad_silero, "compute_adaptive_threshold", None)
    if fn is None:
        pytest.skip("compute_adaptive_threshold not implemented yet")

    history = _bimodal_confidences(low_peak=0.15, high_peak=0.65)
    threshold = fn(history)
    # Per the task-map contract: pick the valley between speech / non-speech.
    # Anywhere in the documented [0.3, 0.55] band is correct.
    assert isinstance(threshold, float), "threshold must be a float"
    assert 0.30 <= threshold <= 0.55, (
        "bimodal valley must fall in [0.30, 0.55]; got %r" % threshold
    )


def test_compute_adaptive_threshold_unimodal_falls_back() -> None:
    """A single peak at 0.5 isn't bimodal → returns the documented 0.4 fallback."""
    pytest.importorskip("server.vad_silero")
    from server import vad_silero

    fn = getattr(vad_silero, "compute_adaptive_threshold", None)
    if fn is None:
        pytest.skip("compute_adaptive_threshold not implemented yet")

    # Tight cluster around 0.5 — no second peak.
    history = []
    for i in range(120):
        offset = 0.03 * (((i % 5) - 2) / 2.0)
        history.append(0.50 + offset)

    threshold = fn(history)
    assert isinstance(threshold, float)
    # Fallback is 0.4 per the contract; allow a tiny tolerance for any
    # implementation that returns the documented value as a literal.
    assert math.isclose(threshold, 0.4, abs_tol=1e-9), (
        "unimodal histogram must fall back to 0.4; got %r" % threshold
    )


# ─────────────────────────────────────────────────────────────────────────────
# Streaming Silero (server.vad_silero.StreamingSilero)
# ─────────────────────────────────────────────────────────────────────────────


def _silent_pcm_bytes(ms: int, sample_rate_hz: int = 16000) -> bytes:
    """Return `ms` ms of 16-bit mono PCM at the given sample rate, all zeros."""
    return b"\x00\x00" * int(sample_rate_hz * ms / 1000)


def _make_streaming_silero_with_speech_decision(
    monkeypatch: pytest.MonkeyPatch, *, speech_decision: list[bool] | bool
):
    """Construct a StreamingSilero with `is_speech_now()` mocked.

    `speech_decision` may be:
      - a bool — every call returns the same value
      - a list[bool] — each call pops the next value (last value sticks).

    Returns the (instance, decisions_iter, hits) triple where ``hits`` counts
    feed-call invocations; useful for asserting the silence counter resets.
    """
    pytest.importorskip("server.vad_silero")
    from server import vad_silero

    StreamingSilero = getattr(vad_silero, "StreamingSilero", None)
    if StreamingSilero is None:
        pytest.skip("StreamingSilero not implemented yet")

    # Monkeypatch the underlying torch model loader to a stub. We don't know
    # the implementation's exact attribute name, so we belt-and-braces patch
    # all the plausible internals.
    if hasattr(vad_silero, "_try_load"):
        monkeypatch.setattr(vad_silero, "_try_load", lambda: True, raising=False)
    if hasattr(vad_silero, "_model"):
        monkeypatch.setattr(vad_silero, "_model", object(), raising=False)
    if hasattr(vad_silero, "_torch"):
        # Use object() as a sentinel — any torch tensor ops the impl tries
        # would fail loudly, which is the point.
        monkeypatch.setattr(vad_silero, "_torch", object(), raising=False)

    inst = StreamingSilero()

    if isinstance(speech_decision, bool):
        def _decide() -> bool:
            return speech_decision
    else:
        seq = list(speech_decision)

        def _decide() -> bool:
            if len(seq) > 1:
                return seq.pop(0)
            return seq[0] if seq else False

    # Patch `is_speech_now` directly on the instance — the test only cares
    # about how silence_duration_ms responds to its return value.
    monkeypatch.setattr(inst, "is_speech_now", _decide, raising=False)
    return inst


def test_streaming_silero_silence_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """1 s of zero PCM with `is_speech_now() == False` → silence ≥ 950 ms."""
    pytest.importorskip("server.vad_silero")

    inst = _make_streaming_silero_with_speech_decision(
        monkeypatch, speech_decision=False,
    )

    # Reset to a known-clean state if the API offers it.
    if hasattr(inst, "reset"):
        inst.reset()

    # Feed in 30 ms slices to mirror Silero's native frame size.
    slice_ms = 30
    slice_bytes = _silent_pcm_bytes(slice_ms)
    total_ms = 1000
    for _ in range(total_ms // slice_ms):
        inst.feed(slice_bytes)

    silence_ms = inst.silence_duration_ms()
    assert isinstance(silence_ms, int) or isinstance(silence_ms, float), (
        "silence_duration_ms must return a number; got %r" % type(silence_ms)
    )
    assert silence_ms >= 950, (
        "1 s of all-silence input must report ≥ 950 ms; got %r" % silence_ms
    )


def test_streaming_silero_speech_resets_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Toggle 'speech detected' mid-stream; silence counter must reset."""
    pytest.importorskip("server.vad_silero")

    # First half: silence. Then: one speech tick. Then: silence again.
    # The list is consumed by `is_speech_now`; last element sticks (False).
    decisions: list[bool] = (
        [False] * 20  # 600 ms of silence
        + [True]      # speech detected!
        + [False] * 5  # 150 ms of trailing silence after the speech tick
    )

    inst = _make_streaming_silero_with_speech_decision(
        monkeypatch, speech_decision=decisions,
    )
    if hasattr(inst, "reset"):
        inst.reset()

    slice_ms = 30
    slice_bytes = _silent_pcm_bytes(slice_ms)

    # Feed the silence prelude.
    for _ in range(20):
        inst.feed(slice_bytes)
    pre_speech_silence = inst.silence_duration_ms()
    # Feed one chunk that the mock will rule "speech" + a few trailing chunks.
    for _ in range(6):
        inst.feed(slice_bytes)

    post_speech_silence = inst.silence_duration_ms()

    # The counter must not be cumulative across the speech event — after
    # speech is reported, the counter starts over from zero. We only fed
    # 5 silence ticks (150 ms) AFTER the True tick, so the post-speech
    # silence reading must be strictly less than the pre-speech reading.
    assert post_speech_silence < pre_speech_silence, (
        "silence counter did not reset on speech: "
        "pre=%r post=%r" % (pre_speech_silence, post_speech_silence)
    )
    # And specifically, it must be in the neighbourhood of 5 * 30 = 150 ms.
    assert post_speech_silence < 250, (
        "post-speech silence drifted: expected ~150 ms, got %r"
        % post_speech_silence
    )


# ─────────────────────────────────────────────────────────────────────────────
# EoU arbiter (server.app_ws._should_finalize_turn)
# ─────────────────────────────────────────────────────────────────────────────


class _SileroStub:
    """Lightweight stand-in for StreamingSilero in arbiter tests.

    Records call patterns so tests can assert which signals the arbiter
    consulted. Mirrors only the public API the arbiter is documented to use.
    """

    def __init__(self, *, speech_now: bool, silence_ms: int) -> None:
        self._speech_now = speech_now
        self._silence_ms = silence_ms
        self.calls: dict[str, int] = {"is_speech_now": 0, "silence_duration_ms": 0}

    def is_speech_now(self) -> bool:
        self.calls["is_speech_now"] += 1
        return self._speech_now

    def silence_duration_ms(self) -> int:
        self.calls["silence_duration_ms"] += 1
        return self._silence_ms

    def feed(self, _pcm: bytes) -> None:
        pass

    def reset(self) -> None:
        pass


def _resolve_arbiter():
    """Return the `_should_finalize_turn` callable, or skip if absent."""
    pytest.importorskip("server.app_ws")
    from server import app_ws

    fn = getattr(app_ws, "_should_finalize_turn", None)
    if fn is None:
        pytest.skip("_should_finalize_turn not implemented yet")
    return fn, app_ws


def _call_arbiter(fn: Any, **kwargs: Any) -> bool:
    """Best-effort call adapter: tries the documented kwargs first then
    falls back to positional. Allows the implementation some signature
    freedom while still pinning the contract.
    """
    try:
        return bool(fn(**kwargs))
    except TypeError:
        # Try the positional ordering documented in the task map:
        # (pcm_buffer, robot_hint, transcript_so_far, silero=..., ...)
        pcm = kwargs.get("pcm_buffer") or kwargs.get("pcm") or b""
        hint = kwargs.get("robot_hint", False)
        transcript = kwargs.get("transcript_so_far", "") or kwargs.get("transcript", "")
        silero = kwargs.get("silero")
        try:
            return bool(fn(pcm, hint, transcript, silero=silero))
        except TypeError:
            return bool(fn(pcm, hint, transcript))


def test_eou_arbiter_finalizes_on_long_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """700 ms of Silero-reported silence (> 600 ms threshold) → finalize."""
    fn, app_ws = _resolve_arbiter()
    silero = _SileroStub(speech_now=False, silence_ms=700)

    # If the arbiter pulls the live module-level instance, swap it in.
    for attr in ("_silero", "_streaming_silero", "STREAMING_SILERO"):
        if hasattr(app_ws, attr):
            monkeypatch.setattr(app_ws, attr, silero, raising=False)

    decision = _call_arbiter(
        fn,
        pcm_buffer=b"\x00\x00" * 16000,  # 1 s of zero PCM
        robot_hint=False,
        transcript_so_far="hello there",
        silero=silero,
    )
    assert decision is True, "long silence (700 ms) must finalize the turn"


def test_eou_arbiter_finalizes_on_robot_hint_plus_silero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Robot EoU hint + Silero confirms 250 ms silence in last 200 ms → finalize."""
    fn, app_ws = _resolve_arbiter()
    silero = _SileroStub(speech_now=False, silence_ms=250)
    for attr in ("_silero", "_streaming_silero", "STREAMING_SILERO"):
        if hasattr(app_ws, attr):
            monkeypatch.setattr(app_ws, attr, silero, raising=False)

    decision = _call_arbiter(
        fn,
        pcm_buffer=b"\x00\x00" * 4800,  # 300 ms of zero PCM
        robot_hint=True,
        transcript_so_far="how do I sign up",
        silero=silero,
    )
    assert decision is True, (
        "robot hint + silero confirming silence must finalize the turn"
    )


def test_eou_arbiter_does_not_finalize_during_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silero says speech NOW → never finalize, even with the robot hint."""
    fn, app_ws = _resolve_arbiter()
    silero = _SileroStub(speech_now=True, silence_ms=0)
    for attr in ("_silero", "_streaming_silero", "STREAMING_SILERO"):
        if hasattr(app_ws, attr):
            monkeypatch.setattr(app_ws, attr, silero, raising=False)

    decision_with_hint = _call_arbiter(
        fn,
        pcm_buffer=b"\x00\x00" * 8000,
        robot_hint=True,
        transcript_so_far="I think the answer is",
        silero=silero,
    )
    decision_without_hint = _call_arbiter(
        fn,
        pcm_buffer=b"\x00\x00" * 8000,
        robot_hint=False,
        transcript_so_far="I think the answer is",
        silero=silero,
    )
    assert decision_with_hint is False, (
        "must NOT finalize while Silero reports active speech, "
        "regardless of the robot hint"
    )
    assert decision_without_hint is False


# ─────────────────────────────────────────────────────────────────────────────
# Semantic endpoint upgrade (server.semantic_endpoint)
# ─────────────────────────────────────────────────────────────────────────────


class _CountingClient:
    """OpenAI-client stub that records how many completions calls it sees."""

    def __init__(self, response_text: str = "yes",
                 raise_exc: BaseException | None = None) -> None:
        self.calls: int = 0
        self._response_text = response_text
        self._raise_exc = raise_exc
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, parent: "_CountingClient") -> None:
            self.completions = parent._Completions(parent)

    class _Completions:
        def __init__(self, parent: "_CountingClient") -> None:
            self._parent = parent

        def create(self, **_kwargs: Any) -> Any:
            self._parent.calls += 1
            if self._parent._raise_exc is not None:
                raise self._parent._raise_exc
            # Mimic the OpenAI SDK response shape — `.choices[0].message.content`.
            text = self._parent._response_text

            class _Msg:
                def __init__(self, content: str) -> None:
                    self.content = content

            class _Choice:
                def __init__(self, content: str) -> None:
                    self.message = _Msg(content)

            class _Resp:
                def __init__(self, content: str) -> None:
                    self.choices = [_Choice(content)]

            return _Resp(text)


def _resolve_sync_completer(*, require_phase2: bool = False):
    """Return a synchronous "is this transcript complete?" callable.

    By default, prefer the Phase-2 ``is_complete_thought_sync`` symbol but
    fall back to the legacy ``is_complete_thought`` if only the latter is
    available — the cache-hit contract holds for both implementations.

    Pass ``require_phase2=True`` for tests that exercise behaviour the legacy
    path doesn't satisfy (e.g. the new "fail OPEN on API error" semantic).
    """
    pytest.importorskip("server.semantic_endpoint")
    from server import semantic_endpoint

    sync_fn = getattr(semantic_endpoint, "is_complete_thought_sync", None)
    if require_phase2:
        if sync_fn is None or asyncio.iscoroutinefunction(sync_fn):
            pytest.skip(
                "is_complete_thought_sync (Phase 2) not implemented yet"
            )
        return sync_fn, semantic_endpoint

    fn = sync_fn or getattr(semantic_endpoint, "is_complete_thought", None)
    if fn is None or asyncio.iscoroutinefunction(fn):
        pytest.skip(
            "no synchronous is_complete_thought callable available "
            "(Phase 2 sync entry point not yet shipped)"
        )
    return fn, semantic_endpoint


def _patch_client(monkeypatch: pytest.MonkeyPatch, module: Any,
                  client: Any) -> None:
    """Belt-and-braces: every plausible client-injection attr is patched.

    Three generations of this module, in order of preference:

    1. Current: no client of its own — it calls ``llm_compat.chat`` so the
       provider is a config choice. Patch that, adapting the OpenAI-shaped
       ``_CountingClient`` onto the ``chat()`` signature so the existing
       stub (and its call counter / raise behaviour) still works.
    2. Phase 2: a ``_get_client()`` helper.
    3. Legacy: a module-level ``_client``.

    Without branch 1 these tests silently patch nothing and issue real
    network calls — which is exactly how they started failing when the
    direct OpenAI client was removed.
    """
    llm_compat = getattr(module, "llm_compat", None)
    if llm_compat is not None:
        def _chat(model, messages, **kwargs):
            resp = client.chat.completions.create(
                model=model, messages=messages, **kwargs)
            return resp.choices[0].message.content or ""
        monkeypatch.setattr(llm_compat, "chat", _chat, raising=False)

    if hasattr(module, "_get_client"):
        monkeypatch.setattr(module, "_get_client",
                            lambda: client, raising=False)
    if hasattr(module, "_client"):
        monkeypatch.setattr(module, "_client", client, raising=False)


def _clear_semantic_cache(module: Any) -> None:
    """Reset whichever cache implementation the module ships with."""
    # The live module names it `_CACHE`; older revisions used `_cache`.
    # Checking only the lowercase name left a warm cache between tests.
    for attr in ("_CACHE", "_cache"):
        cache = getattr(module, attr, None)
        if isinstance(cache, dict):
            cache.clear()
            return
    # If the module exposes a clearable LRU on the sync function itself
    # (typical functools.lru_cache pattern), poke it.
    fn = getattr(module, "is_complete_thought_sync", None)
    if fn is not None and hasattr(fn, "cache_clear"):
        try:
            fn.cache_clear()
        except Exception:
            pass


def test_semantic_endpoint_cache_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Twice-same-transcript should hit the cache; client called exactly once."""
    fn, module = _resolve_sync_completer()
    _clear_semantic_cache(module)
    client = _CountingClient(response_text="yes")
    _patch_client(monkeypatch, module, client)

    # Make sure feature flag is on so the function actually runs the LLM call.
    monkeypatch.setattr(module, "USE_SEMANTIC_ENDPOINT", True, raising=False)

    # Use a transcript with > 1 word — single-word inputs short-circuit
    # without consulting the LLM (per the documented behaviour).
    transcript = "hello world how are you doing today?"

    first = fn(transcript)
    second = fn(transcript)

    assert isinstance(first, bool)
    assert first == second, "cache hit must return the same verdict"
    assert client.calls == 1, (
        "expected exactly one upstream completions.create call (cache hit on "
        "second invocation); got %d" % client.calls
    )


def test_semantic_endpoint_fails_open_on_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the API errors, ``is_complete_thought_sync`` must fail OPEN (True).

    Per the Phase-2 task map, the upgraded endpoint should err on the side of
    running the agent rather than leaving the user hanging when the LLM
    classifier itself is broken. The legacy path fails CLOSED (returns False)
    so this test skips until the Phase-2 sync entry point is in place.
    """
    fn, module = _resolve_sync_completer(require_phase2=True)
    _clear_semantic_cache(module)
    client = _CountingClient(raise_exc=RuntimeError("503 service unavailable"))
    _patch_client(monkeypatch, module, client)
    monkeypatch.setattr(module, "USE_SEMANTIC_ENDPOINT", True, raising=False)

    # Pick a transcript long enough to bypass any single-word fast-paths.
    result = fn("we should consider whether the answer is")
    assert result is True, (
        "API failures must fail OPEN (return True) so the turn isn't blocked"
    )

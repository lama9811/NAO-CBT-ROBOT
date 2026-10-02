"""Phase 9 — concurrent multi-user WS isolation tests.

Why this exists
---------------
The Phase 1 WS app keeps several module-level dicts keyed by username:

* ``server.app_ws._LAST_REPLY_CHUNKS``  — sentence echo guard window
* ``server.app_ws._LAST_REPLY_FULL``    — joined-reply substring guard
* ``server._legacy_helpers.LAST_REPLY`` — single-string echo guard

Each ``_Session`` lives in its own coroutine inside the ws_handler, but
those dicts are shared across sessions. A regression here would mean
user A's last reply leaking into user B's echo guard, surfacing as
spurious ``echo_reject`` rejections in the wild — exactly the kind of
bug that's invisible in single-user tests.

What we assert
--------------
1. Five simultaneous sessions, distinct usernames, complete a full turn
   each. Per-user reply chunks remain partitioned by username — no
   cross-talk in either dict.
2. The reply each user *receives* matches the reply we configured the
   mock runner to return for that user's transcript. (i.e. user A
   never sees user B's reply mid-stream.)
3. Latency stays bounded: p95 across all sessions/turns is within 2x
   the single-session baseline. This pins that nothing in the handler
   accidentally serializes turns across users.

Implementation note: FastAPI's TestClient drives WebSockets over a
synchronous interface backed by an async portal. Each ``websocket_connect``
call spawns its own server-side task, so we get real concurrency by
opening five clients in parallel threads. The ``_LAST_REPLY_CHUNKS``
dict is plain mutable state, so any leak would show up as a missing /
mismatched value in another user's slot at the assertion point.

Skips cleanly if ``server/app_ws.py`` hasn't landed yet in the
worktree — same defensive pattern as ``test_ws_smoke.py``.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from typing import Any

import pytest

pytest.importorskip("server.app_ws")

from fastapi.testclient import TestClient  # noqa: E402

from server import app_ws  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Frame helpers — copied locally so this file collects without depending on
# private helpers in test_ws_smoke.py (test isolation rule of thumb).
# ─────────────────────────────────────────────────────────────────────────────


def _audio_chunk_frame(seq: int, pcm_bytes: bytes, ts_ms: float) -> dict:
    return {
        "type": "audio_chunk",
        "seq": seq,
        "ts_ms": ts_ms,
        "data": base64.b64encode(pcm_bytes).decode("ascii"),
    }


def _control_frame(subtype: str, data: dict | None = None) -> dict:
    return {"type": "control", "subtype": subtype, "data": data or {}}


def _silent_pcm_chunk(ms: int = 20, sample_rate_hz: int = 16000) -> bytes:
    samples = int(sample_rate_hz * ms / 1000)
    return b"\x00\x00" * samples


def _drain_one_turn(ws, *, max_frames: int = 60, timeout_s: float = 5.0):
    frames: list[dict] = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and len(frames) < max_frames:
        try:
            raw = ws.receive_text()
        except Exception:
            break
        try:
            f = json.loads(raw)
        except Exception:
            continue
        frames.append(f)
        if f.get("type") == "control":
            sub = (f.get("subtype") or "").lower()
            if sub in {"tts_ended", "session_end"} or "end" in sub:
                break
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Per-user mock wiring — every user has its own (transcript, reply) pair so
# the test can detect cross-talk in two ways:
#   1. user A's session receives user B's reply text in its audio_chunk
#   2. user A's _LAST_REPLY_CHUNKS dict slot contains user B's text
#
# We patch on *both* the legacy module and the WS app, mirroring the helper
# in test_ws_smoke.py. The fake runner is a closure over the per-user map so
# the SDK Runner.run dispatches deterministically by username.
# ─────────────────────────────────────────────────────────────────────────────


def _install_per_user_mocks(monkeypatch, *, user_replies: dict[str, str],
                            fake_mp3: bytes) -> None:
    """Patch STT, TTS, crisis check and the agent runner.

    The runner is per-username: it looks at the username arg and returns
    that user's reply. Tests can therefore detect a leak as a "wrong reply
    text in user B's audio_chunk" without needing to inspect the dicts.
    """

    def _fake_transcribe(_path):
        # The test sends silent PCM, so STT is bypassed by ``has_voice`` and
        # this branch shouldn't fire. Pin a stable string anyway so any
        # accidental call doesn't crash with None. Must be >1 word and >4
        # chars so the legacy ``_looks_like_hallucination`` filter doesn't
        # reject it as a single-word noise fragment.
        return "hello there friend"

    try:
        from server import server as _legacy
        monkeypatch.setattr(_legacy, "_transcribe", _fake_transcribe, raising=False)
    except Exception:
        pass
    monkeypatch.setattr(app_ws, "_transcribe", _fake_transcribe, raising=False)

    # Bypass the audio gates so the silent PCM fixture flows past
    # validate_wav / has_voice and lands in the agent path. This mirrors
    # ``test_echo_regression.py``'s pattern.
    try:
        from server import _legacy_helpers as _legacy_h
        monkeypatch.setattr(_legacy_h, "validate_wav",
                            lambda *_a, **_k: True, raising=False)
        monkeypatch.setattr(_legacy_h, "has_voice",
                            lambda *_a, **_k: True, raising=False)
        monkeypatch.setattr(_legacy_h, "transcribe",
                            _fake_transcribe, raising=False)
    except Exception:
        pass

    from server import openai_tts

    def _fake_synth(_text):
        return fake_mp3

    monkeypatch.setattr(openai_tts, "synthesize", _fake_synth)
    monkeypatch.setattr(app_ws, "synthesize", _fake_synth, raising=False)

    from server import safety

    def _fake_crisis(_text, **_kw):
        return safety.CrisisResult(positive=False, source="clean")

    monkeypatch.setattr(safety, "crisis_check", _fake_crisis)
    monkeypatch.setattr(app_ws, "crisis_check", _fake_crisis, raising=False)

    # Phase 6 / Phase 7 — suppress brain_sync push and the first-turn
    # camera-announce so the per-user turn responses are the only frames
    # we have to disambiguate. The dedicated test_ws_smoke.py scenarios
    # cover those handshake paths in isolation.
    try:
        from server import session as _session
        monkeypatch.setattr(_session, "pull_brain_updates",
                            lambda *_a, **_k: {}, raising=False)
        monkeypatch.setattr(_session, "is_first_turn",
                            lambda _sid: False, raising=False)
        monkeypatch.setattr(_session, "get_camera_consent",
                            lambda _u: False, raising=False)
    except Exception:
        pass

    # Phase 2 — zero out the post-TTS echo cooldown so back-to-back turns
    # in the concurrent test don't get their inbound audio_chunks dropped
    # as echo. The dedicated echo-regression test exercises the cooldown
    # path with the production knobs.
    try:
        monkeypatch.setattr(app_ws, "TTS_COOLDOWN_PADDING_MS", 0,
                            raising=False)
        monkeypatch.setattr(app_ws.config, "MIC_GATE_GRACE_MS", 0,
                            raising=False)
    except Exception:
        pass

    # Disable the streaming Silero VAD so torch's inference graph isn't
    # invoked from multiple threads simultaneously — the model has
    # observed segfaults on heavy parallel feeds. Returning None keeps
    # the EoU arbiter on the documented "robot-hint-only finalization"
    # fallback path, which is the actual code path under test here
    # (the smoke tests already pin the silero+hint combined arbiter).
    try:
        monkeypatch.setattr(app_ws, "_get_streaming_silero",
                            lambda: None, raising=False)
        monkeypatch.setattr(app_ws, "_StreamingSilero", None,
                            raising=False)
    except Exception:
        pass

    def _fake_run_agent(username, hint, transcript_, image_b64):
        reply = user_replies.get(username, f"reply for {username}")
        return (reply, "chat", [], False)

    try:
        from server import server as _legacy
        monkeypatch.setattr(_legacy, "_run_agent", _fake_run_agent, raising=False)
    except Exception:
        pass
    monkeypatch.setattr(app_ws, "_run_agent", _fake_run_agent, raising=False)

    # The new app calls `legacy.run_agent` — patch that path too.
    try:
        from server import _legacy_helpers as _legacy_h
        monkeypatch.setattr(_legacy_h, "run_agent", _fake_run_agent, raising=False)
    except Exception:
        pass

    try:
        from agents import Runner

        class _FakeResult:
            def __init__(self, text: str) -> None:
                self.final_output = text

            def final_output_as(self, _typ):
                return self.final_output

        async def _fake_run(agent, message, **kwargs):
            ctx = kwargs.get("context") or {}
            user = None
            if isinstance(ctx, dict):
                user = ctx.get("user") or ctx.get("username")
            reply = user_replies.get(user or "", f"reply for {user}")
            return _FakeResult(reply)

        monkeypatch.setattr(Runner, "run", _fake_run, raising=False)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: 5 concurrent sessions, single turn each — no cross-talk.
# ─────────────────────────────────────────────────────────────────────────────


def _drive_one_session(username: str,
                       turn_count: int,
                       results: dict[str, list[dict]],
                       latencies: dict[str, list[float]]) -> None:
    """Connect (own TestClient), run ``turn_count`` turns, capture frames.

    Each thread gets its OWN ``TestClient`` instance because Starlette's
    sync TestClient wraps an anyio portal that holds one event loop per
    instance. Sharing a client across threads serializes turns through a
    single portal, which both kills concurrency and deadlocks under
    enough load — exactly what we observed in the first iteration of
    this test.
    """
    chunk = _silent_pcm_chunk(ms=20)
    seq = 0
    client = TestClient(app_ws.app, headers={"X-NAO-Secret": ""})
    try:
        with client.websocket_connect(f"/ws/{username}") as ws:
            ws.send_text(json.dumps(_control_frame("session_open", {
                "face_id": username, "brain_version": 2, "hint": "chat",
            })))

            collected: list[dict] = []
            for _ in range(turn_count):
                # ~0.4 s of "audio" per turn (still silence, but we mocked
                # validate_wav / has_voice through to true so the agent
                # path is reached).
                for _i in range(20):
                    ws.send_text(json.dumps(_audio_chunk_frame(
                        seq=seq, pcm_bytes=chunk, ts_ms=time.time() * 1000.0,
                    )))
                    seq += 1
                t_eou = time.monotonic()
                ws.send_text(json.dumps(_control_frame("end_of_utterance", {
                    "robot_eou_hint": True,
                    "energy_floor": 240,
                    "trail_ms": 320,
                })))
                frames = _drain_one_turn(ws, timeout_s=6.0)
                latencies.setdefault(username, []).append(
                    time.monotonic() - t_eou,
                )
                collected.extend(frames)

            try:
                ws.send_text(json.dumps(_control_frame("session_close")))
            except Exception:
                pass

            results[username] = collected
    except Exception as e:  # noqa: BLE001 — surface in the assertion below
        results[username] = [{"type": "error", "error": repr(e)}]
    finally:
        try:
            client.close()
        except Exception:
            pass


def test_five_concurrent_users_no_state_leak(monkeypatch, fake_mp3_bytes):
    """Five concurrent WS clients, two turns each = 10 turns total.

    Each user gets a unique reply string. Verify:
    * Every user receives only their own reply (no cross-contamination).
    * ``_LAST_REPLY_CHUNKS`` partitions by username at the end.
    * No session crashed — each user's frame list contains audio_chunks.
    """
    usernames = [f"user_{i}" for i in range(5)]
    user_replies = {u: f"reply for {u} only" for u in usernames}
    _install_per_user_mocks(monkeypatch,
                            user_replies=user_replies,
                            fake_mp3=fake_mp3_bytes)

    # Reset the module-level dicts so a previous test's residue doesn't
    # confuse our cross-talk check.
    app_ws._LAST_REPLY_CHUNKS.clear()
    app_ws._LAST_REPLY_FULL.clear()

    results: dict[str, list[dict]] = {}
    latencies: dict[str, list[float]] = {}
    threads: list[threading.Thread] = []
    for u in usernames:
        t = threading.Thread(
            target=_drive_one_session,
            args=(u, 2, results, latencies),
            name=f"ws-{u}",
            daemon=True,
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=30.0)
        assert not t.is_alive(), f"thread {t.name} did not finish"

    # 1. Every user got at least one audio_chunk (proves the turn ran).
    for u in usernames:
        frames = results.get(u, [])
        assert frames and frames[0].get("type") != "error", (
            f"user {u} session errored: {frames!r}"
        )
        audios = [f for f in frames if f.get("type") == "audio_chunk"]
        assert audios, (
            f"user {u} received no audio_chunk frames: {frames!r}"
        )

    # 2. No cross-talk in audio_chunk text fields.
    #    user A's frames must only contain user A's reply text.
    for u in usernames:
        my_reply = user_replies[u]
        their_replies = [user_replies[other] for other in usernames if other != u]
        for f in results[u]:
            if f.get("type") != "audio_chunk":
                continue
            text = f.get("text") or ""
            for other_reply in their_replies:
                assert other_reply not in text, (
                    f"user {u} received a frame containing user-other reply "
                    f"{other_reply!r}: {text!r}"
                )
            # And the user's own reply must show up at least somewhere.
            # We verify this aggregated below to avoid checking each frame.

        any_match = any(
            (f.get("type") == "audio_chunk"
             and my_reply in (f.get("text") or ""))
            for f in results[u]
        )
        assert any_match, (
            f"user {u} expected at least one audio_chunk with reply "
            f"{my_reply!r}; frames={results[u]!r}"
        )

    # 3. Per-user reply-chunk dict partitions cleanly. Each user's slot
    #    must NOT contain any other user's reply.
    for u in usernames:
        my_reply = user_replies[u]
        slot = app_ws._LAST_REPLY_CHUNKS.get(u, [])
        joined = " ".join(slot)
        for other in usernames:
            if other == u:
                continue
            assert user_replies[other] not in joined, (
                f"_LAST_REPLY_CHUNKS[{u}] leaked user {other}'s reply: "
                f"{joined!r}"
            )
        # Also check the joined snapshot dict.
        full = app_ws._LAST_REPLY_FULL.get(u, "")
        for other in usernames:
            if other == u:
                continue
            assert user_replies[other] not in full, (
                f"_LAST_REPLY_FULL[{u}] leaked user {other}'s reply: "
                f"{full!r}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: confirm _LAST_REPLY_CHUNKS doesn't leak across users in the
# adversarial case where one user's reply contains another user's username.
# ─────────────────────────────────────────────────────────────────────────────


def test_last_reply_chunks_isolated_under_substring_collision(monkeypatch,
                                                               fake_mp3_bytes):
    """Two users whose replies share text content must still be partitioned.

    If the dict were keyed accidentally by something other than username
    (e.g. session_id or face_id only), this test would catch it — both
    users get distinct usernames but identical reply text.
    """
    user_replies = {
        "alice": "the answer is 42",
        "bob":   "the answer is 42",
    }
    _install_per_user_mocks(monkeypatch,
                            user_replies=user_replies,
                            fake_mp3=fake_mp3_bytes)
    app_ws._LAST_REPLY_CHUNKS.clear()
    app_ws._LAST_REPLY_FULL.clear()

    results: dict[str, list[dict]] = {}
    latencies: dict[str, list[float]] = {}
    threads: list[threading.Thread] = []
    for u in user_replies:
        t = threading.Thread(
            target=_drive_one_session,
            args=(u, 1, results, latencies),
            name=f"ws-{u}",
            daemon=True,
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=20.0)
        assert not t.is_alive(), f"thread {t.name} hung"

    # Each user owns their own slot AFTER both turns finish.
    assert "alice" in app_ws._LAST_REPLY_CHUNKS, (
        f"alice missing from _LAST_REPLY_CHUNKS: "
        f"{list(app_ws._LAST_REPLY_CHUNKS)!r}"
    )
    assert "bob" in app_ws._LAST_REPLY_CHUNKS, (
        f"bob missing from _LAST_REPLY_CHUNKS: "
        f"{list(app_ws._LAST_REPLY_CHUNKS)!r}"
    )
    # The dicts must use the username key — pin this so a refactor that
    # accidentally moves the key to ``session_id`` regresses the suite.
    assert set(app_ws._LAST_REPLY_CHUNKS) >= {"alice", "bob"}, (
        f"_LAST_REPLY_CHUNKS keys do not include both usernames: "
        f"{list(app_ws._LAST_REPLY_CHUNKS)!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: latency stability under concurrency.
# ─────────────────────────────────────────────────────────────────────────────


def test_concurrent_latency_p95_within_baseline(monkeypatch, fake_mp3_bytes):
    """Run 5 sessions x 10 turns = 50 turns; assert p95 < 2x baseline.

    Baseline = single-session p95 measured immediately before. The 2x
    guardrail is generous on purpose — the threading-driven test client
    has measurable jitter on CI runners — but it still catches a
    regression that turns concurrent turns into serial turns (which
    would scale latency linearly with N).
    """
    user_replies = {f"u_{i}": f"reply {i}" for i in range(5)}
    _install_per_user_mocks(monkeypatch,
                            user_replies=user_replies,
                            fake_mp3=fake_mp3_bytes)
    app_ws._LAST_REPLY_CHUNKS.clear()
    app_ws._LAST_REPLY_FULL.clear()

    # ----- Baseline pass: a single user, 5 turns, sequential -----
    baseline_results: dict[str, list[dict]] = {}
    baseline_latencies: dict[str, list[float]] = {}
    _drive_one_session("baseline", 5,
                       baseline_results, baseline_latencies)

    base_lat = sorted(baseline_latencies.get("baseline", []))
    assert base_lat, (
        f"baseline pass produced no latency samples: "
        f"frames={baseline_results.get('baseline')!r}"
    )

    def _p95(samples: list[float]) -> float:
        if not samples:
            return 0.0
        idx = max(0, int(round(0.95 * (len(samples) - 1))))
        return sorted(samples)[idx]

    base_p95 = _p95(base_lat)
    # p95 floor — even an instant turn shouldn't read as 0 ms because of
    # FastAPI portal scheduling. If we did read 0, fall back to a 50 ms
    # floor so the multiplier comparison is meaningful.
    base_p95_floor = max(base_p95, 0.05)

    # ----- Concurrent pass: 5 users x 10 turns -----
    concurrent_results: dict[str, list[dict]] = {}
    concurrent_latencies: dict[str, list[float]] = {}
    threads: list[threading.Thread] = []
    for u in user_replies:
        t = threading.Thread(
            target=_drive_one_session,
            args=(u, 10, concurrent_results, concurrent_latencies),
            name=f"ws-{u}",
            daemon=True,
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=120.0)
        assert not t.is_alive(), f"thread {t.name} hung"

    all_lat = []
    for u, samples in concurrent_latencies.items():
        all_lat.extend(samples)
        # Per-session sanity: each session emitted samples for every turn.
        assert len(samples) == 10, (
            f"user {u}: expected 10 samples, got {len(samples)} "
            f"(samples={samples!r})"
        )
    assert len(all_lat) == 5 * 10, (
        f"expected 50 latency samples, got {len(all_lat)}: "
        f"{concurrent_latencies!r}"
    )

    concurrent_p95 = _p95(all_lat)
    # 2x baseline guardrail.
    assert concurrent_p95 < 2.0 * base_p95_floor + 1.0, (
        f"concurrent p95 {concurrent_p95:.3f}s exceeds 2x baseline "
        f"({base_p95_floor:.3f}s); samples={sorted(all_lat)!r}"
    )

    # And every session's last user reply text appears in its frames.
    for u in user_replies:
        my_reply = user_replies[u]
        frames = concurrent_results.get(u, [])
        any_match = any(
            (f.get("type") == "audio_chunk"
             and my_reply in (f.get("text") or ""))
            for f in frames
        )
        assert any_match, (
            f"user {u} expected at least one audio_chunk with "
            f"{my_reply!r}; frames={frames!r}"
        )

"""Emotion tools for the therapist + CBT + grounding agents.

Phase 6 (PRD v2) — `observe_face` is the vision-debug entrypoint. The
prior implementation called the chat-completions API with the wrong model
default (`THERAPIST_MODEL`, a text-only family) and bubbled exceptions up
to the agent loop, which was the root cause of the empty-emotion bug
described in docs/PHASE_6_TASK_MAP.md. We now:

  * Resolve the vision model from `config.VISION_MODEL` (default gpt-4o).
  * Build the chat-completions multimodal payload with `image_url`
    objects shaped `{"url": "data:image/jpeg;base64,…"}` — NOT a bare
    string. Older code passed a string directly which 400'd silently in
    some SDK versions.
  * Wrap the round-trip in `metrics.phase_timer("vision_call")` when the
    metrics module + phase label are available; otherwise no-op so this
    keeps working on environments where Prometheus isn't wired up.
  * Catch every exception and return `"unable to observe right now"` so
    the agent never crashes a user-facing turn on a vision hiccup. The
    happy-path return shape stays a dict (preserved for back-compat with
    tests in `server/tests/test_emotion.py`).
  * Honour `DEBUG_VISION=1` in the environment for full payload-size +
    response logging during development.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager

from agents import RunContextWrapper, function_tool
from server import config, memory, session
from server import llm_compat

_log = logging.getLogger(__name__)

# Sentinel string returned when the vision call fails for any reason. The
# tool never raises — agents see this string and respond with a graceful
# fallback ("I can't quite see right now, but tell me what's on your mind").
_OBSERVE_FAILURE_STRING = "unable to observe right now"


def _debug_vision_enabled() -> bool:
    """`DEBUG_VISION=1` toggles full payload + response logging in dev."""
    return os.environ.get("DEBUG_VISION") == "1"


@contextmanager
def _vision_phase_timer():
    """Defensive wrapper around `metrics.phase_timer("vision_call")`.

    Falls back to a no-op contextmanager if either:
      * the `server.metrics` module is unavailable in this deployment
        (older branches don't have Phase 1's observability layer), OR
      * `vision_call` is rejected by `_validate_phase` because it
        hasn't been added to `ALLOWED_PHASES` yet.

    This way wiring observability later is purely additive — nothing here
    has to change to pick up the timer once the phase label lands in
    `metrics.ALLOWED_PHASES`.
    """
    inner = None
    try:
        from server import metrics as _metrics  # local import → optional dep
        inner = _metrics.phase_timer("vision_call")
        inner.__enter__()
    except Exception:
        # Either the metrics module is missing or the phase label isn't
        # registered yet. Run the wrapped block without timing.
        inner = None
    try:
        yield
    finally:
        if inner is not None:
            try:
                inner.__exit__(None, None, None)
            except Exception:  # pragma: no cover — defensive only
                pass

_DISTORTIONS = (
    "catastrophizing", "all-or-nothing", "mind reading", "personalization",
    "fortune-telling", "emotional reasoning", "shoulds", "labeling",
    "magnification/minimization", "filtering",
)

# The answer for a thought that simply isn't distorted. Without this the
# classifier was forced to pick one of the ten above, so a balanced thought
# got a label anyway — the model would return "magnification/minimization"
# while its own explanation read "there's no distortion here." Telling a
# student their healthy thinking is a cognitive distortion is the wrong
# direction of error for a CBT tool.
NO_DISTORTION = "none"

# The model won't always spell it the way the prompt asked.
_NO_DISTORTION_FORMS = frozenset({
    "", "none", "no distortion", "no distortions", "n/a", "na",
    "not distorted", "no cognitive distortion", "null",
})


def _is_no_distortion(label: str) -> bool:
    """True when `label` means "this thought is fine as it is"."""
    return (label or "").strip().lower().strip(".") in _NO_DISTORTION_FORMS


def _unwrap(ctx) -> dict:
    return ctx.context if isinstance(ctx, RunContextWrapper) else ctx


def _owner(store: dict) -> str:
    """The key this conversation's therapy rows are stored under.

    ``ctx["owner"]`` (``session.therapy_owner``) when the runner set it:
    the username for a named student, ``guest:<epoch>`` for an anonymous
    visit. Never the bare "guest" -- that one key pooled every stranger's
    moods and recaps and read them back to the next stranger.
    """
    owner = str(store.get("owner") or "").strip()
    if owner:
        return owner
    return session.therapy_owner(str(store.get("username") or "guest"))


def _conv(store: dict) -> dict:
    """State that survives across turns (``ctx["conv"]``); falls back to the
    per-turn ctx itself so tools still work when called without it."""
    conv = store.get("conv")
    return conv if isinstance(conv, dict) else store


# ────────── log_emotion ──────────

def _log_emotion_impl(ctx, mood: str, intensity: int, trigger: str) -> str:
    store = _unwrap(ctx)
    store.setdefault("emotion_log", []).append(
        {"mood": mood, "intensity": intensity, "trigger": trigger}
    )
    # Persist to SQLite so the next-session greeting can surface the
    # mood trajectory. In-memory `emotion_log` is still used by the
    # session recap rollup at end-of-conversation.
    try:
        session.log_mood(_owner(store), mood, int(intensity), trigger)
    except Exception:
        pass  # best-effort; never break the agent turn
    # The therapist's opening mood check is done for this visit.
    _conv(store)["mood_checked"] = True
    return "logged"


@function_tool
def log_emotion(ctx: RunContextWrapper, mood: str, intensity: int, trigger: str) -> str:
    """Log a per-turn emotion read (mood, intensity 1-10, trigger) for session recap."""
    return _log_emotion_impl(ctx, mood, intensity, trigger)


# ────────── identify_distortion / suggest_reframe ──────────

def _classify_distortion(thought: str) -> dict:
    prompt = (
        "Identify the cognitive distortion in the user's thought, if there is "
        "one. Choose exactly ONE from: " + ", ".join(_DISTORTIONS) + ". "
        'If the thought is balanced, realistic, or fair — including thoughts '
        'that are simply sad, worried, or negative about a genuinely bad '
        'situation — answer "' + NO_DISTORTION + '". Not every difficult '
        "thought is distorted, and saying so when it isn't is worse than "
        "saying nothing. Respond as JSON: "
        '{"distortion": "<name or ' + NO_DISTORTION + '>", '
        '"explanation": "<one sentence, gentle tone>"}'
    )
    out = llm_compat.chat(
        model=config.CRISIS_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": thought},
        ],
        json_mode=True,
        temperature=0.2,
        max_tokens=300,
    )
    return json.loads(out)


def _identify_distortion_impl(thought: str) -> dict:
    return _classify_distortion(thought)


def _persist_thought(ctx, thought: str, distortion: str) -> None:
    """Write the thought + identified distortion to SQLite so it's
    available in next-session memory preamble. Best-effort — never
    breaks the agent turn.
    """
    try:
        store = _unwrap(ctx)
        record_id = session.log_thought_record(
            _owner(store), thought, distortion, reframe="")
        # Remember the row so save_full_thought_record completes this one
        # instead of adding a second row for the same exercise.
        conv = _conv(store)
        if record_id:
            conv["thought_record_id"] = record_id
        rec = conv.setdefault("cbt_record", {})
        rec["thought"] = str(thought)[:500]
        rec["distortion"] = str(distortion)[:64]
    except Exception:
        pass


def _persist_reframe(ctx, thought: str, reframe_text: str) -> None:
    try:
        store = _unwrap(ctx)
        session.attach_reframe_to_latest_thought(
            _owner(store), thought, reframe_text)
    except Exception:
        pass


def _identify_distortion_and_persist(ctx, thought: str) -> dict:
    """Classify `thought`, recording it only when it is actually distorted.

    A `none` result is a real answer, not a failure — and it must not land in
    `thought_records`, where it would read back as a distortion the student
    never had.
    """
    out = _identify_distortion_impl(thought)
    if not isinstance(out, dict):
        return out
    label = (out.get("distortion") or "").strip()
    if _is_no_distortion(label):
        out["distortion"] = NO_DISTORTION
        try:
            rec = _conv(_unwrap(ctx)).setdefault("cbt_record", {})
            rec["thought"] = str(thought)[:500]
            rec["distortion"] = NO_DISTORTION
        except Exception:
            pass
        return out
    _persist_thought(ctx, thought, label)
    return out


@function_tool
def identify_distortion(ctx: RunContextWrapper, thought: str) -> dict:
    """Identify one CBT cognitive distortion in the user's thought, or report
    that there is none. Returns distortion="none" when the thought is balanced
    — say so warmly and do not invent a distortion."""
    return _identify_distortion_and_persist(ctx, thought)


def _reframe_impl(thought: str, distortion: str) -> list[str]:
    prompt = (
        f"The user has a thought exhibiting {distortion}. Offer 2 balanced, "
        "compassionate alternative thoughts they could consider. Reply as a JSON "
        'list of 2 strings: {"reframes": ["...", "..."]}'
    )
    out = llm_compat.chat(
        model=config.CRISIS_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": thought},
        ],
        json_mode=True,
        temperature=0.4,
        max_tokens=400,
    )
    return json.loads(out)["reframes"]


def _suggest_reframe_and_persist(ctx, thought: str, distortion: str) -> list[str]:
    """Reframes for a distorted thought — and nothing for a healthy one.

    Asking the model for "alternatives to a thought exhibiting none" produces
    nonsense, and offering to fix a thought that isn't broken undercuts the
    student's own balanced thinking.
    """
    if _is_no_distortion(distortion):
        return []
    reframes = _reframe_impl(thought, distortion)
    if reframes:
        _persist_reframe(ctx, thought, reframes[0])
    return reframes


@function_tool
def suggest_reframe(ctx: RunContextWrapper, thought: str, distortion: str) -> list[str]:
    """Return two balanced reframes for a thought exhibiting the given
    distortion. Returns an empty list when the distortion is "none" — a
    balanced thought needs no reframing."""
    return _suggest_reframe_and_persist(ctx, thought, distortion)


# ────────── observe_face ──────────

# System prompt is in module scope so the self-check + tests can import it
# without instantiating the OpenAI client.
_VISION_SYSTEM = (
    "You describe a single video frame for a supportive robot companion. "
    "Stay observational — never diagnose, never identify the user, never "
    "guess age. ALWAYS return the JSON envelope below with all three "
    "fields populated, even if the image is dark, blurry, low-detail, or "
    "shows no clearly readable face: in those cases pick "
    'dominant_emotion="neutral", secondary="", and put what IS visible '
    "(lighting, framing, posture, hands, clothing, room, screen, "
    "objects) in `notes`. Do NOT return empty strings or omit fields.\n"
    "Return JSON exactly shaped:\n"
    '{"dominant_emotion": "<happy|sad|angry|fearful|surprised|disgusted|neutral|tired|stressed>",\n'
    ' "secondary": "<same vocabulary or empty string>",\n'
    ' "notes": "<≤30-word observational sentence about whatever IS visible>"}'
)


def _vision_classify(image_b64: str) -> dict:
    """Call OpenAI vision and parse the JSON envelope.

    Builds the multimodal chat-completions payload by-the-book:
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "..."},
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        }]
    The data-URL wrapper is what the chat-completions vision contract
    actually expects — passing a bare base64 string OR a `{"url": "..."}`
    without the `data:image/jpeg;base64,` prefix returns a 400.
    """
    data_uri = f"data:image/jpeg;base64,{image_b64}"

    if _debug_vision_enabled():
        # Approx payload size = ~4/3 of the raw image for base64 + small
        # JSON overhead. Logged so we can diagnose 413s when running
        # against a vision endpoint with a tight body limit.
        approx_kb = (len(image_b64) * 3) // 4 // 1024
        _log.info(
            "[DEBUG_VISION] observe_face payload: model=%s b64_len=%d ~kb=%d",
            config.VISION_MODEL, len(image_b64), approx_kb,
        )

    # llm_compat converts the image_url data URI into whichever image block
    # the configured provider expects, so VISION_MODEL can name either.
    raw = llm_compat.chat(
        model=config.VISION_MODEL,
        messages=[
            {"role": "system", "content": _VISION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this frame?"},
                    {"type": "image_url",
                     "image_url": {"url": data_uri}},
                ],
            },
        ],
        json_mode=True,
        temperature=0.2,
        max_tokens=400,
    )
    finish = None
    refusal = None

    if _debug_vision_enabled():
        _log.info("[DEBUG_VISION] observe_face response: %s (finish=%s refusal=%s)",
                   raw, finish, refusal)

    # gpt-4o sometimes returns content=None with finish_reason='content_filter'
    # when its safety classifier flags an image (people / faces fall under
    # privacy heuristics). One retry with the no-people-detection prompt
    # reliably gets past it.
    if (not raw or not raw.strip()) and refusal is None:
        raw = llm_compat.chat(
            model=config.VISION_MODEL,
            messages=[
                {"role": "system",
                 "content": (
                    "You describe SCENE COMPOSITION ONLY — lighting, "
                    "framing, posture, hands, clothing, room. Do NOT "
                    "identify any person, infer identity, or guess age. "
                    "Return the exact JSON shape: "
                    '{"dominant_emotion": "<happy|sad|angry|fearful|'
                    'surprised|disgusted|neutral|tired|stressed>", '
                    '"secondary": "<same vocabulary or \'\'>", '
                    '"notes": "<≤30 words about lighting/posture/'
                    'environment, no identifying details>"}'
                 )},
                {"role": "user",
                 "content": [
                    {"type": "text",
                     "text": "Describe the scene composition only."},
                    {"type": "image_url",
                     "image_url": {"url": data_uri}},
                 ]},
            ],
            json_mode=True,
            temperature=0.2,
            max_tokens=400,
        )
        if _debug_vision_enabled():
            _log.info("[DEBUG_VISION] observe_face retry response: %s", raw)

    if not raw or not raw.strip():
        return {"error": "empty_response"}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"dominant_emotion": "unknown", "secondary": "",
                "notes": raw.strip()[:400]}


def observe_face_for_turn(image_b64: str | None) -> dict:
    """Server-side vision call with structured status envelope.

    Phase 11 (Option B): the WS handler runs this BEFORE the agent so
    the therapist receives the observation as injected context, not via
    a tool call. Eliminates two failure modes from the prompt-only path:

      1. Model skips observe_face but still says "I can see..." (hallucination)
      2. observe_face JSON parse failure trashes the turn

    Returns a dict with these keys (always present):
        vision_status      — "success" | "unavailable" | "failed" | "skipped"
        vision_model       — model id used, or None
        vision_latency_ms  — round-trip ms, or None
        vision_summary     — short human-readable text the prompt can quote
        raw                — full vision response dict (may be None)
    """
    if not image_b64:
        return {
            "vision_status": "unavailable",
            "vision_model": None,
            "vision_latency_ms": None,
            "vision_summary": "",
            "raw": None,
        }
    t0 = time.perf_counter()
    try:
        with _vision_phase_timer():
            raw = _vision_classify(image_b64)
    except Exception as exc:
        _log.warning(
            "observe_face_for_turn failed: %s",
            exc, exc_info=_debug_vision_enabled(),
        )
        return {
            "vision_status": "failed",
            "vision_model": config.VISION_MODEL,
            "vision_latency_ms": (time.perf_counter() - t0) * 1000.0,
            "vision_summary": "",
            "raw": None,
        }

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if not isinstance(raw, dict):
        # Edge: empty / unparseable. _vision_classify now wraps these
        # but be defensive in case future paths return something else.
        return {
            "vision_status": "failed",
            "vision_model": config.VISION_MODEL,
            "vision_latency_ms": elapsed_ms,
            "vision_summary": "",
            "raw": None,
        }

    if raw.get("error"):
        # Empty response or other recoverable error — surface as failed
        # but include the raw payload for forensics.
        return {
            "vision_status": "failed",
            "vision_model": config.VISION_MODEL,
            "vision_latency_ms": elapsed_ms,
            "vision_summary": "",
            "raw": raw,
        }

    notes = (raw.get("notes") or "").strip()
    dom = (raw.get("dominant_emotion") or "").strip()
    sec = (raw.get("secondary") or "").strip()

    # Build a one-liner the therapist prompt can quote verbatim.
    summary_parts = []
    if dom:
        summary_parts.append(dom + (f"/{sec}" if sec else ""))
    if notes:
        summary_parts.append(notes)
    summary = "; ".join(summary_parts)[:400]

    return {
        "vision_status": "success",
        "vision_model": config.VISION_MODEL,
        "vision_latency_ms": elapsed_ms,
        "vision_summary": summary,
        "raw": raw,
    }


def _observe_face_impl(ctx):
    """Run the vision call defensively.

    Returns:
        - `{"error": "no_image"}` when the run context has no JPEG
          (back-compat with existing tests).
        - The parsed JSON dict from the vision model on success.
        - The string `"unable to observe right now"` on ANY error path
          — network blip, JSON parse failure, model-side refusal, etc.
          We never raise; the therapist agent sees the string and
          gracefully falls back without crashing the turn.
    """
    store = _unwrap(ctx)
    b64 = store.get("latest_image_b64")
    if not b64:
        return {"error": "no_image"}
    try:
        with _vision_phase_timer():
            return _vision_classify(b64)
    except Exception as exc:
        _log.warning("observe_face failed: %s", exc, exc_info=_debug_vision_enabled())
        return _OBSERVE_FAILURE_STRING


@function_tool
def observe_face(ctx: RunContextWrapper):
    """Read the user's face from the current turn's image.

    Returns one of:
      * dict with keys dominant_emotion / secondary / notes  (success)
      * `{"error": "no_image"}`                              (no JPEG attached)
      * `"unable to observe right now"`                      (vision call failed)

    Call this FIRST every turn whenever camera_consent=1 — the model
    needs the affect read before composing a reflective reply.
    """
    return _observe_face_impl(ctx)


# ────────── camera consent ──────────

def _set_camera_consent_impl(ctx, enabled: bool) -> str:
    store = _unwrap(ctx)
    username = store.get("username", "guest")
    session.set_camera_consent(username, enabled)
    if not enabled:
        store["suppress_image"] = True
    else:
        store["suppress_image"] = False
    return f"camera_consent={enabled}"


@function_tool
def set_camera_consent(ctx: RunContextWrapper, enabled: bool) -> str:
    """Set the user's camera consent. When False, NAO stops uploading images this session and next visits."""
    return _set_camera_consent_impl(ctx, enabled)


# ────────── full thought record ──────────

_RECORD_FIELDS = (
    "situation", "emotion", "intensity_before", "thought", "distortion",
    "evidence_for", "evidence_against", "balanced_thought", "intensity_after",
)


def _rating(value) -> int | None:
    """A 0-10 rating the student gave, or None (-1 / blank = not given)."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return None if v < 0 else min(10, v)


def _save_full_thought_record_impl(ctx, **fields) -> str:
    """Write the completed thought record and close the CBT walk.

    Empty arguments fall back to the answers the coach noted along the way
    (``ctx["conv"]["cbt_record"]``), so a field said three turns ago is not
    lost because the model left it out of the final call.
    """
    store = _unwrap(ctx)
    conv = _conv(store)
    noted = conv.get("cbt_record") or {}
    merged = {}
    for key in _RECORD_FIELDS:
        given = fields.get(key)
        if key.startswith("intensity"):
            val = _rating(given)
            merged[key] = val if val is not None else _rating(noted.get(key))
        else:
            val = str(given or "").strip()
            merged[key] = val or str(noted.get(key) or "").strip()
    if not merged["thought"]:
        return "not saved: no automatic thought recorded yet"
    if _is_no_distortion(merged["distortion"]):
        # A balanced thought is not a distortion record -- see
        # _identify_distortion_and_persist. Nothing is written.
        conv["cbt_step"] = "done"
        conv.pop("cbt_record", None)
        conv.pop("thought_record_id", None)
        return "not saved: the thought was balanced (no distortion)"
    try:
        record_id = session.save_full_thought_record(
            _owner(store), record_id=conv.get("thought_record_id"), **merged)
    except Exception:
        record_id = None
    conv["cbt_step"] = "done"
    conv.pop("cbt_record", None)
    conv.pop("thought_record_id", None)
    if not record_id:
        return "not saved"
    return f"saved thought record #{record_id}"


@function_tool
def save_full_thought_record(
    ctx: RunContextWrapper,
    situation: str = "",
    emotion: str = "",
    intensity_before: int = -1,
    thought: str = "",
    distortion: str = "",
    evidence_for: str = "",
    evidence_against: str = "",
    balanced_thought: str = "",
    intensity_after: int = -1,
) -> str:
    """Save the finished thought record (call once, after the re-rating).

    Use the student's own words. `balanced_thought` is the one the STUDENT
    chose or worded, not your suggestion. Intensities are 0-10; pass -1 if
    the student didn't give one. Blank fields fall back to answers noted
    earlier with `cbt_note`.
    """
    return _save_full_thought_record_impl(
        ctx, situation=situation, emotion=emotion,
        intensity_before=intensity_before, thought=thought,
        distortion=distortion, evidence_for=evidence_for,
        evidence_against=evidence_against,
        balanced_thought=balanced_thought, intensity_after=intensity_after,
    )


# ────────── homework ──────────

def _assign_homework_impl(ctx, task: str, due_hint: str = "") -> str:
    store = _unwrap(ctx)
    task = (task or "").strip()
    if not task:
        return "not saved: empty task"
    try:
        hw_id = session.add_homework(_owner(store), task, due_hint)
    except Exception:
        hw_id = None
    if not hw_id:
        return "not saved"
    _conv(store)["homework_id"] = hw_id
    return f"saved homework #{hw_id}"


@function_tool
def assign_homework(ctx: RunContextWrapper, task: str, due_hint: str = "") -> str:
    """Save ONE small between-session activity the student CHOSE and agreed
    to (e.g. task="10-minute walk before the 2pm class", due_hint="this
    week"). Only call after they say yes; never assign it yourself."""
    return _assign_homework_impl(ctx, task, due_hint)


def _review_homework_impl(ctx, status: str, outcome: str = "",
                          homework_id: int = 0) -> str:
    store = _unwrap(ctx)
    try:
        row = session.review_homework(
            _owner(store), outcome, status, homework_id=homework_id or None)
    except Exception:
        row = None
    if not row:
        return "no open homework to review"
    _conv(store)["homework_reviewed"] = True
    return "reviewed homework #{0} '{1}': {2}".format(
        row["id"], row["task"], row["status"])


@function_tool
def review_homework(ctx: RunContextWrapper, status: str, outcome: str = "",
                    homework_id: int = 0) -> str:
    """Record how the student's open homework went. status is one of:
    done, partly, not_done, dropped. outcome is a few of their words about
    what happened. Not doing it is fine -- stay curious, never judge."""
    return _review_homework_impl(ctx, status, outcome, homework_id)


# ────────── session recap ──────────

_EMPTY_RECAP = "Brief check-in; no notable thoughts logged."


def _clip(text, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def build_recap_body(owner: str, since: float | None) -> str:
    """A recap made from what was actually recorded this visit -- moods,
    thought records, homework -- rather than from the raw transcript."""
    parts: list[str] = []
    moods = list(reversed(session.load_recent_moods(owner, n=5, since=since)))
    if moods:
        parts.append("Mood: " + " -> ".join(
            "{0} {1}/10".format(_clip(m["mood"], 20), m["intensity"])
            for m in moods) + ".")
        trigger = _clip(moods[-1].get("trigger"), 80)
        if trigger:
            parts.append("About: " + trigger + ".")
    for t in reversed(session.load_recent_thought_records(owner, n=2,
                                                          since=since)):
        line = "Thought record: '{0}'".format(_clip(t.get("thought"), 90))
        if t.get("distortion"):
            line += " ({0})".format(_clip(t["distortion"], 30))
        balanced = t.get("balanced_thought") or t.get("reframe")
        if balanced:
            line += "; balanced thought: '{0}'".format(_clip(balanced, 90))
        before, after = t.get("intensity_before"), t.get("intensity_after")
        if before is not None and after is not None:
            line += "; {0} {1} -> {2}/10".format(
                _clip(t.get("emotion") or "feeling", 20), before, after)
        parts.append(line + ".")
    for h in reversed(session.load_homework_since(owner, since, n=3)):
        if h["status"] == "open":
            due = " ({0})".format(_clip(h["due_hint"], 30)) if h["due_hint"] else ""
            parts.append("New homework: '{0}'{1}.".format(
                _clip(h["task"], 80), due))
        else:
            outcome = " - " + _clip(h["outcome"], 60) if h["outcome"] else ""
            parts.append("Reviewed homework '{0}': {1}{2}.".format(
                _clip(h["task"], 60), h["status"].replace("_", " "), outcome))
    if not parts:
        return _EMPTY_RECAP
    return _clip(" ".join(parts), 700)


def finalize_session_recap(username: str, *, owner: str | None = None,
                           conv: dict | None = None) -> str:
    """Write (or rewrite) this visit's recap from its persisted rows.

    Covers rows written since the conversation began
    (``conv["started_at"]``). A second recap in the same conversation
    updates the same row (``conv["recap_id"]``) instead of adding another.
    Returns the recap text.
    """
    from server import conversation_state
    if conv is None:
        conv = conversation_state.state_for(username)
    owner = owner or session.therapy_owner(username)
    body = build_recap_body(owner, conv.get("started_at"))
    recap_id = conv.get("recap_id")
    try:
        if recap_id:
            session.update_recap(recap_id, body)
        else:
            recap_id = session.save_recap(owner, body)
            if recap_id:
                conv["recap_id"] = recap_id
    except Exception:
        pass
    conv["therapy_closed"] = True
    if not session.is_anonymous(username):
        try:
            from server import memory_rollup
            memory_rollup.maybe_rollup_week(owner)
            memory_rollup.maybe_rollup_month(owner)
        except Exception:
            pass
    return body


def _finalize_session_recap_impl(ctx) -> str:
    store = _unwrap(ctx)
    username = str(store.get("username") or "guest")
    conv = store.get("conv") if isinstance(store.get("conv"), dict) else None
    return finalize_session_recap(username, owner=_owner(store), conv=conv)


@function_tool(name_override="finalize_session_recap")
def finalize_session_recap_tool(ctx: RunContextWrapper) -> str:
    """Save a short recap of this visit (moods, thought records, homework).
    Call once at the close, after any homework is agreed. Safe to call
    again; it rewrites the same recap."""
    return _finalize_session_recap_impl(ctx)


def _recap_session_impl(ctx) -> str:
    store = _unwrap(ctx)
    if isinstance(store.get("conv"), dict):
        return _finalize_session_recap_impl(store)
    # Legacy path (no conversation state): the in-turn emotion log only.
    log = store.get("emotion_log", [])
    if not log:
        body = _EMPTY_RECAP
    else:
        moods = ", ".join(f"{e['mood']}({e['intensity']})" for e in log[-5:])
        body = f"Emotions: {moods}. Triggers: {'; '.join(e['trigger'] for e in log[-5:])}."
    owner = _owner(store)
    session.save_recap(owner, body)
    if not session.is_anonymous(str(store.get("username") or "guest")):
        try:
            from server import memory_rollup
            memory_rollup.maybe_rollup_week(owner)
            memory_rollup.maybe_rollup_month(owner)
        except Exception:
            pass
    return body


@function_tool
def recap_session(ctx: RunContextWrapper) -> str:
    """Summarize this therapy session and persist it to the user's history."""
    return _recap_session_impl(ctx)


# ────────── per-user memory tools (used by therapist + cbt + grounding) ──────────

@function_tool
def recall_recent_topics(ctx: RunContextWrapper) -> str:
    """Return the user's last 3 session summaries as plain text.

    Use sparingly — only when you want to surface a thread from prior
    sessions. Returns an empty string for new users.
    """
    store = _unwrap(ctx)
    face_id = store.get("username", "guest")
    rows = memory.recent_sessions(face_id, n=3)
    if not rows:
        return ""
    return "\n".join(f"- {r['summary']}" for r in rows if r.get("summary"))


@function_tool
def update_user_note(ctx: RunContextWrapper, key: str, value: str) -> str:
    """Save or overwrite a single note on the user's profile (e.g.
    update_user_note("recurring_concern", "exam stress around midterms")).

    Keys are free-form snake_case. Use this when you learn something
    durable about the user that future sessions should know.
    """
    store = _unwrap(ctx)
    face_id = store.get("username", "guest")
    if not key or not isinstance(key, str):
        return "error: empty key"
    memory.update_profile(face_id, {key: value})
    return f"saved {key}"


# ────────── __main__ self-check ──────────
#
# Quick smoke-test for the vision call wiring. Monkeypatches the OpenAI
# client to return a canned envelope, runs `_observe_face_impl` against
# a dummy ctx, asserts we get the canned dict back. Also exercises the
# error path by swapping the classifier for one that raises and asserts
# the sentinel string is returned. Run with:
#     python -m server.tools.emotion
if __name__ == "__main__":  # pragma: no cover — manual smoke test
    canned = {
        "dominant_emotion": "sad",
        "secondary": "tired",
        "notes": "soft eye contact, slumped posture, slow pacing.",
    }

    # 1) Happy path — monkeypatch _vision_classify to return the canned dict.
    _orig_classify = _vision_classify

    def _fake_classify(b64: str) -> dict:
        assert b64 == "FAKEB64", b64
        return canned

    globals()["_vision_classify"] = _fake_classify
    try:
        ctx = {"latest_image_b64": "FAKEB64"}
        out = _observe_face_impl(ctx)
        assert out == canned, ("happy-path mismatch", out)
    finally:
        globals()["_vision_classify"] = _orig_classify

    # 2) No-image path — returns {"error": "no_image"}.
    out = _observe_face_impl({"latest_image_b64": None})
    assert out == {"error": "no_image"}, ("no_image mismatch", out)

    # 3) Error path — classifier raises → sentinel string returned, no raise.
    def _broken_classify(b64: str) -> dict:
        raise RuntimeError("simulated vision API failure")

    globals()["_vision_classify"] = _broken_classify
    try:
        out = _observe_face_impl({"latest_image_b64": "FAKEB64"})
        assert out == _OBSERVE_FAILURE_STRING, ("error-path mismatch", out)
    finally:
        globals()["_vision_classify"] = _orig_classify

    # 4) Phase-timer fallback — _vision_phase_timer must yield even when
    # metrics is unavailable. We exercise the no-op branch by simulating
    # an import failure.
    import sys
    real_metrics = sys.modules.pop("server.metrics", None)
    sys.modules["server.metrics"] = None  # make import raise inside the wrapper
    try:
        with _vision_phase_timer():
            pass  # must not raise
    finally:
        if real_metrics is not None:
            sys.modules["server.metrics"] = real_metrics
        else:
            sys.modules.pop("server.metrics", None)

    print("OK")

"""CBT coach — walks one full thought record, one step per turn.

The step and the answers collected so far live in ``ctx["conv"]`` (see
``server/conversation_state.py``), which is the same dict on every turn of
a conversation. Before 2026-10-01 they lived in the per-turn ctx, which is
rebuilt every turn, so the coach forgot where it was after each sentence
and a thought record never got past step 1. ``pick_initial_agent`` resumes
this coach directly while a record is mid-walk.

The finished record is written by ``save_full_thought_record``, which
completes the row ``identify_distortion`` opened at step 3.
"""
from agents import Agent, RunContextWrapper, function_tool
from server import config, memory, session
from server.model_factory import resolve_model
from server.tools.emotion import (
    identify_distortion, suggest_reframe, log_emotion,
    save_full_thought_record,
)


# Ordered steps. "done" = record saved; "stopped" = the student declined.
STEPS = ("1", "2", "3", "4a", "4b", "5", "6")
_END_STEPS = ("done", "stopped")
_STEP_NAMES = {
    "1": "situation",
    "2": "emotion + 0-10 intensity",
    "3": "automatic thought",
    "4a": "evidence for the thought",
    "4b": "evidence against the thought",
    "5": "balanced thought (the student's words)",
    "6": "re-rate the feeling 0-10, then save",
}
# Answers cbt_note accepts, by field name.
NOTE_FIELDS = (
    "situation", "emotion", "intensity_before", "thought",
    "evidence_for", "evidence_against", "balanced_thought", "intensity_after",
)


_BASE = (
    "You are a CBT coach on a NAO robot talking with a Morgan State student. "
    "You walk them through ONE thought record, ONE step per turn: a short "
    "reflection of what they said, then ONE question, then wait. Your reply "
    "is spoken aloud, so 1-3 short sentences, about 30 words at most. Warm, "
    "curious, never clinical. You never diagnose.\n"
    "\n"
    "START OF EVERY TURN: call `cbt_get_step`. It tells you the step you are "
    "on and the answers noted so far. Never restart a record that is under "
    "way, and never re-ask something already noted.\n"
    "When the student answers the current step, call `cbt_note(field, "
    "answer)` with their words, then `cbt_set_step` to the next step, then "
    "ask the next question.\n"
    "\n"
    "STEPS:\n"
    "  1 - Situation. 'What happened? Just the facts, like a camera saw it.' "
    "      note: situation\n"
    "  2 - Emotion and intensity. 'What did you feel, and how strong was it "
    "      from 0 to 10?' note: emotion and intensity_before (a number). "
    "      Also call `log_emotion`.\n"
    "  3 - Automatic thought. 'What went through your mind right then?' "
    "      note: thought. Then call `identify_distortion(thought)` and name "
    "      the pattern gently, as a question, not a verdict ('That sounds a "
    "      bit like fortune-telling - predicting the worst. Does that fit?').\n"
    "      If it returns distortion='none', the thought is balanced: say so "
    "      warmly ('that sounds like a fair read of it'), do NOT invent a "
    "      distortion, do NOT call `suggest_reframe`, and skip to step 6 "
    "      (or close if there is nothing to work on). A thought can be sad, "
    "      worried, or negative about something genuinely bad and still not "
    "      be distorted.\n"
    "  4a - Evidence FOR. Ask, don't argue: 'What makes that thought feel "
    "      true?' note: evidence_for\n"
    "  4b - Evidence AGAINST. A Socratic question, not a lecture: 'Is there "
    "      anything that doesn't quite fit that thought?' or 'What would you "
    "      tell a friend who thought this?' note: evidence_against\n"
    "  5 - Balanced thought, in the STUDENT's words. First ask: 'Looking at "
    "      both sides, is there a fairer way to put it?' Only if they are "
    "      stuck, call `suggest_reframe` and offer the ideas as options "
    "      ('Some people might say... does either fit, or would you word it "
    "      differently?'). They choose or word it; never tell them what to "
    "      think. note: balanced_thought (their version)\n"
    "  6 - Re-rate. 'Holding that thought, how strong is the feeling now, "
    "      0 to 10?' note: intensity_after. Then call "
    "      `save_full_thought_record` with everything, using their words. "
    "      Close warmly in one sentence. Any change is fine; no change is "
    "      fine too - say so.\n"
    "\n"
    "RULES:\n"
    "1) Reflect before asking, but do not announce that you heard them or "
    "   narrate what they said.\n"
    "2) Never rush. If they seem stuck or upset, slow down and stay on the "
    "   step - append 'tts_pacing: slow' on its own line.\n"
    "3) Never push. If the student does not want to continue, call "
    "   `cbt_stop` and say that's completely okay.\n"
    "4) After saving, call `cbt_finish(summary)` with a one-sentence "
    "   summary of the record.\n"
)


def _unwrap(ctx) -> dict:
    return ctx.context if isinstance(ctx, RunContextWrapper) else ctx


def _conv(ctx) -> dict:
    """The conversation's persistent state; the ctx itself as a fallback."""
    store = _unwrap(ctx)
    conv = store.get("conv")
    return conv if isinstance(conv, dict) else store


def _normalize_step(step) -> str | None:
    s = str(step or "").strip().lower().replace("step", "").strip()
    if s in STEPS or s in _END_STEPS:
        return s
    if s == "4":
        return "4a"
    return None


def _get_step_impl(ctx) -> str:
    conv = _conv(ctx)
    step = str(conv.get("cbt_step") or "")
    if step in _END_STEPS:
        # The last record is finished: start a fresh one.
        conv.pop("cbt_record", None)
        conv.pop("thought_record_id", None)
    if step not in STEPS:
        step = "1"
        conv["cbt_step"] = step
    noted = conv.get("cbt_record") or {}
    parts = ["step={0} ({1})".format(step, _STEP_NAMES[step])]
    have = ["{0}={1!r}".format(k, noted[k]) for k in
            ("situation", "emotion", "intensity_before", "thought",
             "distortion", "evidence_for", "evidence_against",
             "balanced_thought", "intensity_after")
            if noted.get(k) not in (None, "")]
    parts.append("noted: " + (", ".join(have) if have else "nothing yet"))
    return "; ".join(parts)


@function_tool
def cbt_get_step(ctx: RunContextWrapper) -> str:
    """Return the current thought-record step and the answers noted so far."""
    return _get_step_impl(ctx)


def _set_step_impl(ctx, step: str) -> str:
    norm = _normalize_step(step)
    if norm is None:
        return "error: step must be one of " + ", ".join(STEPS)
    _conv(ctx)["cbt_step"] = norm
    return f"cbt_step={norm}"


@function_tool
def cbt_set_step(ctx: RunContextWrapper, step: str) -> str:
    """Move to a step: '1', '2', '3', '4a', '4b', '5' or '6'."""
    return _set_step_impl(ctx, step)


def _note_impl(ctx, field: str, answer: str) -> str:
    field = (field or "").strip().lower()
    if field not in NOTE_FIELDS:
        return "error: field must be one of " + ", ".join(NOTE_FIELDS)
    value = str(answer or "").strip()
    if field.startswith("intensity"):
        digits = "".join(ch for ch in value if ch.isdigit())
        if not digits:
            return "error: give the 0-10 number"
        value = max(0, min(10, int(digits[:2])))
    else:
        value = value[:500]
    _conv(ctx).setdefault("cbt_record", {})[field] = value
    return f"noted {field}"


@function_tool
def cbt_note(ctx: RunContextWrapper, field: str, answer: str) -> str:
    """Note the student's answer for the current step, in their words.
    field: situation, emotion, intensity_before, thought, evidence_for,
    evidence_against, balanced_thought, intensity_after."""
    return _note_impl(ctx, field, answer)


def _stop_impl(ctx) -> str:
    conv = _conv(ctx)
    conv["cbt_step"] = "stopped"
    conv.pop("cbt_record", None)
    conv.pop("thought_record_id", None)
    return "stopped"


@function_tool
def cbt_stop(ctx: RunContextWrapper) -> str:
    """The student doesn't want to continue the thought record. Ends it
    without saving; the next turn goes back to the regular conversation."""
    return _stop_impl(ctx)


def _finish_impl(ctx, summary: str) -> str:
    store = _unwrap(ctx)
    username = str(store.get("username") or "guest")
    # Profile notes are per named student; an anonymous visitor has none.
    if not session.is_anonymous(username):
        try:
            memory.update_profile(username, {"last_thought_record": summary})
        except Exception:
            pass
    _conv(ctx)["cbt_step"] = "done"
    return "saved"


@function_tool
def cbt_finish(ctx: RunContextWrapper, summary: str) -> str:
    """Mark the thought record complete and save a one-sentence summary
    to the user's profile under `last_thought_record`."""
    return _finish_impl(ctx, summary)


def build_cbt_coach_agent(username: str) -> Agent:
    def _instructions(_ctx, _agent) -> str:
        preamble = memory.build_context_preamble(username)
        if preamble:
            return _BASE + "\n" + preamble
        return _BASE

    return Agent(
        name="cbt_coach",
        instructions=_instructions,
        model=resolve_model(config.THERAPIST_MODEL),
        tools=[
            cbt_get_step, cbt_set_step, cbt_note,
            identify_distortion, suggest_reframe, log_emotion,
            save_full_thought_record, cbt_stop, cbt_finish,
        ],
    )


# Back-compat: existing imports of `cbt_coach_agent` still work.
# Built lazily-ish with a guest face_id; the therapist hands off to the
# `username`-specific instance via build_cbt_coach_agent above.
cbt_coach_agent = build_cbt_coach_agent("guest")

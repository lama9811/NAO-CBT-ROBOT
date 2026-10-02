"""Grounding-coach tools: paced breathing scripts and step tracking.

The grounding coach used to have one tool, ``observe_face``, and ran every
exercise from memory. That meant it lost count of where it was in a
5-4-3-2-1 between turns, and wrote its own breathing counts -- which the
TTS then rattled off in under a second when the model forgot the break
tags. These tools give it:

* ``breathing_script`` -- the exact words for one breathing round, with
  the ``<break>`` tags ``server/breathing_pacing.py`` turns into robot-side
  pauses (and that drive the eye-LED breathing in ``app_ws``).
* ``grounding_step`` -- the prompt for step N of a 5-4-3-2-1 or body scan,
  with the coach's place kept in ``ctx["conv"]`` so it survives between
  turns (the agent ctx itself is rebuilt every turn).

Pure functions do the work (``_breathing_script_impl`` /
``_grounding_step_impl``) so tests need no SDK machinery.
"""
from __future__ import annotations

from agents import RunContextWrapper, function_tool

# One spoken count takes ~200 ms, so an 800 ms gap gives ~1 s per beat.
_BEAT_BREAK = '<break time="800ms"/>'
_NUMBER_WORDS = ("one", "two", "three", "four", "five", "six", "seven",
                 "eight", "nine", "ten")

# (phase, seconds) per round. Phase words are the ones breathing_pacing's
# LED cue matcher recognises: "Breathe in", "hold", "breathe out".
BREATHING_PATTERNS: dict[str, tuple[tuple[str, int], ...]] = {
    # Panic: equal sides.
    "box": (("in", 4), ("hold", 4), ("out", 4), ("hold", 4)),
    # Falling asleep / big anxiety: long hold, longer exhale.
    "4-7-8": (("in", 4), ("hold", 7), ("out", 8)),
    # Gentle default: a longer exhale than inhale calms the body.
    "calm": (("in", 4), ("out", 6)),
}
_PATTERN_ALIASES = {
    "box breathing": "box", "square": "box",
    "478": "4-7-8", "4 7 8": "4-7-8", "relaxing": "4-7-8",
    "simple": "calm", "basic": "calm", "slow": "calm", "": "calm",
}
_PHASE_LEADS = {
    "in": "Breathe in slowly",
    "hold": "And hold",
    "out": "Now breathe out gently",
}
MAX_ROUNDS = 4


def _count(seconds: int) -> str:
    words = _NUMBER_WORDS[:max(1, min(len(_NUMBER_WORDS), int(seconds)))]
    return _BEAT_BREAK.join(words)


def _normalize_pattern(pattern: str | None) -> str:
    key = (pattern or "").strip().lower()
    key = _PATTERN_ALIASES.get(key, key)
    return key if key in BREATHING_PATTERNS else "calm"


def _breathing_script_impl(pattern: str = "calm", rounds: int = 1) -> dict:
    name = _normalize_pattern(pattern)
    try:
        rounds = int(rounds)
    except (TypeError, ValueError):
        rounds = 1
    rounds = max(1, min(MAX_ROUNDS, rounds))
    phases = BREATHING_PATTERNS[name]
    # One sentence per phase: the streaming chunker splits on periods, and
    # breathing_pacing expands each sentence's break tags into pauses.
    one_round = " ".join(
        "{0} {1}.".format(_PHASE_LEADS[phase], _count(seconds))
        for phase, seconds in phases
    )
    return {
        "pattern": name,
        "rounds": rounds,
        "seconds_per_round": sum(seconds for _p, seconds in phases),
        "script": one_round,
        "how_to_use": (
            "Say the script exactly as written, break tags included, once "
            "per turn. After each round, check in briefly before the next. "
            "Do not add your own counts."
        ),
    }


GROUNDING_EXERCISES: dict[str, tuple[str, ...]] = {
    "5-4-3-2-1": (
        "Look around and name five things you can see.",
        "Now listen. What are four things you can hear?",
        "Notice three things you can feel, like your feet on the floor.",
        "What are two things you can smell, or two smells you like?",
        "And one thing you can taste, or one you enjoy.",
    ),
    "body_scan": (
        "Let's start at the top. Notice your forehead and jaw, and let them "
        "soften.",
        "Now your neck and shoulders. Let them drop a little.",
        "Notice your chest and belly rising and falling as you breathe.",
        "Now your arms and hands. Let them rest heavy.",
        "Finally your legs and feet. Feel them supported by the floor.",
    ),
}
_EXERCISE_ALIASES = {
    "54321": "5-4-3-2-1", "5 4 3 2 1": "5-4-3-2-1", "senses": "5-4-3-2-1",
    "five senses": "5-4-3-2-1",
    "body scan": "body_scan", "bodyscan": "body_scan", "scan": "body_scan",
}


def _normalize_exercise(exercise: str | None) -> str | None:
    key = (exercise or "").strip().lower()
    key = _EXERCISE_ALIASES.get(key, key)
    return key if key in GROUNDING_EXERCISES else None


def _grounding_step_impl(store: dict, exercise: str, step: int = 0) -> dict:
    """Prompt for one step. ``step`` 0 means "the next one".

    Progress lives in ``store["conv"]["grounding"]`` when a conversation
    state dict is present, so asking for "the next step" works across
    turns.
    """
    name = _normalize_exercise(exercise)
    if name is None:
        return {"error": "unknown exercise",
                "exercises": sorted(GROUNDING_EXERCISES)}
    steps = GROUNDING_EXERCISES[name]
    conv = store.get("conv") if isinstance(store, dict) else None
    progress = conv.get("grounding") if isinstance(conv, dict) else None
    try:
        step = int(step)
    except (TypeError, ValueError):
        step = 0
    if step <= 0:
        if isinstance(progress, dict) and progress.get("exercise") == name:
            step = int(progress.get("step") or 0) + 1
        else:
            step = 1
    if step > len(steps):
        if isinstance(conv, dict):
            conv.pop("grounding", None)
        return {"exercise": name, "step": len(steps),
                "total_steps": len(steps), "done": True,
                "prompt": "",
                "next": "Ask how they feel now, then hand back to the "
                        "therapist."}
    if isinstance(conv, dict):
        conv["grounding"] = {"exercise": name, "step": step}
    return {"exercise": name, "step": step, "total_steps": len(steps),
            "done": False, "prompt": steps[step - 1]}


def _unwrap(ctx) -> dict:
    return ctx.context if isinstance(ctx, RunContextWrapper) else ctx


@function_tool
def breathing_script(pattern: str = "calm", rounds: int = 1) -> dict:
    """Get the exact words for one paced breathing round.

    pattern: "calm" (in 4, out 6 -- the gentle default), "box" (4-4-4-4,
    for panic) or "4-7-8" (for strong anxiety). The script carries timing
    tags so NAO counts in real seconds and its eyes pulse with the breath.
    Say it exactly as written, one round per turn.
    """
    return _breathing_script_impl(pattern, rounds)


@function_tool
def grounding_step(ctx: RunContextWrapper, exercise: str, step: int = 0) -> dict:
    """Get the prompt for one step of a grounding exercise.

    exercise: "5-4-3-2-1" or "body_scan". step: 1-5, or 0 for the next
    step after the last one you used (your place is remembered between
    turns). When done is true, ask how they feel and hand back.
    """
    return _grounding_step_impl(_unwrap(ctx), exercise, step)

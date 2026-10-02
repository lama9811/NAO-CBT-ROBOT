"""Homework tools, recaps built from persisted rows, the therapist's
session-structure notes, and the memory preamble -- all keyed by the
therapy owner, never the bare "guest" name."""
from __future__ import annotations

import time
import uuid

import pytest

from server import conversation_state as cs
from server import memory, session
from server.agents import therapist
from server.tools import emotion


@pytest.fixture(autouse=True)
def _no_rollups(monkeypatch):
    # Week/month rollups call a model; recaps here must stay offline.
    from server import memory_rollup
    monkeypatch.setattr(memory_rollup, "maybe_rollup_week", lambda u: None)
    monkeypatch.setattr(memory_rollup, "maybe_rollup_month", lambda u: None)


def _named() -> str:
    return "Student" + uuid.uuid4().hex[:6]


def _ctx(username: str, conv: dict | None = None) -> dict:
    return {"username": username, "owner": session.therapy_owner(username),
            "conv": {"started_at": time.time() - 5} if conv is None else conv,
            "emotion_log": []}


# ------------------------------------------------------------------ homework

def test_assign_and_review_homework_tools():
    who = _named()
    ctx = _ctx(who)
    out = emotion._assign_homework_impl(ctx, "10-minute walk", "this week")
    assert out.startswith("saved homework #")
    assert session.load_open_homework(who)[0]["task"] == "10-minute walk"
    out = emotion._review_homework_impl(ctx, "partially", "walked once")
    assert "partly" in out
    assert ctx["conv"]["homework_reviewed"] is True
    assert emotion._review_homework_impl(ctx, "done") == \
        "no open homework to review"


def test_assign_homework_refuses_empty_task():
    assert emotion._assign_homework_impl(_ctx(_named()), "  ").startswith(
        "not saved")


def test_anonymous_homework_is_per_visit_not_pooled_under_guest():
    session.retire_anonymous_epoch()
    ctx = _ctx("guest")
    assert ctx["owner"].startswith("guest:")
    emotion._assign_homework_impl(ctx, "drink some water")
    assert session.load_open_homework(ctx["owner"])[0]["task"] == \
        "drink some water"
    # The bare name sees nothing, and the next stranger starts clean.
    assert session.load_open_homework("guest") == []
    session.retire_anonymous_epoch()
    assert session.load_open_homework(session.therapy_owner("guest")) == []


def test_owner_falls_back_to_therapy_owner_when_ctx_lacks_it():
    who = _named()
    emotion._assign_homework_impl({"username": who}, "journal")
    assert session.load_open_homework(who)[0]["task"] == "journal"


def test_log_emotion_uses_owner_and_marks_mood_checked():
    who = _named()
    ctx = _ctx(who)
    emotion._log_emotion_impl(ctx, "anxious", 7, "exam")
    assert session.load_recent_moods(who)[0]["mood"] == "anxious"
    assert ctx["conv"]["mood_checked"] is True


# -------------------------------------------------------------------- recaps

def test_recap_is_built_from_persisted_rows():
    who = _named()
    ctx = _ctx(who)
    emotion._log_emotion_impl(ctx, "anxious", 8, "final exam")
    session.save_full_thought_record(
        who, thought="I'll fail", distortion="fortune-telling",
        emotion="anxious", intensity_before=8,
        balanced_thought="I've passed before", intensity_after=5)
    emotion._log_emotion_impl(ctx, "calmer", 5, "final exam")
    emotion._assign_homework_impl(ctx, "study 20 minutes", "tomorrow")

    body = emotion._finalize_session_recap_impl(ctx)
    assert "anxious 8/10 -> calmer 5/10" in body
    assert "I'll fail" in body and "fortune-telling" in body
    assert "I've passed before" in body
    assert "8 -> 5/10" in body
    assert "study 20 minutes" in body
    assert session.load_recent_recaps(who, n=5) == [body]
    assert ctx["conv"]["therapy_closed"] is True


def test_second_recap_updates_the_same_row():
    who = _named()
    ctx = _ctx(who)
    emotion._log_emotion_impl(ctx, "sad", 6, "breakup")
    first = emotion._finalize_session_recap_impl(ctx)
    emotion._assign_homework_impl(ctx, "text a friend")
    second = emotion._finalize_session_recap_impl(ctx)
    assert first != second
    assert session.load_recent_recaps(who, n=5) == [second]


def test_recap_covers_only_this_visit():
    who = _named()
    old_ctx = _ctx(who)
    emotion._log_emotion_impl(old_ctx, "angry", 9, "last week's fight")
    # A new conversation that started in the future of that row.
    ctx = _ctx(who, conv={"started_at": time.time() + 2})
    body = emotion._finalize_session_recap_impl(ctx)
    assert "angry" not in body
    assert "check-in" in body.lower()


def test_finalize_by_username_uses_live_conversation_state():
    cs._reset_for_tests()
    who = _named()
    conv = cs.state_for(who)
    conv["started_at"] = time.time() - 5
    session.log_mood(who, "hopeful", 4, "new job")
    body = emotion.finalize_session_recap(who)
    assert "hopeful 4/10" in body
    assert conv["recap_id"]


def test_anonymous_recap_never_lands_under_guest():
    session.retire_anonymous_epoch()
    ctx = _ctx("guest")
    emotion._log_emotion_impl(ctx, "tired", 6, "work")
    body = emotion._finalize_session_recap_impl(ctx)
    assert "tired" in body
    assert session.load_recent_recaps("guest") == []
    assert session.load_recent_recaps(ctx["owner"]) == [body]


def test_legacy_recap_without_conv_still_works(monkeypatch):
    saved = []
    monkeypatch.setattr(session, "save_recap",
                        lambda u, body: saved.append((u, body)))
    out = emotion._recap_session_impl({"username": "bob", "emotion_log": []})
    assert "check-in" in out.lower() and saved[0][0] == "bob"


# ---------------------------------------------------------- therapist prompt

def _render(username: str, ctx: dict) -> str:
    agent = therapist.build_therapist_agent(username)

    class _Wrap:
        context = ctx

    return agent.instructions(_Wrap(), agent)


def test_therapist_has_session_tools_and_flow():
    agent = therapist.build_therapist_agent("alice")
    names = {t.name for t in agent.tools}
    assert {"assign_homework", "review_homework",
            "finalize_session_recap", "log_emotion"} <= names
    prompt = agent.instructions(None, agent)
    assert "SESSION FLOW" in prompt
    assert "mood check" in prompt.lower()
    assert "never assign" in prompt.lower()


def test_therapist_sees_open_homework_and_visit_state():
    who = _named()
    session.add_homework(who, "walk before class", "this week")
    ctx = _ctx(who, conv={})
    text = _render(who, ctx)
    assert "OPEN HOMEWORK (student data" in text
    assert "walk before class" in text
    assert "Mood check: not yet" in text

    ctx["conv"].update({"mood_checked": True, "homework_reviewed": True,
                        "therapy_closing": 1})
    text = _render(who, ctx)
    assert "OPEN HOMEWORK (student data" not in text
    assert "Mood check: done" in text
    assert "CLOSING" in text


# ------------------------------------------------------------ memory preamble

def test_preamble_surfaces_homework_and_full_thought_record_for_named_user():
    who = _named()
    session.add_homework(who, "call my sister", "weekend")
    session.save_full_thought_record(
        who, thought="Nobody cares", distortion="mind reading",
        emotion="lonely", intensity_before=7,
        balanced_thought="My sister checks on me", intensity_after=4)
    pre = memory.build_context_preamble(who)
    assert "Open homework: 'call my sister' (weekend)" in pre
    assert "My sister checks on me" in pre
    assert "lonely 7/10 -> 4/10" in pre


def test_preamble_shows_nothing_for_anonymous_users():
    session.retire_anonymous_epoch()
    owner = session.therapy_owner("guest")
    session.add_homework(owner, "secret task")
    session.log_mood(owner, "sad", 8, "private")
    assert memory.build_context_preamble("guest") == ""
    assert memory.build_context_preamble(owner) == ""

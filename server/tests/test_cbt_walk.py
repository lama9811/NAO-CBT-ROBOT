"""The CBT coach walks a whole thought record across turns.

Each turn gets a fresh ctx; only ``ctx["conv"]`` survives. These tests
rebuild ctx every "turn" the way the runner does, so a step or answer kept
anywhere else would be lost exactly as it was before 2026-10-01.
"""
from __future__ import annotations

import uuid
from unittest.mock import patch

from server import session
from server.agents import cbt_coach as cc
from server.tools import emotion


def _ctx(conv: dict, owner: str) -> dict:
    """A fresh per-turn ctx sharing the conversation's ``conv``."""
    return {"username": owner, "owner": owner, "conv": conv,
            "emotion_log": [], "actions_queue": []}


def _owner() -> str:
    return "cbt_" + uuid.uuid4().hex[:8]


def test_step_survives_a_fresh_ctx():
    conv: dict = {}
    who = _owner()
    assert cc._get_step_impl(_ctx(conv, who)).startswith("step=1")
    cc._set_step_impl(_ctx(conv, who), "2")
    assert cc._get_step_impl(_ctx(conv, who)).startswith("step=2")


def test_invalid_step_rejected_and_4_means_4a():
    conv: dict = {}
    assert cc._set_step_impl(_ctx(conv, "x"), "9").startswith("error")
    assert cc._set_step_impl(_ctx(conv, "x"), "4") == "cbt_step=4a"


def test_get_step_reports_noted_answers():
    conv: dict = {}
    who = _owner()
    cc._note_impl(_ctx(conv, who), "situation", "Got a C on the quiz")
    out = cc._get_step_impl(_ctx(conv, who))
    assert "Got a C on the quiz" in out


def test_intensity_note_takes_the_number():
    conv: dict = {}
    assert cc._note_impl(_ctx(conv, "x"), "intensity_before",
                         "maybe like an 8") == "noted intensity_before"
    assert conv["cbt_record"]["intensity_before"] == 8
    assert cc._note_impl(_ctx(conv, "x"), "intensity_after",
                         "not sure").startswith("error")
    assert cc._note_impl(_ctx(conv, "x"), "mood", "x").startswith("error")


def test_full_walk_writes_one_complete_row():
    conv: dict = {}
    who = _owner()

    # Turn 1-2: situation, emotion + intensity.
    cc._note_impl(_ctx(conv, who), "situation", "Got a C on the quiz")
    cc._set_step_impl(_ctx(conv, who), "2")
    cc._note_impl(_ctx(conv, who), "emotion", "anxious")
    cc._note_impl(_ctx(conv, who), "intensity_before", "8")
    cc._set_step_impl(_ctx(conv, who), "3")

    # Turn 3: automatic thought -> identify_distortion opens the row.
    with patch.object(emotion, "_identify_distortion_impl",
                      return_value={"distortion": "fortune-telling",
                                    "explanation": "Predicting the worst."}):
        emotion._identify_distortion_and_persist(
            _ctx(conv, who), "I'm going to fail the final")
    assert conv["thought_record_id"]
    cc._set_step_impl(_ctx(conv, who), "4a")

    # Turns 4-6: evidence, the student's balanced thought, re-rate.
    cc._note_impl(_ctx(conv, who), "evidence_for", "I got a C")
    cc._note_impl(_ctx(conv, who), "evidence_against",
                  "I got a B on the midterm")
    cc._note_impl(_ctx(conv, who), "balanced_thought",
                  "One quiz doesn't decide the final")
    cc._note_impl(_ctx(conv, who), "intensity_after", "5")

    out = emotion._save_full_thought_record_impl(_ctx(conv, who))
    assert out.startswith("saved thought record #")

    rows = session.load_recent_thought_records(who, n=5)
    assert len(rows) == 1
    r = rows[0]
    assert r["situation"] == "Got a C on the quiz"
    assert r["emotion"] == "anxious"
    assert r["thought"] == "I'm going to fail the final"
    assert r["distortion"] == "fortune-telling"
    assert r["evidence_against"] == "I got a B on the midterm"
    assert r["balanced_thought"] == "One quiz doesn't decide the final"
    assert (r["intensity_before"], r["intensity_after"]) == (8, 5)
    # The walk is closed and its scratch state cleared.
    assert conv["cbt_step"] == "done"
    assert "cbt_record" not in conv and "thought_record_id" not in conv


def test_explicit_args_override_noted_answers():
    conv: dict = {}
    who = _owner()
    cc._note_impl(_ctx(conv, who), "balanced_thought", "draft")
    emotion._save_full_thought_record_impl(
        _ctx(conv, who), thought="Everyone hates me", distortion="mind reading",
        balanced_thought="Two friends texted me today", intensity_after=-1)
    r = session.load_recent_thought_records(who)[0]
    assert r["balanced_thought"] == "Two friends texted me today"
    assert r["intensity_after"] is None


def test_no_distortion_path_writes_nothing():
    conv: dict = {}
    who = _owner()
    with patch.object(emotion, "_identify_distortion_impl",
                      return_value={"distortion": "none",
                                    "explanation": "Balanced."}):
        out = emotion._identify_distortion_and_persist(
            _ctx(conv, who), "I'm sad my grandma is sick")
    assert out["distortion"] == "none"
    assert emotion._suggest_reframe_and_persist(
        _ctx(conv, who), "I'm sad my grandma is sick", "none") == []
    saved = emotion._save_full_thought_record_impl(_ctx(conv, who))
    assert "balanced" in saved
    assert session.load_recent_thought_records(who) == []
    assert conv["cbt_step"] == "done"


def test_save_without_a_thought_is_refused():
    conv: dict = {}
    assert emotion._save_full_thought_record_impl(
        _ctx(conv, _owner())).startswith("not saved")


def test_stop_ends_the_walk_without_saving():
    conv: dict = {"cbt_step": "4b", "cbt_record": {"thought": "x"}}
    cc._stop_impl(_ctx(conv, _owner()))
    assert conv["cbt_step"] == "stopped" and "cbt_record" not in conv


def test_finished_record_starts_fresh_next_time():
    conv: dict = {"cbt_step": "done", "cbt_record": {"thought": "old"}}
    assert cc._get_step_impl(_ctx(conv, _owner())).startswith("step=1")
    assert "cbt_record" not in conv


def test_cbt_finish_skips_profile_for_anonymous(monkeypatch):
    calls = []
    monkeypatch.setattr(cc.memory, "update_profile",
                        lambda fid, d: calls.append(fid))
    conv: dict = {}
    cc._finish_impl({"username": "guest", "conv": conv}, "summary")
    assert calls == [] and conv["cbt_step"] == "done"
    cc._finish_impl({"username": "Mingma", "conv": conv}, "summary")
    assert calls == ["Mingma"]


def test_coach_has_the_walk_tools_and_prompt_covers_every_step():
    agent = cc.build_cbt_coach_agent("alice")
    names = {t.name for t in agent.tools}
    assert {"cbt_get_step", "cbt_set_step", "cbt_note", "identify_distortion",
            "suggest_reframe", "save_full_thought_record",
            "cbt_stop"} <= names
    prompt = agent.instructions(None, agent)
    for marker in ("Situation", "0 to 10", "Automatic thought",
                   "Evidence FOR", "Evidence AGAINST", "STUDENT's words",
                   "Re-rate", "distortion='none'"):
        assert marker in prompt, marker

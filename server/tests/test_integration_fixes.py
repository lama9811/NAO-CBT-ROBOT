# -*- coding: utf-8 -*-
"""Integration bugs between the CBT lane, safety/privacy and turn-taking.

Each test names the reviewer finding it pins. `pytest-asyncio` is not
installed, so coroutines are driven with `asyncio.run()`.
"""
import asyncio
import json
import sqlite3
import time
from unittest.mock import patch

import pytest

from server import (app_ws, conversation_state, privacy, safety, session,
                    turn_taking)


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_text(self, text):
        try:
            self.sent.append(json.loads(text))
        except (TypeError, ValueError):
            self.sent.append(text)

    def audio(self):
        return [f for f in self.sent
                if isinstance(f, dict) and f.get("type") == "audio_chunk"]


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(app_ws, "_synth_for", lambda *a, **k: b"\x01mp3")
    conversation_state._reset_for_tests()
    privacy._reset_for_tests()
    yield
    conversation_state._reset_for_tests()
    privacy._reset_for_tests()


# ───────── 1. goodbye right after a crisis reply ─────────

def test_goodbye_fast_path_skipped_while_crisis_followup_pending(monkeypatch):
    monkeypatch.setattr(app_ws, "_support_lane", lambda u: None)
    sess = app_ws._Session("ix_crisis_bye")
    conv = {"crisis_followup": {"level": "keyword"}}
    assert app_ws._goodbye_fast_path(sess, conv, "okay bye") is False
    assert app_ws._goodbye_fast_path(sess, {}, "okay bye") is True


def test_goodbye_fast_path_still_defers_first_in_lane_goodbye(monkeypatch):
    monkeypatch.setattr(app_ws, "_support_lane", lambda u: "therapist")
    sess = app_ws._Session("ix_lane_bye")
    assert app_ws._goodbye_fast_path(sess, {}, "bye") is False
    assert app_ws._goodbye_fast_path(
        sess, {"therapy_closing": 1}, "bye") is True


# ───────── 2. crisis opens the support lane ─────────

def test_after_crisis_opens_the_support_lane(monkeypatch, tmp_path):
    from server import config
    monkeypatch.setattr(session, "_DB_PATH", str(tmp_path / "c.db"))
    monkeypatch.setattr(config, "CRISIS_ALERT_WEBHOOK_URL", "")
    sess = app_ws._Session("ix_lane_after_crisis")
    conv = {}
    app_ws._after_crisis(sess, conv, safety.CrisisResult(True, "keyword"))
    assert conversation_state.active_lane(conv) == "therapist"


# ───────── 3. in-lane goodbye phrases match turn_taking ─────────

@pytest.mark.parametrize("text", [
    "good night", "take care", "talk to you later", "see ya",
    "I'm heading out", "okay, thanks. Take care!",
])
def test_is_closing_accepts_every_turn_taking_goodbye(text):
    from server.agents import _is_closing
    assert turn_taking.is_goodbye(text)
    assert _is_closing(text) is True


# ───────── 4/5. goodbye detaches state synchronously, disarms ─────────

def test_goodbye_clears_state_before_the_recap_finishes(monkeypatch):
    monkeypatch.setattr(app_ws, "_support_lane", lambda u: "therapist")
    started = {}

    def slow_recap(username, *, owner=None, conv=None):
        started["owner"] = owner
        started["conv"] = conv
        time.sleep(0.3)
        return "r"

    monkeypatch.setattr(app_ws._emotion_module, "finalize_session_recap",
                        slow_recap, raising=False)

    async def _go():
        ws, sess = FakeWS(), app_ws._Session("ix_bye_sync")
        conv = conversation_state.state_for("ix_bye_sync")
        conversation_state.set_lane(conv, "therapist")
        conv["cbt_step"] = 3
        await app_ws._emit_goodbye(ws, sess, "bye", {})
        # Recap still running, but the conversation is already closed:
        fresh = conversation_state.state_for("ix_bye_sync")
        assert "cbt_step" not in fresh and "lane" not in fresh
        # A turn in this window keeps its new state.
        fresh["crisis_followup"] = {"level": "llm"}
        await asyncio.gather(*list(app_ws._BACKGROUND_TASKS))
        return sess

    sess = asyncio.run(_go())
    assert started["owner"] == "ix_bye_sync"
    assert started["conv"]["cbt_step"] == 3
    assert conversation_state.state_for("ix_bye_sync")["crisis_followup"]
    # 5: idle check-in and repair are disarmed after the goodbye.
    assert sess.idle_checkin_done is True
    assert sess.user_turns == 0
    assert sess.repair_armed is False
    assert app_ws._claim_idle_checkin(sess) is False


def test_anonymous_goodbye_recaps_under_the_old_epoch(monkeypatch):
    monkeypatch.setattr(app_ws, "_support_lane", lambda u: "therapist")
    seen = {}
    monkeypatch.setattr(
        app_ws._emotion_module, "finalize_session_recap",
        lambda username, *, owner=None, conv=None: seen.setdefault(
            "owner", owner), raising=False)
    old = session.therapy_owner("guest")

    async def _go():
        await app_ws._emit_goodbye(FakeWS(), app_ws._Session("guest"),
                                   "bye", {})
        # Epoch retired synchronously, before the recap ran.
        assert session.live_anonymous_key() is None
        await asyncio.gather(*list(app_ws._BACKGROUND_TASKS))

    asyncio.run(_go())
    assert seen["owner"] == old


def test_recap_is_not_written_after_forget_me(monkeypatch, tmp_path):
    from server.tools import emotion
    monkeypatch.setattr(session, "_DB_PATH", str(tmp_path / "f.db"))
    with session._conn():
        pass
    conv = {"started_at": time.time() - 60}
    privacy.forget_user_data("ix_forgot")
    emotion.finalize_session_recap("ix_forgot", owner="ix_forgot", conv=conv)
    with sqlite3.connect(str(tmp_path / "f.db")) as c:
        assert c.execute("SELECT COUNT(*) FROM recaps").fetchone()[0] == 0
    # A conversation that began after the forget records normally.
    assert privacy.forgotten_since("ix_forgot", time.time() + 1) is False


# ───────── 6. idle check-in re-checked after synthesis ─────────

def test_idle_checkin_aborts_if_the_user_started_speaking(monkeypatch):
    sess = app_ws._Session("ix_idle")

    def synth_while_user_talks(*a, **k):
        sess.had_speech = True  # user began talking during synthesis
        return b"\x01mp3"

    monkeypatch.setattr(app_ws, "_synth_for", synth_while_user_talks)
    app_ws._mark_turn_accepted(sess)
    long_ago = time.time() * 1000.0 - (turn_taking.IDLE_CHECKIN_S * 1000 + 5000)
    sess.last_user_speech_ms = long_ago
    sess.tts_active_until_ms = long_ago
    ws = FakeWS()
    assert asyncio.run(app_ws._maybe_idle_checkin(ws, sess)) is False
    assert ws.audio() == []


# ───────── 7. forget_me tool defers deletion to after the run ─────────

def test_forget_me_tool_defers_until_the_run_ends(monkeypatch):
    from server.tools import privacy_tools
    calls = []
    monkeypatch.setattr(privacy, "forget_user_data",
                        lambda u: calls.append(u) or {})

    class Ctx:
        context = {"username": "ix_tool"}

    out = privacy_tools._forget_me_impl(Ctx(), True)
    assert calls == []  # nothing deleted mid-run
    assert "face" in out.lower()
    assert privacy.run_pending_forget("ix_tool") is not None
    assert calls == ["ix_tool"]
    assert privacy.run_pending_forget("ix_tool") is None


def test_forget_replies_are_honest_about_the_face():
    assert "face" in privacy.FORGET_DONE_REPLY.lower()
    assert "everything" not in privacy.FORGET_DONE_REPLY.lower()


# ───────── 8. stitched-only soft hit fails open on classifier error ───

def test_stitched_only_soft_hit_fails_open_on_error():
    with patch("server.safety._llm_classify", side_effect=RuntimeError("x")):
        r = safety.crisis_check("what time is the library open",
                                recent=["I feel hopeless about math"])
    assert r.positive is False


def test_current_soft_hit_still_fails_safe_on_error():
    with patch("server.safety._llm_classify", side_effect=RuntimeError("x")):
        r = safety.crisis_check("I feel hopeless",
                                recent=["hi there"])
    assert r.positive is True and r.source == "failsafe"


# ───────── 9. CS direct leaves the therapy lane ─────────

def test_cs_direct_leaves_the_therapy_lane(monkeypatch):
    from server.tools import cs_navigator as csn

    async def fake_search(ctx, q):
        return "COSC 220 is taught by Jin Guo."

    class FakeHist:
        async def add_items(self, items):
            pass

    monkeypatch.setattr(csn, "_cs_navigator_search_impl", fake_search)
    monkeypatch.setattr(session, "get_or_create_session",
                        lambda u: FakeHist())
    conv = conversation_state.state_for("ix_cs")
    conversation_state.set_lane(conv, "therapist")
    conv["therapy_closing"] = 1
    asyncio.run(app_ws._emit_cs_direct(
        FakeWS(), app_ws._Session("ix_cs"), "Who teaches COSC 220?", {}, 0.0))
    conv = conversation_state.state_for("ix_cs")
    assert conversation_state.active_lane(conv) is None
    assert "therapy_closing" not in conv


# ───────── 10. "fine without me" style phrasings reach the classifier ─

@pytest.mark.parametrize("text", [
    "I feel like everyone would be fine without me",
    "nobody would notice if I was gone",
    "would anyone even miss me if I left for good",
    "they would all be happier without me",
])
def test_without_me_phrasings_are_soft_triggers(text):
    assert safety.soft_match(text)
    with patch("server.safety._llm_classify", return_value=True) as m:
        assert safety.crisis_check(text).positive
    assert m.called


@pytest.mark.parametrize("text", [
    "they went to the movies without me",
    "can you miss me with that",
])
def test_ordinary_without_me_is_not_a_trigger(text):
    assert not safety.soft_match(text)


# ───────── 12. dashboard counts a goodbye as answered ─────────

def test_dashboard_goodbye_is_not_dropped(monkeypatch):
    from server import dashboard as d
    monkeypatch.setattr(d, "STATE", d._State())
    d.capture(None, "info", {"event": "turn_complete", "session_id": "s1",
                             "outcome": "goodbye", "transcript": "bye"})
    today = d.snapshot()["today"]
    assert today["dropped"] == 0
    assert today["answered"] == 1


def test_run_agent_deletes_after_the_run_returns(monkeypatch):
    from server import _legacy_helpers as legacy
    order = []
    monkeypatch.setattr(legacy, "pick_initial_agent",
                        lambda *a, **k: object())
    monkeypatch.setattr(legacy.session, "get_or_create_session",
                        lambda u: object())
    monkeypatch.setattr(legacy.session, "therapy_owner", lambda u: u)
    monkeypatch.setattr(privacy, "forget_user_data",
                        lambda u: order.append("forget") or {})

    def fake_topology(agent, message, *, context, session):
        privacy.request_forget_after_run(context["username"])  # the tool
        order.append("run_done")  # SDK saves the turn's items here
        return "Done.", "skills", None, {}

    monkeypatch.setattr(legacy, "run_topology", fake_topology)
    legacy.run_agent("ix_run", None, "wipe our chats", None)
    assert order == ["run_done", "forget"]

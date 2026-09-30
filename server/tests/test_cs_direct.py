"""Obvious CS questions go straight to CS Navigator.

Measured 2026-09-30: router + two Claude Sonnet calls added ~4 s around the
CS Navigator lookup. The direct lane skips them and says "Let me check
that." only when the lookup is slow.
"""
import asyncio
import json

import pytest

from server import app_ws, cs_direct


@pytest.mark.parametrize("q", [
    "Who teaches COSC 220?", "what are the prereqs for cosc 350",
    "How many credits do I need to graduate?",
    "When are office hours for the CS department?",
    "Who is the chair of the computer science department?",
])
def test_obvious_cs_questions_take_the_direct_lane(q):
    assert cs_direct.is_direct_cs_question(q) is True


@pytest.mark.parametrize("q", [
    "I'm so stressed about COSC 220",
    "I'm failing my computer science major and I feel hopeless",
    "Hey NAO how are you", "Tell me about your classes",
    "What's the weather?", "",
])
def test_everything_else_takes_the_normal_path(q):
    assert cs_direct.is_direct_cs_question(q) is False


def test_shorten_strips_markdown_and_keeps_whole_sentences():
    ans = ("The prerequisite for **COSC 220 - Data Structures** is **COSC 112** "
           "with a C or higher. It is offered every fall. It is also offered "
           "every spring. Summer sections vary. Check WEBSIS.")
    out = cs_direct.shorten_for_speech(ans, max_sentences=3)
    assert "**" not in out
    assert out.count(".") == 3 and out.endswith("spring.")


def test_shorten_caps_a_runaway_sentence():
    out = cs_direct.shorten_for_speech("word " * 200, max_words=20)
    assert len(out.split()) <= 20


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_text(self, text):
        self.sent.append(json.loads(text))

    def spoken(self):
        return [f["text"] for f in self.sent if f.get("type") == "audio_chunk"]


class FakeHistory:
    items = []

    async def add_items(self, items):
        FakeHistory.items.extend(items)


@pytest.fixture
def lane(monkeypatch):
    from server import session as s
    from server.tools import cs_navigator as csn
    monkeypatch.setattr(app_ws, "_synth_for", lambda *a, **k: b"mp3")
    monkeypatch.setattr(s, "get_or_create_session", lambda u: FakeHistory())
    FakeHistory.items = []
    state = {"delay": 0.0, "answer": "COSC 220 is taught by Jin Guo."}

    async def fake_nav(ctx, q):
        await asyncio.sleep(state["delay"])
        return state["answer"]

    monkeypatch.setattr(csn, "_cs_navigator_search_impl", fake_nav)
    monkeypatch.setattr(app_ws, "CS_FILLER_AFTER_S", 0.05)
    return state


def _run(q="Who teaches COSC 220?"):
    ws, sess = FakeWS(), app_ws._Session("guest")
    asyncio.run(app_ws._emit_cs_direct(ws, sess, q, {}, 0.0))
    return ws


def test_fast_answer_has_no_filler(lane):
    ws = _run()
    assert ws.spoken() == ["COSC 220 is taught by Jin Guo."]
    assert FakeHistory.items[-1]["content"] == "COSC 220 is taught by Jin Guo."


def test_slow_answer_gets_a_filler_first(lane):
    lane["delay"] = 0.2
    ws = _run()
    assert ws.spoken() == [app_ws.CS_FILLER, "COSC 220 is taught by Jin Guo."]


def test_lookup_failure_speaks_the_apology(lane):
    from server.tools import cs_navigator as csn
    lane["answer"] = ""
    ws = _run()
    assert ws.spoken()[-1] == csn._FALLBACK_REPLY


def test_answer_is_not_heard_back_as_the_user(lane):
    _run()
    assert app_ws._is_substring_or_sentence_echo(
        "guest", "COSC 220 is taught by Jin Guo")

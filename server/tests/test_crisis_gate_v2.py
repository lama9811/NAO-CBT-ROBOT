"""Crisis gate v2: normalisation, missing phrasings, always-on classifier on
emotional / therapy-lane turns, and the stitched multi-turn check.

Before this, the gate matched contracted forms only ("I do not want to be
here anymore" missed), and the LLM ran only on a keyword hit, so indirect
wording never reached a model at all.
"""
from unittest.mock import patch

import pytest

from server import safety


@pytest.fixture(autouse=True)
def _always_on(monkeypatch):
    monkeypatch.delenv("CRISIS_ALWAYS_ON", raising=False)
    monkeypatch.delenv("CRISIS_BG_MIN_WORDS", raising=False)


# ───────── normalisation ─────────
@pytest.mark.parametrize("raw,expected", [
    ("I don't want to be here anymore", "i do not want to be here anymore"),
    ("I DONT want to be here anymore!", "i do not want to be here anymore"),
    ("I can’t go on.", "i can not go on"),
    ("I'm a burden...", "i am a burden"),
    ("I won't", "i will not"),
])
def test_normalize(raw, expected):
    assert safety.normalize(raw) == expected


# ───────── phrasings that used to slip through ─────────
@pytest.mark.parametrize("text", [
    "I do not want to be here anymore",
    "I want to disappear",
    "I feel like a burden to everyone",
    "I can't take it anymore",
    "I've been saving up my pills",
    "honestly kms",
])
def test_previously_missed_phrasings_reach_the_classifier(text):
    with patch("server.safety._llm_classify", return_value=True) as m:
        r = safety.crisis_check(text)
    assert r.positive is True and r.source == "llm"
    assert m.called


@pytest.mark.parametrize("text", [
    "I don't want to live anymore",
    "everyone would be better off without me",
    "I wish I was dead",
])
def test_new_hard_phrases_fire_without_a_model(text):
    with patch("server.safety._llm_classify") as m:
        r = safety.crisis_check(text)
    assert r.positive is True and r.source == "keyword"
    assert not m.called


@pytest.mark.parametrize("text", [
    "this homework is killing me",
    "I'm dying to see that movie",
    "what time is it",
])
def test_figures_of_speech_are_not_hard_hits(text):
    assert safety.hard_match(text) is False


# ───────── always-on classifier ─────────
def test_emotional_turn_without_keyword_asks_the_classifier():
    with patch("server.safety._llm_classify", return_value=True) as m:
        r = safety.crisis_check("I feel so empty and nobody would notice")
    assert m.called and r.positive and r.source == "llm"


def test_therapy_lane_turn_asks_the_classifier_even_with_neutral_words():
    with patch("server.safety._llm_classify", return_value=False) as m:
        r = safety.crisis_check("it happened again last night",
                                therapy_lane=True)
    assert m.called and r.positive is False


def test_small_talk_outside_the_lane_makes_no_model_call():
    with patch("server.safety._llm_classify") as m:
        r = safety.crisis_check("what is the weather like today")
    assert not m.called and r == safety.CrisisResult(False, "clean")


def test_always_on_classifier_error_fails_open():
    # Keyword layers are the fail-safe; an outage must not turn every
    # support reply into the hotline.
    with patch("server.safety._llm_classify", side_effect=RuntimeError("x")):
        r = safety.crisis_check("I feel so lonely lately", therapy_lane=True)
    assert r.positive is False and r.source == "clean_llm_error"


def test_soft_hit_classifier_error_still_fails_safe():
    with patch("server.safety._llm_classify", side_effect=RuntimeError("x")):
        r = safety.crisis_check("I want to disappear")
    assert r.positive is True and r.source == "failsafe"


def test_always_on_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("CRISIS_ALWAYS_ON", "0")
    with patch("server.safety._llm_classify") as m:
        r = safety.crisis_check("I feel so lonely lately", therapy_lane=True)
    assert not m.called and r.positive is False


# ───────── stitched multi-turn ─────────
def test_hard_phrase_split_across_turns_is_caught():
    with patch("server.safety._llm_classify") as m:
        r = safety.crisis_check("myself", recent=["sometimes I want to kill"])
    assert r.positive and r.source == "keyword" and r.stitched
    assert not m.called


def test_classifier_sees_the_stitched_text():
    seen = []

    def fake(text):
        seen.append(text)
        return True

    with patch("server.safety._llm_classify", side_effect=fake):
        r = safety.crisis_check("about not waking up tomorrow",
                                recent=["I keep thinking", "I feel hopeless"])
    assert r.positive
    assert seen[0] == ("I keep thinking I feel hopeless "
                       "about not waking up tomorrow")


def test_stitch_keeps_only_the_last_few_turns():
    recent = ["a", "b", "c", "d", "e"]
    assert safety.stitch("now", recent).split() == (
        recent[-safety.STITCH_TURNS:] + ["now"])


# ───────── reply text ─────────
def test_hotline_reply_has_988_and_morgan_counseling(monkeypatch):
    from server import config
    reply = safety.hotline_reply()
    assert "988" in reply
    assert config.MORGAN_COUNSELING_TEXT in reply
    assert "443-885-3130" in config.MORGAN_COUNSELING_TEXT
    assert reply.rstrip().endswith("?")


def test_hotline_reply_without_counseling_text(monkeypatch):
    from server import config
    monkeypatch.setattr(config, "MORGAN_COUNSELING_TEXT", "")
    assert safety.hotline_reply() == safety.HOTLINE_REPLY

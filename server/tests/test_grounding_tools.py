# -*- coding: utf-8 -*-
"""Grounding coach tools: paced breathing scripts and step tracking."""
import pytest

from server import breathing_pacing
from server.streaming import iter_sentences
from server.tools import grounding_tools as gt


@pytest.mark.parametrize("pattern", ["calm", "box", "4-7-8"])
def test_breathing_script_is_paced_with_real_seconds(pattern):
    out = gt._breathing_script_impl(pattern, 1)
    assert out["pattern"] == pattern
    phases = gt.BREATHING_PATTERNS[pattern]
    assert out["seconds_per_round"] == sum(s for _p, s in phases)
    sentences = [s for s in out["script"].split(".") if s.strip()]
    # One sentence per phase, each one expanded into a paced count.
    assert len(sentences) == len(phases)
    for sentence, (_phase, seconds) in zip(sentences, phases):
        chunks = breathing_pacing.expand_tts_pacing(sentence + ".")
        assert breathing_pacing.is_paced(chunks)
        assert len(chunks) == seconds


def test_breathing_script_drives_the_eye_phases():
    out = gt._breathing_script_impl("box", 1)
    seen = []
    for sentence in [s + "." for s in out["script"].split(".") if s.strip()]:
        chunks = breathing_pacing.expand_tts_pacing(sentence)
        cues = [c for c in breathing_pacing.breath_phase_cues(chunks) if c]
        seen.append(cues[0]["phase"])
    assert seen == ["inhale", "hold", "exhale", "hold"]


def test_breathing_script_survives_the_sentence_chunker():
    """The streaming chunker must not split a phase apart mid-count."""
    out = gt._breathing_script_impl("calm", 1)
    pieces = list(iter_sentences([out["script"]]))
    assert len(pieces) == 2
    assert all("<break" in p for p in pieces)


def test_unknown_pattern_and_rounds_are_clamped():
    out = gt._breathing_script_impl("whatever", 99)
    assert out["pattern"] == "calm"
    assert out["rounds"] == gt.MAX_ROUNDS
    assert gt._breathing_script_impl("box breathing", 0)["pattern"] == "box"
    assert gt._breathing_script_impl("box", 0)["rounds"] == 1


def test_grounding_step_remembers_its_place_between_turns():
    store = {"conv": {}}
    first = gt._grounding_step_impl(store, "5-4-3-2-1")
    assert first["step"] == 1 and "five things" in first["prompt"]
    # A fresh ctx dict each turn, sharing the same conv state.
    second = gt._grounding_step_impl({"conv": store["conv"]}, "54321")
    assert second["step"] == 2 and second["total_steps"] == 5
    for expected in (3, 4, 5):
        assert gt._grounding_step_impl(store, "5-4-3-2-1")["step"] == expected
    done = gt._grounding_step_impl(store, "5-4-3-2-1")
    assert done["done"] is True
    assert "grounding" not in store["conv"]


def test_grounding_step_explicit_and_switching_exercise():
    store = {"conv": {}}
    assert gt._grounding_step_impl(store, "body scan", 3)["step"] == 3
    # Switching exercise starts at the top.
    assert gt._grounding_step_impl(store, "5-4-3-2-1")["step"] == 1


def test_grounding_step_without_conv_state_and_unknown_exercise():
    assert gt._grounding_step_impl({}, "body_scan")["step"] == 1
    assert "error" in gt._grounding_step_impl({}, "yoga")


def test_grounding_coach_has_real_tools_and_no_camera():
    from server.agents.grounding_coach import build_grounding_coach_agent
    agent = build_grounding_coach_agent("tester")
    names = {t.name for t in agent.tools}
    assert "observe_face" not in names
    assert {"breathing_script", "grounding_step", "log_emotion"} <= names

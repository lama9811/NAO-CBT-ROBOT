"""A healthy thought must be allowed to be healthy.

`_classify_distortion` prompted "Choose exactly ONE from" the ten labels in
`_DISTORTIONS`, with no none-of-these option — so a balanced thought got a
label anyway. Verified against the live Pi before the fix:

    "I studied hard, I did well, and I feel good about it."
      -> magnification/minimization
      -> "This thought actually seems balanced and kind toward yourself —
          there's no distortion here, just a fair recognition..."

The model knew. The prompt wouldn't let it say so, and NAO told the student
their balanced thinking was a cognitive distortion. In a CBT tool that is the
wrong direction of error: pathologising healthy thought.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from server.tools import emotion


def test_none_is_an_allowed_answer_in_the_prompt():
    """The model can only return 'none' if the prompt offers it."""
    captured = {}

    def _fake_chat(**kwargs):
        captured["system"] = kwargs["messages"][0]["content"]
        return '{"distortion": "none", "explanation": "Balanced thinking."}'

    with patch.object(emotion.llm_compat, "chat", _fake_chat):
        emotion._classify_distortion("I studied hard and did well.")

    assert "none" in captured["system"].lower()


def test_healthy_thought_reports_none(monkeypatch):
    with patch.object(
        emotion, "_classify_distortion",
        return_value={"distortion": "none",
                      "explanation": "That reads as balanced and fair."},
    ):
        out = emotion._identify_distortion_impl(
            "I studied hard, I did well, and I feel good about it."
        )

    assert out["distortion"] == "none"


@pytest.mark.parametrize(
    "raw", ["none", "None", "NONE", " none ", "no distortion", "n/a", ""]
)
def test_none_variants_normalise(raw):
    """The model won't always spell it the way the prompt asked."""
    assert emotion._is_no_distortion(raw) is True


@pytest.mark.parametrize("raw", ["catastrophizing", "mind reading", "shoulds"])
def test_real_distortions_are_not_mistaken_for_none(raw):
    assert emotion._is_no_distortion(raw) is False


def test_no_thought_record_is_written_for_a_healthy_thought(monkeypatch):
    """A 'none' result must not land in thought_records as a distortion."""
    saved: list[tuple] = []
    monkeypatch.setattr(
        emotion, "_persist_thought",
        lambda ctx, thought, distortion: saved.append((thought, distortion)),
    )
    with patch.object(
        emotion, "_identify_distortion_impl",
        return_value={"distortion": "none", "explanation": "Balanced."},
    ):
        emotion.identify_distortion.on_invoke_tool  # tool wrapper exists
        out = emotion._identify_distortion_and_persist(
            {}, "I studied hard, I did well, and I feel good about it."
        )

    assert out["distortion"] == "none"
    assert saved == [], "healthy thought was recorded as a distortion"


def test_real_distortion_is_still_recorded(monkeypatch):
    saved: list[tuple] = []
    monkeypatch.setattr(
        emotion, "_persist_thought",
        lambda ctx, thought, distortion: saved.append((thought, distortion)),
    )
    with patch.object(
        emotion, "_identify_distortion_impl",
        return_value={"distortion": "catastrophizing", "explanation": "..."},
    ):
        out = emotion._identify_distortion_and_persist(
            {}, "If I fail this the whole semester is ruined."
        )

    assert out["distortion"] == "catastrophizing"
    assert saved == [("If I fail this the whole semester is ruined.",
                      "catastrophizing")]


def test_reframe_declines_to_reframe_a_healthy_thought():
    """Asking for alternatives to a thought exhibiting 'none' is nonsense."""
    called = []

    with patch.object(emotion, "_reframe_impl", lambda t, d: called.append(d) or ["x"]):
        out = emotion._suggest_reframe_and_persist(
            {}, "I studied hard and did well.", "none"
        )

    assert out == []
    assert called == [], "sent a 'none' distortion to the reframe model"


def test_coach_prompt_tells_the_agent_what_to_do_with_none():
    """The tool can return 'none' — the coach has to know not to invent one."""
    from server.agents import cbt_coach

    assert "none" in cbt_coach._BASE.lower()

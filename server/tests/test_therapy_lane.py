"""The therapy lane holds across follow-ups that carry no trigger word.

Routing used to be recomputed from keywords every turn, so "yeah, about
an 8" after "I'm so anxious" dropped from the therapist back to plain chat,
losing the CBT tools mid-exercise. And a clearly emotional turn paid a
router hop to reach a decision Python had already made.
"""
from __future__ import annotations

from server import conversation_state as cs
from server.agents import pick_initial_agent


def _turn(conv: dict, text: str) -> str:
    """Pick the agent for ``text`` and record it as the turn's answerer."""
    agent = pick_initial_agent("alice", None, text, conv=conv)
    cs.note_turn(conv, agent.name)
    return agent.name


def test_emotional_turn_skips_the_router():
    assert _turn({}, "I've been feeling really lonely lately") == "therapist"


def test_follow_up_without_trigger_word_stays_in_therapy():
    conv: dict = {}
    assert _turn(conv, "I'm so anxious about tomorrow") == "therapist"
    assert _turn(conv, "yeah, about an 8") == "therapist"
    assert _turn(conv, "I guess it's mostly my roommate") == "therapist"


def test_without_conv_follow_up_goes_to_plain_chat():
    # The old behaviour, still what callers without state get.
    assert pick_initial_agent("alice", None, "yeah, about an 8").name == "chat"


def test_school_feelings_do_not_leave_the_lane():
    conv: dict = {}
    _turn(conv, "I'm overwhelmed")
    assert _turn(conv, "I'm behind in my class and my dog died") == "therapist"


def test_factual_question_leaves_the_lane():
    conv: dict = {}
    _turn(conv, "I'm stressed")
    assert _turn(conv, "who teaches COSC 220?") == "router"
    assert "lane" not in conv
    assert _turn(conv, "cool, thanks") == "chat"


def test_explicit_topic_change_leaves_the_lane():
    conv: dict = {}
    _turn(conv, "I'm worried")
    assert _turn(conv, "can we talk about something else") == "chat"
    assert "lane" not in conv


def test_utility_question_leaves_the_lane():
    conv: dict = {}
    _turn(conv, "I'm sad")
    assert _turn(conv, "what time is it?") == "router"


def test_under_the_weather_is_not_a_weather_question():
    conv: dict = {}
    _turn(conv, "I'm stressed")
    assert _turn(conv, "I've been under the weather all week") == "therapist"


def test_goodbye_gets_one_close_turn_then_lane_ends():
    conv: dict = {}
    _turn(conv, "I'm anxious")
    assert _turn(conv, "okay I have to go now") == "therapist"
    assert conv["therapy_closing"] == 1
    # Answering the optional-homework offer is still the therapist.
    assert _turn(conv, "sure, a walk sounds good") == "therapist"
    # Then the lane is over.
    assert _turn(conv, "what do you think about robots") == "chat"
    assert "lane" not in conv and "therapy_closing" not in conv


def test_lane_ends_right_after_recap_is_written():
    conv: dict = {}
    _turn(conv, "I'm anxious")
    assert _turn(conv, "bye") == "therapist"
    conv["therapy_closed"] = True  # finalize_session_recap ran
    assert _turn(conv, "tell me a joke") == "chat"


def test_maybe_is_not_bye():
    conv: dict = {}
    _turn(conv, "I'm anxious")
    _turn(conv, "maybe")
    assert "therapy_closing" not in conv


def test_cbt_coach_resumes_while_record_is_mid_walk():
    conv: dict = {}
    cs.set_lane(conv, "cbt_coach")
    conv["cbt_step"] = "4a"
    assert _turn(conv, "well, I did fail one quiz") == "cbt_coach"


def test_finished_or_stopped_record_resumes_the_therapist():
    for step in ("done", "stopped", None):
        conv: dict = {}
        cs.set_lane(conv, "cbt_coach")
        if step:
            conv["cbt_step"] = step
        assert _turn(conv, "okay") == "therapist"


def test_grounding_lane_resumes_the_therapist():
    conv: dict = {}
    cs.set_lane(conv, "grounding_coach")
    assert _turn(conv, "okay, my chest is still tight") == "therapist"


def test_lane_lapses_after_ttl():
    conv: dict = {}
    cs.set_lane(conv, "therapist", now=0.0)
    # lane_ts=0 is far older than the TTL from the real clock.
    assert _turn(conv, "yeah") == "chat"


def test_explicit_hints_still_win():
    conv: dict = {}
    cs.set_lane(conv, "therapist")
    assert pick_initial_agent("alice", "chat", "yeah", conv=conv).name == "chat"
    assert pick_initial_agent("alice", "morgan", "yeah", conv=conv).name == "chatbot"


def test_note_turn_closes_lane_for_non_support_agents():
    conv: dict = {}
    cs.note_turn(conv, "therapist")
    assert cs.active_lane(conv) == "therapist"
    cs.note_turn(conv, "chatbot")
    assert cs.active_lane(conv) is None


def test_conversation_state_records_start_time():
    cs._reset_for_tests()
    st = cs.state_for("Lane_Tester", now=1234.0)
    assert st["started_at"] == 1234.0
    assert cs.state_for("Lane_Tester", now=1240.0)["started_at"] == 1234.0

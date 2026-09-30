"""CS questions must reach the router, and small talk must not.

The keyword pre-router decides whether a turn pays the router hop at all.
Anything it misses is answered by general chat, which knows nothing about
Morgan State and can invent an answer. "Who teaches COSC 220?" matched no
keyword until 2026-09-30.
"""
import pytest

from server.agents import _needs_specialist_router


@pytest.mark.parametrize("q", [
    "Who teaches COSC 220?",
    "who teaches cosc220",
    "What's COSC-111 about?",
    "Is Math 241 required?",
    "How many credits do I need to graduate?",
    "What electives can I take next semester?",
    "Who is my advisor?",
    "What are the prereqs for data structures?",
    "When are Dr. Ali's office hours?",
    "Tell me about the computer science internship program",
    "How do I register for spring classes?",
])
def test_cs_questions_reach_the_router(q):
    assert _needs_specialist_router(q) is True


@pytest.mark.parametrize("q", [
    "Hey NAO, how are you?",
    "Tell me a joke",
    "What's the capital of France?",
    "I love pizza",
    "Can you dance for me?",
    "What do you think about robots taking over?",
])
def test_small_talk_stays_on_the_fast_path(q):
    assert _needs_specialist_router(q) is False


def test_router_prompt_sends_school_feelings_to_therapist():
    from server.agents import router
    text = router.ROUTER_INSTRUCTIONS if hasattr(
        router, "ROUTER_INSTRUCTIONS") else router.SYSTEM
    assert "stressed about my classes" in text
    assert "Who teaches COSC 220?" in text

"""Every agent that can answer a user must send Morgan CS questions to CS
Navigator rather than answer from the model's own knowledge.

Before 2026-09-30 the everyday chat agent had no tools at all, so a CS
question that missed the router's keywords was answered from memory.
"""
import pytest


def _agents():
    from server.agents.chat import pure_chat_agent, chat_embodied_agent
    from server.agents.skills import skills_agent
    from server.agents.therapist import build_therapist_agent
    from server.agents.chatbot import chatbot_agent
    return [pure_chat_agent, chat_embodied_agent, skills_agent,
            build_therapist_agent("guest"), chatbot_agent]


def _instructions(agent):
    ins = agent.instructions
    return ins(None, agent) if callable(ins) else ins


@pytest.mark.parametrize("idx", range(5))
def test_agent_has_the_cs_navigator_tool(idx):
    agent = _agents()[idx]
    names = [getattr(t, "name", "") for t in agent.tools]
    assert "cs_navigator_search" in names, agent.name


@pytest.mark.parametrize("idx", range(4))
def test_agent_prompt_carries_the_cs_rule(idx, monkeypatch):
    from server import memory, session, memory_rollup
    monkeypatch.setattr(memory, "build_context_preamble", lambda u: "")
    monkeypatch.setattr(session, "load_recent_recaps", lambda u, n=3: [])
    monkeypatch.setattr(memory_rollup, "load_week_themes", lambda u, n=1: [])
    monkeypatch.setattr(memory_rollup, "load_month_personas", lambda u, n=1: [])
    agent = _agents()[idx]
    text = _instructions(agent)
    assert "MORGAN STATE CS QUESTIONS" in text, agent.name
    assert "cs_navigator_search" in text


def test_router_hands_cs_questions_to_the_cs_agent():
    from server.agents import router
    assert "looks it up in CS Navigator" in router.SYSTEM

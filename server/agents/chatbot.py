"""Chatbot specialist — Morgan State CS knowledge via the CS Navigator API.

Phase 5 of the v2 rework moves Morgan-CS knowledge off the in-process Vertex
AI Search client and onto the operator's deployed CS Navigator Cloud Run
service (see ``docs/PHASE_5_TASK_MAP.md``). The new tool returns one clean,
already-RAG-enriched answer string instead of raw passages, so the agent's
job shrinks to: ask CS Navigator, then re-voice the reply for NAO.

CS Navigator is the only knowledge source. The old in-process Vertex AI
Search fallback was removed on 2026-09-30: it only ran if this import failed,
and it had not run once that month.
"""
from agents import Agent, ModelSettings
from server import config
from server.model_factory import resolve_model
from server.agents._memory_inject import with_memory_preamble

from server.tools.cs_navigator import cs_navigator_search as _SEARCH_TOOL

_SEARCH_TOOL_NAME = "cs_navigator_search"


# Agent prompt. We deliberately don't
# mention "embeddings", "Pinecone", "Vertex", or "RAG" — those are
# implementation details the user-facing voice should never leak.
SYSTEM = (
    "You are NAO, the Morgan State University Computer Science department's "
    "humanoid robot assistant. For ANY question about Morgan's CS curriculum, "
    "courses, faculty, schedule, advising, deadlines, or graduation "
    "requirements, ALWAYS call `{tool}(query)` first with the user's question, "
    "then re-voice the returned answer in your own warm, brief, conversational "
    "tone. Keep replies tight — 25 words or fewer per turn — because they are "
    "spoken aloud, not read. If the tool returns an apology or empty result, "
    "say you're not sure and offer to look again. Never paste raw passages, "
    "URLs, or filenames into your reply — synthesize one clean spoken sentence."
).format(tool=_SEARCH_TOOL_NAME)


chatbot_agent = Agent(
    name="chatbot",
    instructions=with_memory_preamble(SYSTEM),
    model=resolve_model(config.CHATBOT_MODEL),
    model_settings=ModelSettings(max_tokens=config.MINI_MAX_TOKENS),
    tools=[_SEARCH_TOOL],
)

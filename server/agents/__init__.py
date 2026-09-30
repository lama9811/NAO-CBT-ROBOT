"""Agent graph builders."""
import re
from server.agents.chat import (
    chat_agent, chat_embodied_agent, pure_chat_agent,
)
from server.agents.chatbot import chatbot_agent
from server.agents.skills import skills_agent
from server.agents.therapist import build_therapist_agent
from server.agents.router import build_router


# Phase 11.11 — embodiment trigger keywords. When the transcript inside
# a hint='chat' turn matches any of these, route to chat_embodied_agent
# (which has the 18 NAO action tools + 10 gestures). Otherwise stay on
# pure_chat_agent for lower-variance, sub-2s replies.
#
# Ordered short-to-broad. Matched as case-insensitive substrings.
_EMBODIED_TRIGGERS: tuple[str, ...] = (
    # explicit motion verbs
    "dance", "wave", "spin", "kneel", "stand up", "sit down",
    "follow me", "follow movement", "stop following",
    "move forward", "move back", "step forward", "step back",
    "turn left", "turn right", "rotate",
    # explicit body action requests
    "show me", "use your body", "use your hand", "use your arm",
    "raise your hand", "lift your hand",
    "nod", "shake your head", "shake head", "clap", "applaud",
    # gesture-class
    "gesture", "do a gesture", "make a gesture",
    # voice picker (Phase 11.8 voice profile)
    "switch voice", "change voice", "voice 1", "voice 2", "voice 3",
    "girl voice", "man voice", "neutral voice", "female voice",
    "male voice",
    # LED color changes
    "eye color", "eyes red", "eyes blue", "eyes green", "eyes yellow",
    "eyes white", "eyes purple", "led",
    # animation library
    "play animation", "do an animation", "animate",
    "elephant", "gorilla", "gorrila", "monkey", "dragon", "dinosaur",
    "lion", "tiger", "bear", "bird", "eagle", "chicken", "penguin",
    "duck", "rabbit", "cat", "dog", "horse", "snake", "spider",
    "shark", "frog", "animal",
    "kung fu", "kung-fu", "air guitar", "headbang", "head bang",
    "bandmaster", "conductor", "helicopter", "knight", "monster",
    "magic", "wizard", "spaceship", "space shuttle", "rocket",
    "zombie", "waddle", "claw", "wings",
)

_SPECIALIST_TRIGGERS: tuple[str, ...] = (
    # Morgan / CS advising. These only send the question to the router,
    # which makes the real call, so erring broad costs one router hop while
    # missing a word means a CS question is answered from general chat
    # (and possibly made up). "Who teaches COSC 220?" matched nothing here
    # until 2026-09-30. Course codes are matched separately below.
    "morgan", "msu", "course", "class", "faculty", "professor", "advising",
    "advisor", "prerequisite", "pre-req", "prereq", "degree requirement",
    "major requirement", "computer science", "comp sci",
    "cs department", "cs major", "cs minor", "department", "schedule",
    "who teaches", "instructor", "lecturer", "office hours",
    "semester", "credit", "credits", "gpa", "graduat", "curriculum",
    "syllabus", "elective", "internship", "co-op", "research lab",
    "enroll", "register for", "registration", "websis", "degreeworks",
    "degree works", "transfer credit", "concentration",
    "cosc", "cybersecurity program", "data science program",
    # Utility lane.
    "what time", "today's date", "what date", "weather", "timer",
    "remind me", "reminder", "todo", "to-do",
    # Emotional / support lane. Keep this conservative so normal podcast
    # debate does not get therapy phrasing.
    "anxious", "anxiety", "panic", "depressed", "depression", "sad",
    "lonely", "overwhelmed", "stressed", "stress", "worried", "worry",
    "grief", "hopeless", "suicidal", "kill myself", "self harm",
    "therapy", "therapist", "cbt", "grounding exercise", "breathing exercise",
)


# "COSC 220", "cosc220", "Math 241", "CS 351": a department code plus a
# three-digit number is a course, whatever else the sentence says.
_COURSE_CODE_RE = re.compile(
    r"\b(?:cosc|cs|math|eeng|ieng|clcs|phys|chem|biol|stat|engl|comp)"
    r"\s*-?\s*\d{3}\b", re.IGNORECASE)


def _wants_embodied(transcript: str | None) -> bool:
    """True if the transcript suggests the user wants a robot action."""
    t = (transcript or "").lower()
    if not t:
        return False
    return any(kw in t for kw in _EMBODIED_TRIGGERS)


def _needs_specialist_router(transcript: str | None) -> bool:
    """True when a default turn should pay the router hop.

    Most podcast/small-talk/opinion turns should go straight to the fast chat
    lane. We keep the router for clear Morgan, utility, or emotional-support
    turns where a specialist handoff is useful.
    """
    t = (transcript or "").lower()
    if not t:
        return True
    if _COURSE_CODE_RE.search(t):
        return True
    return any(kw in t for kw in _SPECIALIST_TRIGGERS)


def pick_initial_agent(username: str, hint: str | None,
                        transcript: str | None = None):
    """Return the agent to start a turn with, based on hint + transcript.

    Phase 11.11: hint='chat' splits into pure_chat (default, no tools)
    vs chat_embodied (when the transcript triggers an embodiment keyword).

    Default (no hint) uses a fast local pre-router. Bare "nao" wake should
    feel like a normal robot conversation first; obvious podcast/chat/action
    turns go straight to chat, while clear emotional/CBT, Morgan, and utility
    turns still pay the router hop for specialist selection.
    """
    if hint == "chat":
        if _wants_embodied(transcript):
            return chat_embodied_agent
        return pure_chat_agent
    if hint == "morgan":
        return chatbot_agent
    if hint == "therapy":
        return build_therapist_agent(username)
    if hint == "skills":
        return skills_agent
    if hint == "router":
        return build_router(username)
    # Default: fast chat unless the text clearly needs a specialist.
    if _wants_embodied(transcript):
        return chat_embodied_agent
    if _needs_specialist_router(transcript):
        return build_router(username)
    return pure_chat_agent

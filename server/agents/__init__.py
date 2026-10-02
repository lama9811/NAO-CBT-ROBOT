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

# Emotional / support lane. Keep this conservative so normal podcast debate
# does not get therapy phrasing.
_SUPPORT_TRIGGERS: tuple[str, ...] = (
    "anxious", "anxiety", "panic", "depressed", "depression", "sad",
    "lonely", "overwhelmed", "stressed", "stress", "worried", "worry",
    "grief", "hopeless", "suicidal", "kill myself", "self harm",
    "therapy", "therapist", "cbt", "grounding exercise", "breathing exercise",
)

_FACT_TRIGGERS: tuple[str, ...] = (
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
)

_SPECIALIST_TRIGGERS: tuple[str, ...] = _FACT_TRIGGERS + _SUPPORT_TRIGGERS


# Inside the therapy lane, short follow-ups ("yeah", "about an 8", "I
# guess my roommate") carry no trigger word, and re-routing them from
# scratch dropped the student out of therapy mid-exercise. The lane holds
# until one of these: a factual question for another specialist, or the
# student plainly changing the subject. Deliberately narrow -- "class",
# "course" and the animal words are NOT here, because "I'm behind in my
# class" and "my dog died" are therapy-lane sentences.
_LEAVE_LANE_PHRASES: tuple[str, ...] = (
    "who teaches", "prerequisite", "prereq", "pre-req", "office hours",
    "what time is it", "what's the time", "today's date", "what's the date",
    "what date is it", "what's the weather", "weather like",
    "weather today", "weather forecast", "set a timer", "timer for",
    "set a reminder", "add a todo", "add a to-do", "my todo", "my to-do",
    "change the subject", "talk about something else", "different topic",
    "new topic", "let's just chat",
)

# Goodbyes inside the lane: the therapist gets this turn for a short close
# (optional homework, recap), then the lane ends.
_CLOSE_PHRASES: tuple[str, ...] = (
    "goodbye", "bye", "see you", "i have to go", "i've got to go",
    "i gotta go", "i need to go", "i should go", "that's all for today",
    "that's all for now", "talk later", "gotta run",
    "i'm good now", "i am good now", "i'm fine now", "i feel better now",
    "i'm feeling better now", "that helped",
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


def _has_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    """Whole-word-ish phrase match ("bye" must not match "maybe")."""
    text = text.replace("\u2019", "'")
    for p in phrases:
        if re.search(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])", text):
            return True
    return False


def _leaves_therapy_lane(transcript: str | None) -> bool:
    t = (transcript or "").lower()
    if not t:
        return False
    if _COURSE_CODE_RE.search(t):
        return True
    return _has_phrase(t, _LEAVE_LANE_PHRASES)


def _is_closing(transcript: str | None) -> bool:
    return _has_phrase((transcript or "").lower(), _CLOSE_PHRASES)


def _clearly_support(transcript: str | None) -> bool:
    """An emotional turn with no factual (Morgan / utility) words in it.

    Python has already made the router's decision for these, so paying a
    second sequential model call to repeat it only added ~3 s (measured
    5.7 s therapist alone vs 8.9 s via the router).
    """
    t = (transcript or "").lower()
    if not t or _COURSE_CODE_RE.search(t):
        return False
    if any(kw in t for kw in _FACT_TRIGGERS):
        return False
    return any(kw in t for kw in _SUPPORT_TRIGGERS)


def _lane_agent(username: str, lane: str, conv: dict):
    """The support agent that resumes an open therapy lane.

    The CBT coach resumes only while a thought record is mid-walk; the
    coaches cannot hand back, so every other case resumes the therapist,
    which can hand off again if it needs to.
    """
    step = str(conv.get("cbt_step") or "")
    if lane == "cbt_coach" and step and step not in ("done", "stopped"):
        from server.agents.cbt_coach import build_cbt_coach_agent
        return build_cbt_coach_agent(username)
    return build_therapist_agent(username)


def _resume_lane(username: str, transcript: str | None, conv: dict | None):
    """Agent to continue the therapy lane with, or None to route afresh."""
    if conv is None:
        return None
    from server import conversation_state as cs
    lane = cs.active_lane(conv)
    if not lane:
        conv.pop("therapy_closing", None)
        conv.pop("therapy_closed", None)
        return None
    if _leaves_therapy_lane(transcript):
        cs.leave_lane(conv)
        conv.pop("therapy_closing", None)
        return None
    closing = int(conv.get("therapy_closing") or 0)
    if closing:
        # The goodbye was last turn. Once the recap is written the close is
        # done; otherwise allow one more turn (the student answering the
        # optional-homework offer) before the lane ends.
        if conv.pop("therapy_closed", None) or closing >= 2:
            conv.pop("therapy_closing", None)
            cs.leave_lane(conv)
            return None
        conv["therapy_closing"] = closing + 1
        return build_therapist_agent(username)
    if _is_closing(transcript):
        conv["therapy_closing"] = 1
        conv.pop("therapy_closed", None)
        return build_therapist_agent(username)
    return _lane_agent(username, lane, conv)


def pick_initial_agent(username: str, hint: str | None,
                        transcript: str | None = None,
                        conv: dict | None = None):
    """Return the agent to start a turn with, based on hint + transcript.

    Phase 11.11: hint='chat' splits into pure_chat (default, no tools)
    vs chat_embodied (when the transcript triggers an embodiment keyword).

    Default (no hint) uses a fast local pre-router. Bare "nao" wake should
    feel like a normal robot conversation first; obvious podcast/chat/action
    turns go straight to chat, while Morgan and utility turns still pay the
    router hop for specialist selection.

    ``conv`` is the conversation's persistent state (``ctx["conv"]``).
    With it, an open therapy lane holds across follow-ups that carry no
    trigger word, and a clearly emotional turn goes straight to the
    therapist instead of through the router.
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
    resumed = _resume_lane(username, transcript, conv)
    if resumed is not None:
        return resumed
    # Default: fast chat unless the text clearly needs a specialist.
    if _wants_embodied(transcript):
        return chat_embodied_agent
    if _clearly_support(transcript):
        return build_therapist_agent(username)
    if _needs_specialist_router(transcript):
        return build_router(username)
    return pure_chat_agent

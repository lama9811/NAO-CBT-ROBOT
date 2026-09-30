"""Direct lane for obvious Morgan CS questions.

The normal path for a CS question was: router (Claude Haiku) -> CS agent
(Claude Sonnet) decides to call CS Navigator -> CS Navigator -> Sonnet
re-voices the answer. Measured 2026-09-30, the three model calls added ~4 s
on top of CS Navigator's own 3.5-6.5 s. CS Navigator already answers in
short, direct sentences ("COSC 220 is taught by Radwan Shushane and Jin
Guo"), so for questions that are unmistakably CS we skip all three model
calls and speak its answer directly.

Only unambiguous questions take this lane. Anything with emotional
language goes the normal way so the router can hand it to the therapist
("I'm stressed about my COSC 220 class" is not a lookup).
"""
from __future__ import annotations

import re

from server.agents import _COURSE_CODE_RE

# Phrases that on their own make a question a CS lookup. Deliberately
# narrower than the router keyword list: that list only costs a router hop
# when it over-matches, whereas this one skips the router entirely.
_DIRECT_PHRASES: tuple[str, ...] = (
    "who teaches", "who is teaching", "who's teaching", "prerequisite",
    "prereq", "pre-req", "office hours", "credits do i need",
    "credits to graduate", "how many credits", "cs department",
    "computer science department", "comp sci department", "cs major",
    "computer science major", "cs minor", "computer science minor",
    "cs advisor", "academic advisor", "degree requirement",
    "major requirement", "cs elective", "computer science elective",
    "websis", "degreeworks", "degree works", "syllabus for",
    "department chair", "chair of the computer science",
    "chair of the cs",
)

# Emotional language sends the turn through the router instead, so a
# feeling that happens to mention a course reaches the therapist.
_EMOTIONAL: tuple[str, ...] = (
    "stress", "anxious", "anxiety", "worried", "worry", "scared", "afraid",
    "sad", "depress", "overwhelm", "panic", "lonely", "hopeless", "cry",
    "hate", "failing", "upset", "frustrat", "struggl", "hurt", "kill",
    "suicid", "self harm", "give up",
)


def is_direct_cs_question(transcript: str | None) -> bool:
    t = (transcript or "").lower()
    if not t.strip():
        return False
    if any(w in t for w in _EMOTIONAL):
        return False
    if _COURSE_CODE_RE.search(t):
        return True
    return any(p in t for p in _DIRECT_PHRASES)


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def shorten_for_speech(text: str, max_sentences: int = 3,
                       max_words: int = 60) -> str:
    """Trim a CS Navigator answer to something NAO can say in a breath.

    Strips Markdown first (CS Navigator answers for a web UI), keeps whole
    sentences, and stops at ``max_sentences`` or ``max_words``, whichever
    comes first. Never cuts mid-sentence unless a single sentence is itself
    longer than ``max_words``.
    """
    from server.tts_text import to_speakable

    clean = to_speakable(text or "").strip()
    if not clean:
        return ""
    sentences = [s.strip() for s in _SENTENCE_RE.split(clean) if s.strip()]
    out: list[str] = []
    words = 0
    for s in sentences:
        n = len(s.split())
        if out and (len(out) >= max_sentences or words + n > max_words):
            break
        out.append(s)
        words += n
    result = " ".join(out)
    if len(result.split()) > max_words:
        result = " ".join(result.split()[:max_words]).rstrip(",;:") + "."
    return result

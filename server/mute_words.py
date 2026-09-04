# -*- coding: utf-8 -*-
"""Spoken mute / unmute command matching.

Kept separate from ``app_ws`` for two reasons: it is pure text logic that
unit-tests without a WebSocket or an event loop, and the phrase lists are
the part most likely to need tuning against real STT output.

The matcher is deliberately **strict**. A false positive here silences the
robot mid-sentence for a word the user never said, which reads as a crash;
a false negative just means they repeat themselves. So the phrases must
appear as whole words, near the start of the utterance, in a short one.
"""
from __future__ import annotations

import re

# Order matters: "mute" is a substring of "unmute", so unmute is always
# tested first. Keeping them in one module makes that ordering hard to
# get wrong at the call site.
_UNMUTE_PHRASES = (
    "unmute",
    "un mute",
    "you can talk",
    "you can speak",
    "start talking",
    "speak again",
    "talk again",
)

_MUTE_PHRASES = (
    "mute",
    "be quiet",
    "stop talking",
    "quiet please",
)

# Wake-word prefixes the user naturally puts in front ("Nao, mute"). Also
# covers what STT actually returns for "NAO" -- it is routinely heard as
# "now" or "no", and rejecting those would make the feature look broken.
_ADDRESS_PREFIX = re.compile(
    r"^\s*(hey\s+|ok\s+|okay\s+)?(nao|now|no|nau|neo)\b[\s,.!]*",
    re.IGNORECASE,
)

# A command is short. Anything longer is a sentence that merely contains
# the word ("I had to mute my laptop"), which must not trigger.
_MAX_COMMAND_WORDS = 4

_PUNCT_RE = re.compile(r"[^\w\s]+")
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    t = _PUNCT_RE.sub(" ", str(text).lower())
    return _WS_RE.sub(" ", t).strip()


def _strip_address(text: str) -> str:
    """Remove a leading "nao"/"hey nao" address term, if present."""
    return _ADDRESS_PREFIX.sub("", text).strip()


def _matches(text: str, phrases: tuple[str, ...]) -> bool:
    """True when `text` is a short command built around one of `phrases`.

    Requires a whole-word hit -- substring matching would fire "mute" on
    "commuter" -- and caps the utterance length so only an actual command
    counts, not a sentence that happens to mention the word.
    """
    if not text:
        return False
    if len(text.split()) > _MAX_COMMAND_WORDS:
        return False
    for p in phrases:
        if re.search(r"\b%s\b" % re.escape(p), text):
            return True
    return False


def _tokens(text: str) -> list[str]:
    return _normalize(text).split()


def strip_leading_echo(transcript: str, spoken_text: str,
                       min_stripped: int = 3) -> str:
    """Drop NAO's own words off the front of a transcript.

    The mic stays open while NAO talks, so its voice and the user's land in
    one buffer and arrive as a single long transcript:

        "thanks for asking how are things going with you now on mute"
         \________ NAO's own reply _________________/ \__ user __/

    `classify` caps a command at four words, so a transcript like that never
    matches and the spoken command is lost -- which is exactly how the
    feature failed the first time it was tried on the robot.

    Walk the transcript from the left, dropping tokens while they appear in
    what NAO just said, and stop at the first token that does not. What
    remains is the user's own words, which the ordinary strict matcher can
    then judge.

    `min_stripped` keeps this from firing on a normal utterance that merely
    opens with a common word: unless a real echo of at least that many
    tokens was removed, the transcript is returned untouched.
    """
    if not transcript or not spoken_text:
        return transcript
    t_tokens = _tokens(transcript)
    spoken = set(_tokens(spoken_text))
    if not t_tokens or not spoken:
        return transcript
    i = 0
    while i < len(t_tokens) and t_tokens[i] in spoken:
        i += 1
    if i < min_stripped or i == len(t_tokens):
        # Either nothing meaningful was echoed, or the whole transcript was
        # NAO talking to itself and there is no user command hiding in it.
        return transcript
    return " ".join(t_tokens[i:])


def classify_with_echo(transcript: str, spoken_text: str) -> str | None:
    """`classify`, retried against the transcript minus NAO's own speech.

    The plain transcript is tried first so nothing about the existing
    behaviour changes; the echo-stripped retry only ever turns a `None`
    into a match, never the reverse.
    """
    direct = classify(transcript)
    if direct is not None:
        return direct
    stripped = strip_leading_echo(transcript, spoken_text)
    if stripped == transcript:
        return None
    return classify(stripped)


def classify(transcript: str) -> str | None:
    """Return ``"mute"``, ``"unmute"``, or ``None`` for a transcript.

    ``None`` means "not a mute command" -- the caller should treat the
    transcript as ordinary speech.
    """
    norm = _strip_address(_normalize(transcript))
    if not norm:
        return None
    # Unmute first: "unmute" contains "mute".
    if _matches(norm, _UNMUTE_PHRASES):
        return "unmute"
    if _matches(norm, _MUTE_PHRASES):
        return "mute"
    return None


def is_self_trigger(command: str, spoken_text: str) -> bool:
    """True when NAO's *own* reply contains the command word.

    The mic stays open while NAO speaks so it can hear "mute", which means
    NAO also hears itself. If the sentence it is currently speaking
    contains "mute" ("I'll mute myself now"), the transcript that comes
    back is NAO's voice, not the user's, and acting on it would make the
    robot silence itself at random. Callers suppress the trigger in that
    case.
    """
    if not command or not spoken_text:
        return False
    norm = _normalize(spoken_text)
    phrases = _UNMUTE_PHRASES if command == "unmute" else _MUTE_PHRASES
    for p in phrases:
        # Leading word boundary only, deliberately: NAO inflects ("I am
        # unmuted now", "muting myself"), and this guard should err toward
        # suppressing. A false positive here only ignores one possible
        # command; a false negative lets the robot silence itself at random.
        if re.search(r"\b%s" % re.escape(p), norm):
            return True
    return False

"""Small conversational-feel policies for the live WS path.

Pure functions and env knobs only -- ``app_ws`` owns the I/O. Kept apart
so each rule can be tested without a WebSocket:

* **Spoken repair** -- when a turn is thrown away as noise / silence /
  non-English, NAO says "Sorry, I didn't catch that." instead of going
  silent. Rate-limited so background noise cannot make it nag.
* **Therapy thinking time** -- a longer end-of-utterance silence while a
  support conversation is in progress, so a student who pauses to find
  words is not cut off mid-thought.
* **Graceful close** -- recognising a clear goodbye.
* **Idle check-in** -- "I'm still here whenever you're ready." once per
  long silence.
"""
from __future__ import annotations

import os
import random
import re

# ───────── spoken repair ─────────

REPAIR_ENABLED = os.environ.get("REPAIR_PROMPT", "1") == "1"
# A few wordings so NAO doesn't sound like a recording. ``REPAIR_LINE``
# (one line) or ``REPAIR_LINES`` ("|"-separated) override them.
_DEFAULT_REPAIR_LINES = (
    "Sorry, I didn't catch that. Could you say it again?",
    "Sorry, I missed that. Could you say it one more time?",
    "I didn't quite hear you. Could you repeat that?",
)
REPAIR_LINES = tuple(
    line.strip()
    for line in (os.environ.get("REPAIR_LINES")
                 or os.environ.get("REPAIR_LINE")
                 or "|".join(_DEFAULT_REPAIR_LINES)).split("|")
    if line.strip()
)
# Kept for callers that want "the" repair line (e.g. the echo guard).
REPAIR_LINE = REPAIR_LINES[0] if REPAIR_LINES else ""
# At most one repair line per this many seconds.
REPAIR_MIN_INTERVAL_S = float(os.environ.get("REPAIR_MIN_INTERVAL_S", "20"))
# A `no_voice` / `silero_no_speech` clip shorter than this is a click or a
# cough, not someone trying to talk -- no repair for those.
REPAIR_MIN_CLIP_MS = int(os.environ.get("REPAIR_MIN_CLIP_MS", "700"))

# Reject reasons that mean "someone may have spoken and we lost it".
# Deliberately excludes the echo reasons (NAO heard itself), `invalid_audio`
# (a transport fault), `mute_command` and `wait_more_audio`.
REPAIR_REASONS = frozenset({
    "no_voice",
    "silero_no_speech",
    "hallucination_or_noise",
    "non_english",
    "empty_transcript",
})
# Of those, the ones where the server heard no words at all. They need the
# clip-length floor: the robot's energy VAD endpoints on any loud sound.
_SILENT_REASONS = frozenset({"no_voice", "silero_no_speech"})


def repair_allowed(*, reason: str | None, clip_ms: float, muted: bool,
                   engaged: bool, armed: bool, last_repair_ms: float,
                   now_ms: float, nao_speaking: bool = False,
                   speech_confirmed: bool | None = None) -> bool:
    """Whether NAO should say the repair line for this rejected turn.

    ``armed`` is False after a repair until the next successful turn, so
    NAO never asks twice in a row. ``last_repair_ms`` enforces the
    minimum interval on top of that. ``nao_speaking`` is True while NAO's
    own reply is playing or a reply is being prepared -- what the mic
    picked up then is most likely NAO itself. ``speech_confirmed`` is the
    streaming VAD's verdict (None when it was unavailable): a transcript
    thrown out as noise when the VAD heard no speech was noise.
    """
    if not REPAIR_ENABLED or not REPAIR_LINES:
        return False
    if muted or not engaged or not armed or nao_speaking:
        return False
    if speech_confirmed is False and reason not in _SILENT_REASONS:
        return False
    if reason not in REPAIR_REASONS:
        return False
    if reason in _SILENT_REASONS and clip_ms < REPAIR_MIN_CLIP_MS:
        return False
    if last_repair_ms and (now_ms - last_repair_ms) < REPAIR_MIN_INTERVAL_S * 1000.0:
        return False
    return True


def pick_repair_line(previous: str | None = None) -> str:
    """Next repair wording, never the same one twice in a row."""
    if not REPAIR_LINES:
        return ""
    choices = [l for l in REPAIR_LINES if l != previous] or list(REPAIR_LINES)
    return random.choice(choices)


# ───────── end-of-utterance timing ─────────

# Silence that ends a turn while a support conversation is in progress.
# The default 500 ms suits quick chat; people talking about something hard
# pause longer and were being cut off mid-sentence. While it applies, the
# robot's own end-of-utterance hint and the semantic "sounds finished"
# shortcut no longer end the turn early (see ``app_ws._should_finalize_turn``).
THERAPY_EOU_SILENCE_MS = int(os.environ.get("THERAPY_EOU_SILENCE_MS", "1400"))


def eou_silence_ms(default_ms: int, support_lane: str | None) -> int:
    """The silence that finalizes a turn, given the active support lane."""
    if support_lane:
        return max(int(default_ms), THERAPY_EOU_SILENCE_MS)
    return int(default_ms)


# ───────── graceful close ─────────

GOODBYE_ENABLED = os.environ.get("GOODBYE_FAST_PATH", "1") == "1"
GOODBYE_REPLY = os.environ.get(
    "GOODBYE_REPLY", "It was really nice talking with you. Take care!")
GOODBYE_SUPPORT_REPLY = os.environ.get(
    "GOODBYE_SUPPORT_REPLY",
    "Thank you for talking with me today. If things get heavy, the "
    "counseling center is there for you, and you can always call or text "
    "988. Take care.")

# Polite words that may lead or trail a goodbye without changing it.
_LEAD = (r"(?:(?:ok(?:ay)?|alright|all right|well|so|yeah|yes|anyway|"
         r"cool|great|thanks|thank you(?: so much| very much)?|nao|now)\s+)*")
_TRAIL = (r"(?:\s+(?:nao|then|for now|for today|now|thanks|thank you|"
          r"everyone|again|buddy|friend|man|bye|goodbye))*")
_CORE = (
    r"(?:"
    r"bye(?: bye)?|good ?bye|bye for now|bye now"
    r"|see you(?: later| soon| next time| tomorrow| around| then)?"
    r"|see ya(?: later)?|catch you later|talk to you (?:later|soon)"
    r"|take care(?: of yourself)?|good ?night"
    r"|have a (?:good|great|nice) (?:day|night|one|evening|weekend)"
    r"|i (?:have|got|need) to go(?: now)?|i gotta go(?: now)?"
    r"|i (?:have|got|need) to get going|i should (?:go|get going)"
    r"|i'?m (?:leaving|heading out|going to go|gonna go|off)(?: now)?"
    r"|i'?ll let you go"
    r"|i'?m done for (?:now|today)"
    r"|that'?s all for (?:now|today)"
    r"|that'?s it for (?:now|today)"
    r")"
)
_GOODBYE_RE = re.compile(r"^" + _LEAD + _CORE + _TRAIL + r"$")
# "That's all, thanks" / "That's all I needed, thank you" -- "that's all"
# alone is too often an answer inside an exercise ("anything else?"
# "that's all") to end a conversation on.
_THATS_ALL_THANKS_RE = re.compile(
    r"^" + _LEAD + r"that'?s all(?: i needed)?\s+(?:thanks|thank you)"
    r"(?: so much| very much)?" + _TRAIL + r"$")


def _normalize(text: str) -> str:
    t = (text or "").lower().replace("’", "'").replace("‘", "'")
    t = t.replace("that is", "that's").replace("i am", "i'm")
    t = t.replace("i will", "i'll").replace("got to", "gotta")
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _matches(clause: str) -> bool:
    c = _normalize(clause)
    if not c:
        return False
    return bool(_GOODBYE_RE.match(c) or _THATS_ALL_THANKS_RE.match(c))


def is_goodbye(transcript: str) -> bool:
    """True when the utterance is, or clearly ends with, a goodbye.

    Matches the whole utterance or its final clause(s) -- "Okay, thank you.
    Bye!" -- but not a goodbye mentioned inside a sentence about something
    else ("I never got to say goodbye to my grandmother").
    """
    raw = (transcript or "").strip()
    if not raw:
        return False
    if _matches(raw):
        return True
    clauses = [c for c in re.split(r"[.!?;,]+", raw) if c.strip()]
    if len(clauses) < 2:
        return False
    # Last clause on its own ("..., bye"), or the last two together
    # ("That's all, thanks").
    return _matches(clauses[-1]) or _matches(" ".join(clauses[-2:]))


def goodbye_reply(support_lane: str | None) -> str:
    return GOODBYE_SUPPORT_REPLY if support_lane else GOODBYE_REPLY


# ───────── idle check-in ─────────

IDLE_CHECKIN_ENABLED = os.environ.get("IDLE_CHECKIN", "1") == "1"
IDLE_CHECKIN_S = float(os.environ.get("IDLE_CHECKIN_S", "45"))
IDLE_CHECKIN_LINE = os.environ.get(
    "IDLE_CHECKIN_LINE", "I'm still here whenever you're ready.")


def idle_checkin_due(*, now_ms: float, last_activity_ms: float, muted: bool,
                     already_done: bool, busy: bool) -> bool:
    """Whether to say the idle check-in now.

    ``last_activity_ms`` is the later of the user's last speech and NAO's
    last spoken line. ``already_done`` latches once per silence period;
    the caller clears it when the user speaks again.
    """
    if not IDLE_CHECKIN_ENABLED or not IDLE_CHECKIN_LINE.strip():
        return False
    if muted or already_done or busy:
        return False
    return (now_ms - last_activity_ms) >= IDLE_CHECKIN_S * 1000.0

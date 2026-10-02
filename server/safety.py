"""Pre-dispatch crisis gate. Runs before any agent sees the user message.

``crisis_check`` runs on every turn, before dispatch:

* Hard phrases return a crisis at once, with no model call.
* Soft phrases (ambiguous on their own) ask the LLM classifier and fail
  safe: a classifier error counts as a crisis.
* Every therapy-lane or emotional turn also asks the classifier, even with
  no keyword hit -- indirect wording ("I feel like everyone would be fine
  without me") matches no list. A classifier error here does NOT count as
  a crisis, or an outage would turn every support reply into the hotline.
* The last few user turns are checked stitched together, so a crisis split
  across turns ("I keep thinking" / "about not waking up") is still seen.

Everything is matched against ``normalize(text)``, so "I don't", "I do
not" and a dropped-apostrophe "I dont" all look the same.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from server import config, llm_compat

_log = logging.getLogger("sage.safety")

# ───────── normalisation ─────────

# Order matters: specific forms before the generic n't / 're / 'm rules.
_CONTRACTIONS = (
    (r"\bcan't\b", "can not"), (r"\bcannot\b", "can not"),
    (r"\bwon't\b", "will not"), (r"\bshan't\b", "shall not"),
    (r"\bain't\b", "am not"),
    (r"\b(\w+)n't\b", r"\1 not"),
    (r"\bi'm\b", "i am"), (r"\b(\w+)'re\b", r"\1 are"),
    (r"\b(\w+)'ve\b", r"\1 have"), (r"\b(\w+)'ll\b", r"\1 will"),
    (r"\bi'd\b", "i would"),
    (r"\bgonna\b", "going to"), (r"\bwanna\b", "want to"),
    (r"\bgotta\b", "got to"),
)

# STT frequently drops the apostrophe. Only tokens that are not real words
# on their own ("ill", "id", "well", "were" are deliberately absent).
_BARE_CONTRACTIONS = {
    "dont": "do not", "cant": "can not", "wont": "will not",
    "didnt": "did not", "doesnt": "does not", "isnt": "is not",
    "wasnt": "was not", "werent": "were not", "arent": "are not",
    "havent": "have not", "hasnt": "has not", "wouldnt": "would not",
    "couldnt": "could not", "shouldnt": "should not", "im": "i am",
    "ive": "i have", "youre": "you are", "theyre": "they are",
}


def normalize(text: str) -> str:
    """Lower-case, expand contractions, strip punctuation, collapse spaces."""
    s = (text or "").lower()
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")
    for pat, rep in _CONTRACTIONS:
        s = re.sub(pat, rep, s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    words = [_BARE_CONTRACTIONS.get(w, w) for w in s.split()]
    return " ".join(words)


# ───────── phrase lists (canonical, normalised form) ─────────

# Hard -- unambiguous first-person intent. Any match is a crisis, no LLM.
# Keep these specific: "this homework is killing me" and "I'm dying to see
# that movie" must NOT land here.
_HARD_KEYWORDS = (
    "kill myself", "killing myself", "kill my self",
    "end my life", "ending my life", "take my own life",
    "suicide", "suicidal",
    "want to die", "going to die tonight",
    "hurt myself", "hurting myself", "cutting myself", "harm myself",
    "unalive myself",
    "end it all",
    "better off without me", "better off dead",
    "wish i was dead", "wish i were dead",
    "wish i was not alive", "wish i were not alive",
    "do not want to be alive", "do not want to live anymore",
    "do not want to live any more",
    "no reason to live", "nothing to live for",
)

# Soft -- concerning, but common in benign speech too. The LLM decides.
_SOFT_TRIGGERS = (
    "do not want to be here", "can not go on", "no point", "tired of living",
    "done with everything", "give up", "hopeless",
    "want to disappear", "wish i could disappear", "wish i was not here",
    "wish i were not here", "a burden", "burden to", "burden on",
    "better off without", "do not want to wake up", "never wake up",
    "can not do this anymore", "can not take it anymore",
    "kms", "unalive", "overdose", "going to jump", "jump off",
    "jump in front", "self harm", "want it all to stop", "sleep forever",
    "say goodbye to everyone", "take my life", "do not want to live",
    # "Everyone would be fine without me" -- specific "<better> without me"
    # forms only; a bare "without me" ("they left without me") is not.
    "fine without me", "happier without me", "easier without me",
    "better without me", "okay without me", "ok without me",
    "would not notice if i", "would not even notice if i",
    "nobody would notice if i", "no one would notice if i",
    "would anyone notice if i", "would anyone even notice if i",
    "notice if i was gone", "notice if i were gone", "miss me if i",
)

# Soft patterns that need more than a substring.
_SOFT_PATTERNS = (
    re.compile(r"\b(saving|save|saved|stockpil\w*|hoard\w*|collect\w*)\b"
               r"(\s+\w+){0,3}\s+(pills|meds|medication|tablets)\b"),
    re.compile(r"\b(take|took|swallow\w*)\s+(all|a bunch)\s+(of\s+)?"
               r"(my\s+)?(pills|meds|tablets)\b"),
)

HOTLINE_REPLY = (
    "What you're carrying sounds really heavy, and you don't have to "
    "hold it alone. Please reach out to someone who can stay with you right now — "
    "you can call or text 988 in the US for the Suicide and Crisis Lifeline, any "
    "time, day or night. Is there someone nearby you can be with too?"
)


def hotline_reply() -> str:
    """The spoken crisis reply: 988, plus the campus counseling line if set."""
    extra = (getattr(config, "MORGAN_COUNSELING_TEXT", "") or "").strip()
    if not extra:
        return HOTLINE_REPLY
    # Keep it short: the counseling line goes before the closing question.
    head, sep, tail = HOTLINE_REPLY.rpartition(" Is there someone")
    if not sep:
        return HOTLINE_REPLY + " " + extra
    return head + " " + extra + sep + tail


# Words that put a turn on the emotional path, where the classifier runs
# even with no crisis keyword. Normalised form, whole-word/phrase matches.
_EMOTIONAL_WORDS = (
    "sad", "depressed", "depression", "anxious", "anxiety", "panic",
    "lonely", "alone", "overwhelmed", "stressed", "stress", "worried",
    "scared", "afraid", "hurt", "hurting", "pain", "cry", "crying",
    "grief", "grieving", "empty", "numb", "worthless", "useless",
    "hate myself", "hate my life", "miserable", "exhausted", "tired of",
    "nobody cares", "no one cares", "no one would", "nobody would",
    "therapy", "therapist", "upset", "broken", "trapped", "failure",
    "can not cope", "falling apart", "breaking down", "lost",
)


@dataclass(frozen=True)
class CrisisResult:
    positive: bool
    # "keyword" | "llm" | "failsafe" | "clean" | "clean_llm_error"
    source: str
    # True when the hit came from the stitched recent-turns transcript
    # rather than this turn alone.
    stitched: bool = False


def _contains(norm: str, phrase: str) -> bool:
    return f" {phrase} " in f" {norm} "


def hard_match(text: str) -> bool:
    norm = normalize(text)
    return any(_contains(norm, k) for k in _HARD_KEYWORDS)


def soft_match(text: str) -> bool:
    norm = normalize(text)
    return (any(_contains(norm, t) for t in _SOFT_TRIGGERS)
            or any(p.search(norm) for p in _SOFT_PATTERNS))


def is_emotional(text: str) -> bool:
    """True when the turn carries emotional/support language."""
    norm = normalize(text)
    return (any(_contains(norm, w) for w in _EMOTIONAL_WORDS)
            or hard_match(text) or soft_match(text))


# Knobs read at call time so tests can flip them.
def _always_on() -> bool:
    return os.environ.get("CRISIS_ALWAYS_ON", "1") != "0"


def _min_words() -> int:
    try:
        return int(os.environ.get("CRISIS_BG_MIN_WORDS", "3"))
    except ValueError:
        return 3


# How many previous user turns are stitched onto this one.
STITCH_TURNS = int(os.environ.get("CRISIS_STITCH_TURNS", "3") or 3)


def stitch(text: str, recent=()) -> str:
    """This turn plus the last ``STITCH_TURNS`` user turns, oldest first."""
    prior = [t for t in list(recent or ())[-STITCH_TURNS:] if t]
    return " ".join(prior + [text or ""]).strip()


def crisis_check(text: str, *, recent=(), therapy_lane: bool = False
                 ) -> CrisisResult:
    """Decide whether this turn gets the hotline reply.

    ``recent`` is the user's previous turns this conversation (oldest
    first); ``therapy_lane`` is True while a support agent owns the
    conversation. Both are optional, so ``crisis_check(text)`` still works.
    """
    joined = stitch(text, recent)
    multi = joined != (text or "").strip()

    if hard_match(text):
        return CrisisResult(True, "keyword")
    if multi and hard_match(joined):
        return CrisisResult(True, "keyword", stitched=True)

    soft_now = soft_match(text)
    if soft_now or (multi and soft_match(joined)):
        try:
            if _llm_classify(joined):
                return CrisisResult(True, "llm", stitched=not soft_now)
            return CrisisResult(False, "llm")
        except Exception as exc:  # noqa: BLE001
            if soft_now:
                return CrisisResult(True, "failsafe")
            # Only the stitched text matched: an earlier turn already went
            # through the gate on its own, so during an outage this would
            # answer ordinary follow-ups with the hotline. Fail open.
            _log.warning("crisis_stitched_classifier_failed: %r", exc)
            return CrisisResult(False, "clean_llm_error")

    if (_always_on()
            and len(normalize(text).split()) >= _min_words()
            and (therapy_lane or is_emotional(text)
                 or (multi and is_emotional(joined)))):
        try:
            if _llm_classify(joined):
                return CrisisResult(True, "llm", stitched=multi)
            return CrisisResult(False, "llm")
        except Exception as exc:  # noqa: BLE001
            # Fail open here: the keyword layers above are the fail-safe.
            _log.warning("crisis_always_on_classifier_failed: %r", exc)
            return CrisisResult(False, "clean_llm_error")

    return CrisisResult(False, "clean")


def _llm_classify(text: str) -> bool:
    # Routed through llm_compat so CRISIS_MODEL can name either provider.
    # Any exception propagates to crisis_check(), which decides whether it
    # counts as a crisis -- do not add a try/except here that swallows it.
    out = llm_compat.chat(
        model=config.CRISIS_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a suicide/self-harm risk classifier. Reply with only "
                    "the word YES or NO. YES if the user is expressing active "
                    "suicidal ideation, intent to self-harm, or immediate crisis, "
                    "including indirect wording (wanting to disappear, feeling "
                    "like a burden, not wanting to wake up, saving up pills, "
                    "slang like kms or unalive). NO for general stress, sadness, "
                    "venting, figures of speech ('this homework is killing me', "
                    "'dying to see it'), or hypothetical talk. The text may be "
                    "several of the user's recent turns joined together; judge "
                    "them as one message."
                ),
            },
            {"role": "user", "content": text},
        ],
        max_tokens=4,
        temperature=0,
    )
    return out.strip().upper().startswith("Y")


# ───────── after a crisis hit ─────────

def record_crisis_event(owner: str, level: str,
                        turn_index: int | None = None) -> None:
    """Write one ``safety_events`` row: who (the therapy owner key), how it
    was caught, and when. Deliberately no message text. Never raises -- a
    DB problem must not stop the hotline reply."""
    try:
        from server import session as _session
        with _session._conn() as c:
            c.execute(
                "INSERT INTO safety_events "
                "(username, turn_index, clause, severity, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (owner or None, turn_index, "crisis_gate", level, ""),
            )
    except Exception as exc:  # noqa: BLE001
        _log.warning("crisis_event_write_failed: %r", exc)


def send_crisis_alert(level: str, *, ts: float | None = None):
    """POST ``{"level", "ts"}`` to ``CRISIS_ALERT_WEBHOOK_URL``, if set.

    Fire-and-forget on a daemon thread: no transcript, no name, and a slow
    or dead webhook never delays the reply. Returns the thread (or None
    when the webhook is off) so tests can join it.
    """
    url = (getattr(config, "CRISIS_ALERT_WEBHOOK_URL", "") or "").strip()
    if not url:
        return None
    import threading
    import time as _time

    payload = {"level": level, "ts": _time.time() if ts is None else ts}

    def _post() -> None:
        try:
            import httpx
            httpx.post(url, json=payload, timeout=5)
        except Exception as exc:  # noqa: BLE001
            _log.warning("crisis_alert_webhook_failed: %r", exc)

    t = threading.Thread(target=_post, daemon=True, name="crisis-alert")
    t.start()
    return t


CRISIS_FOLLOWUP_NOTE = (
    "[CRISIS_FOLLOWUP]\n"
    "(Your previous reply to this person was the crisis safety message with "
    "the 988 line. Before anything else, gently and briefly check in on how "
    "they are doing right now. Stay warm and calm; do not lecture or repeat "
    "the whole hotline message unless they ask, but if they still sound "
    "unsafe, remind them they can call or text 988.)\n"
)


def mark_crisis(conv: dict | None, level: str) -> None:
    """Flag the conversation so the next turn checks in (see
    ``apply_crisis_followup``), and drop the stitched-turn buffer so the
    same words do not re-trigger the gate on every following turn."""
    if not isinstance(conv, dict):
        return
    conv["crisis_followup"] = {"level": level}
    conv.pop("recent_user_turns", None)


def remember_turn(conv: dict | None, text: str) -> None:
    """Keep the last few user turns for the stitched crisis check."""
    if not isinstance(conv, dict) or not (text or "").strip():
        return
    turns = list(conv.get("recent_user_turns") or [])
    turns.append(text.strip()[:500])
    conv["recent_user_turns"] = turns[-STITCH_TURNS:]


def apply_crisis_followup(conv: dict | None, message):
    """Prepend the check-in note to a Runner input once, then clear the flag.

    ``message`` is a plain string or the Responses-API list shape built by
    ``_legacy_helpers._build_user_message``.
    """
    if not isinstance(conv, dict) or not conv.pop("crisis_followup", None):
        return message
    if isinstance(message, str):
        return CRISIS_FOLLOWUP_NOTE + message
    try:
        for item in message:
            for part in item.get("content") or []:
                if part.get("type") == "input_text":
                    part["text"] = CRISIS_FOLLOWUP_NOTE + (part.get("text") or "")
                    return message
    except Exception:  # noqa: BLE001
        pass
    return message

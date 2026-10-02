"""Privacy controls: "forget me", opt-in data retention, log redaction.

Three independent pieces, kept together because they answer the same
question -- what does NAO keep about a student, and for how long:

* ``forget_user_data`` -- delete everything stored under one person: chat
  history, moods, thought records, homework, recaps, themes, profile.
  Reached by voice ("forget me", then "yes" to confirm; see
  ``detect_forget_request`` / ``handle_forget_turn``) and by the
  ``forget_me`` agent tool (``server/tools/privacy_tools.py``).
* ``prune_old_data`` -- with ``DATA_RETENTION_DAYS`` > 0, delete rows older
  than that at startup. Off by default: trimming a therapy record is a
  clinical call, so nothing is deleted unless someone opts in.
* ``redact_processor`` -- a structlog processor that blanks utterance and
  reply text on therapist / crisis / emotional turns, so a student's
  disclosures never land in ``logs/server.log`` or journald.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any

from server import config, session

_log = logging.getLogger("sage.privacy")

# Therapy tables written under the owner key (session.therapy_owner).
# ``homework`` belongs to another feature and may not exist yet; missing
# tables are skipped.
THERAPY_TABLES = (
    "mood_log", "thought_records", "recaps", "weekly_themes",
    "monthly_personas", "homework", "topology_trace",
)
_OWNER_COLUMNS = ("username", "owner", "face_id", "user")

# SDK SQLiteSession default table names (agents/memory/sqlite_session.py).
_SDK_SESSIONS = "agent_sessions"
_SDK_MESSAGES = "agent_messages"


def _db_path() -> str:
    return getattr(session, "_DB_PATH", None) or config.SESSION_DB


@contextmanager
def _connect():
    c = sqlite3.connect(_db_path())
    try:
        yield c
        c.commit()
    finally:
        c.close()


def _table_columns(c: sqlite3.Connection, table: str) -> list[str]:
    try:
        return [r[1] for r in c.execute(f"PRAGMA table_info({table})")]
    except sqlite3.Error:
        return []


def _owner_column(c: sqlite3.Connection, table: str) -> str | None:
    cols = _table_columns(c, table)
    return next((col for col in _OWNER_COLUMNS if col in cols), None)


# ───────── forget me ─────────

def _owner_keys(username: str) -> list[str]:
    """Every key this person's rows may be stored under."""
    keys = {session.therapy_owner(username)}
    if not session.is_anonymous(username):
        keys.add(username.strip())
    return [k for k in keys if k]


def _history_keys(username: str) -> list[str]:
    if session.is_anonymous(username):
        live = session.live_anonymous_key()
        return [live] if live else []
    return [session.session_key_for(username)]


def forget_user_data(username: str) -> dict[str, int]:
    """Delete everything stored about ``username``. Returns rows deleted
    per table (for the log; no content). Never raises.

    ``safety_events`` rows are kept but de-identified (owner set to NULL):
    the count and time of crisis hits stay useful for safety review, and no
    longer point at anyone.
    """
    counts: dict[str, int] = {}
    owners = [k.lower() for k in _owner_keys(username)]
    history = _history_keys(username)
    try:
        with _connect() as c:
            for table in THERAPY_TABLES + ("user_prefs",):
                col = _owner_column(c, table)
                if col is None:
                    continue
                n = 0
                for key in owners:
                    n += c.execute(
                        f"DELETE FROM {table} WHERE lower({col}) = ?",
                        (key,)).rowcount
                counts[table] = n
            if _owner_column(c, "safety_events"):
                n = 0
                for key in owners:
                    n += c.execute(
                        "UPDATE safety_events SET username = NULL "
                        "WHERE lower(username) = ?", (key,)).rowcount
                counts["safety_events_deidentified"] = n
            # memory.py: profile + per-visit summaries (face_id lowercased).
            if _table_columns(c, "sessions"):
                counts["sessions"] = sum(c.execute(
                    "DELETE FROM sessions WHERE lower(face_id) = ?",
                    (k,)).rowcount for k in owners)
            if _table_columns(c, "users"):
                counts["users"] = sum(c.execute(
                    "DELETE FROM users WHERE lower(face_id) = ?",
                    (k,)).rowcount for k in owners)
            # Agents SDK chat history.
            if _table_columns(c, _SDK_MESSAGES):
                counts["chat_messages"] = sum(c.execute(
                    f"DELETE FROM {_SDK_MESSAGES} WHERE session_id = ?",
                    (k,)).rowcount for k in history)
            if _table_columns(c, _SDK_SESSIONS):
                for k in history:
                    c.execute(
                        f"DELETE FROM {_SDK_SESSIONS} WHERE session_id = ?",
                        (k,))
    except Exception as exc:  # noqa: BLE001
        _log.warning("forget_user_data_failed: %r", exc)
        counts["error"] = 1
    # In-memory working state (CBT step, crisis flag, recent turns).
    try:
        from server import conversation_state
        conversation_state.clear(username)
    except Exception:  # noqa: BLE001
        pass
    return counts


# Voice path. Explicit phrases only, and only in a short utterance, so
# "don't forget me" or a story that mentions deleting data never fires.
_FORGET_RE = re.compile(
    r"\b(forget (about )?me|forget everything (you know )?about me"
    r"|forget who i am"
    r"|(delete|erase|wipe|remove) (all )?(of )?my (data|information|info"
    r"|history|records|conversations?|memories))\b")
_NEGATED_RE = re.compile(r"\b(do not|dont|don t|never|not|won t|wont)\s+"
                         r"(ever\s+)?(forget|delete|erase|wipe|remove)\b")
_YES_RE = re.compile(
    r"^(yes|yeah|yep|yup|sure|ok|okay|confirm|do it|please do|go ahead"
    r"|yes please|i am sure|i m sure)\b")
FORGET_CONFIRM_WINDOW_S = float(os.environ.get("FORGET_CONFIRM_WINDOW_S", "60"))

FORGET_CONFIRM_PROMPT = (
    "Do you want me to delete everything I have saved about you, including "
    "our conversations? Say yes to confirm."
)
FORGET_DONE_REPLY = "Done. I've deleted everything I had saved about you."
FORGET_CANCELLED_REPLY = "Okay, I won't delete anything."


def _plain(text: str) -> str:
    s = (text or "").lower().replace("’", "'")
    s = re.sub(r"[^a-z' ]", " ", s).replace("'", " ")
    return " ".join(s.split())


def detect_forget_request(text: str) -> bool:
    t = _plain(text)
    if not t or len(t.split()) > 12:
        return False
    if _NEGATED_RE.search(t):
        return False
    return bool(_FORGET_RE.search(t))


def handle_forget_turn(conv: dict, text: str, *,
                       now: float | None = None) -> str | None:
    """Two-step voice flow. Returns the action for this turn:

    * ``"ask"``     -- a new request; speak ``FORGET_CONFIRM_PROMPT``.
    * ``"confirm"`` -- the user said yes in time; delete, then speak done.
    * ``"cancel"``  -- a pending request answered with anything else.
    * ``None``      -- not a forget turn; dispatch normally.
    """
    stamp = time.time() if now is None else now
    pending = conv.pop("forget_pending", None) if isinstance(conv, dict) else None
    if pending is not None and stamp - float(pending) <= FORGET_CONFIRM_WINDOW_S:
        return "confirm" if _YES_RE.search(_plain(text)) else "cancel"
    if detect_forget_request(text):
        if isinstance(conv, dict):
            conv["forget_pending"] = stamp
        return "ask"
    return None


# ───────── retention ─────────

# (table, timestamp column, kind) -- "text" columns hold SQLite
# CURRENT_TIMESTAMP strings (UTC), "epoch" columns hold time.time().
_RETENTION_TARGETS = (
    ("mood_log", "created_at", "text"),
    ("thought_records", "created_at", "text"),
    ("recaps", "created_at", "text"),
    ("weekly_themes", "created_at", "text"),
    ("monthly_personas", "created_at", "text"),
    ("homework", "created_at", "text"),
    ("topology_trace", "created_at", "text"),
    ("safety_events", "created_at", "text"),
    (_SDK_MESSAGES, "created_at", "text"),
    ("sessions", "started_at", "epoch"),
)


def prune_old_data(days: int | None = None, *,
                   now: float | None = None) -> dict[str, int]:
    """Delete rows older than ``days`` (default ``DATA_RETENTION_DAYS``).

    ``days`` <= 0 does nothing -- retention is opt-in. Missing tables and
    columns are skipped. Never raises.
    """
    days = config.DATA_RETENTION_DAYS if days is None else int(days)
    if not days or days <= 0:
        return {}
    cutoff = (time.time() if now is None else now) - days * 86400
    cutoff_text = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(cutoff))
    counts: dict[str, int] = {}
    try:
        with _connect() as c:
            for table, col, kind in _RETENTION_TARGETS:
                if col not in _table_columns(c, table):
                    continue
                arg = cutoff if kind == "epoch" else cutoff_text
                counts[table] = c.execute(
                    f"DELETE FROM {table} WHERE {col} < ?", (arg,)).rowcount
            if _table_columns(c, _SDK_SESSIONS) and _table_columns(c, _SDK_MESSAGES):
                counts[_SDK_SESSIONS] = c.execute(
                    f"DELETE FROM {_SDK_SESSIONS} WHERE session_id NOT IN "
                    f"(SELECT DISTINCT session_id FROM {_SDK_MESSAGES}) "
                    f"AND updated_at < ?", (cutoff_text,)).rowcount
    except Exception as exc:  # noqa: BLE001
        _log.warning("prune_old_data_failed: %r", exc)
    return counts


# ───────── log redaction ─────────

SUPPORT_AGENTS = frozenset({
    "therapist", "cbt_coach", "grounding_coach", "mi_coach",
})
# Fields that can carry what the student said or what NAO said back.
TEXT_FIELDS = (
    "transcript", "reply_preview", "reply", "question", "text", "partial",
    "utterance", "user_text", "final_reply", "proposed_reply", "stitched",
)
REDACTED = "[redacted]"

# A user stays private for this long after a support/crisis turn, so the
# STT log line of their next turn (written before routing) is covered too.
PRIVATE_TTL_S = float(os.environ.get("LOG_PRIVATE_TTL_S", "900"))
_private_lock = threading.Lock()
_private_until: dict[str, float] = {}


def mark_private(user: str | None, *, now: float | None = None) -> None:
    if not user:
        return
    stamp = time.time() if now is None else now
    with _private_lock:
        _private_until[str(user).lower()] = stamp + PRIVATE_TTL_S


def is_private_user(user: str | None, *, now: float | None = None) -> bool:
    if not user:
        return False
    stamp = time.time() if now is None else now
    with _private_lock:
        until = _private_until.get(str(user).lower())
        if until is None:
            return False
        if stamp > until:
            _private_until.pop(str(user).lower(), None)
            return False
        return True


def _event_is_private(ev: dict[str, Any]) -> bool:
    agent = str(ev.get("active_agent") or ev.get("agent") or "")
    outcome = str(ev.get("outcome") or "")
    name = str(ev.get("event") or "")
    if ev.get("private") or agent in SUPPORT_AGENTS:
        return True
    if "crisis" in agent or "crisis" in outcome or "crisis" in name:
        return True
    user = ev.get("user")
    if is_private_user(user):
        return True
    text = ev.get("transcript")
    if isinstance(text, str) and text:
        try:
            from server import safety
            if safety.is_emotional(text):
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


def redact_processor(_logger: Any, _name: str,
                     ev: dict[str, Any]) -> dict[str, Any]:
    """structlog processor: blank utterance/reply text on private turns.

    Runs first in the chain, so the dashboard capture and the renderer
    both see the redacted event. Never raises.
    """
    try:
        if os.environ.get("LOG_REDACT", "1") == "0":
            return ev
        if not _event_is_private(ev):
            return ev
        agent = str(ev.get("active_agent") or "")
        if (agent in SUPPORT_AGENTS or "crisis" in str(ev.get("event") or "")
                or "crisis" in str(ev.get("outcome") or "")):
            mark_private(ev.get("user"))
        for field in TEXT_FIELDS:
            val = ev.get(field)
            if isinstance(val, str) and val:
                ev[field] = REDACTED
        ev["redacted"] = True
    except Exception:  # noqa: BLE001
        pass
    return ev


def install_log_redaction() -> None:
    """Put :func:`redact_processor` at the front of the active structlog
    chain. Idempotent; leaves the format unchanged."""
    try:
        import structlog
        procs = list(structlog.get_config().get("processors") or [])
        if any(getattr(p, "__name__", "") == "redact_processor" for p in procs):
            return
        structlog.configure(processors=[redact_processor] + procs)
    except Exception:  # noqa: BLE001 -- never break logging over this
        pass


def _reset_for_tests() -> None:
    with _private_lock:
        _private_until.clear()

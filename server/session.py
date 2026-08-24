"""Session persistence: Agents SDK SQLiteSession + per-user prefs/recaps.

SQLiteSession handles the chat history. We add a tiny side-table for camera
consent and a recaps table for therapist cross-session memory.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from agents import SQLiteSession

from server import config

_DB_PATH = config.SESSION_DB


@contextmanager
def _conn():
    c = sqlite3.connect(_DB_PATH)
    c.execute(
        "CREATE TABLE IF NOT EXISTS user_prefs ("
        "username TEXT PRIMARY KEY, camera_consent INTEGER NOT NULL DEFAULT 1)"
    )
    try:
        c.execute("ALTER TABLE user_prefs ADD COLUMN proactive_enabled INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # column already exists
    try:
        c.execute(
            "ALTER TABLE user_prefs ADD COLUMN voice_profile TEXT NOT NULL "
            "DEFAULT ''"
        )
    except sqlite3.OperationalError:
        pass  # column already exists
    c.execute(
        "CREATE TABLE IF NOT EXISTS recaps ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "username TEXT NOT NULL, body TEXT NOT NULL, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    c.execute(
        "CREATE TABLE IF NOT EXISTS weekly_themes ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, "
        "week_start DATE NOT NULL, body TEXT NOT NULL, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
        "UNIQUE(username, week_start))"
    )
    c.execute(
        "CREATE TABLE IF NOT EXISTS monthly_personas ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, "
        "month DATE NOT NULL, body TEXT NOT NULL, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
        "UNIQUE(username, month))"
    )
    # SAGE-CBT: invariant violation log (RQ2).
    c.execute(
        "CREATE TABLE IF NOT EXISTS safety_events ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "username TEXT, turn_index INTEGER, clause TEXT, severity TEXT, "
        "payload TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    # SAGE-CBT: one row per topology turn (for Pareto / post-hoc analysis).
    c.execute(
        "CREATE TABLE IF NOT EXISTS topology_trace ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "username TEXT, topology TEXT, user_text TEXT, "
        "proposed_reply TEXT, final_reply TEXT, verdict TEXT, affect TEXT, "
        "invariant_holds INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    # Per-turn mood log surfaced in next-session greeting.
    # Written by server/tools/emotion.py:log_emotion. Read by
    # server/memory.py:build_context_preamble for the "Recent mood:" line.
    c.execute(
        "CREATE TABLE IF NOT EXISTS mood_log ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, "
        "mood TEXT NOT NULL, intensity INTEGER NOT NULL, "
        "trigger TEXT NOT NULL, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_mood_log_user_ts "
        "ON mood_log(username, created_at DESC)"
    )
    # CBT thought records persisted across sessions so the therapist can
    # refer back ("last time we worked on catastrophizing").
    # Written by server/tools/emotion.py:identify_distortion + suggest_reframe.
    c.execute(
        "CREATE TABLE IF NOT EXISTS thought_records ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, "
        "thought TEXT NOT NULL, distortion TEXT NOT NULL, "
        "reframe TEXT NOT NULL DEFAULT '', "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_thought_records_user_ts "
        "ON thought_records(username, created_at DESC)"
    )
    try:
        yield c
        c.commit()
    finally:
        c.close()


# ---------------------------------------------------------------------------
# Anonymous session scoping
# ---------------------------------------------------------------------------
#
# Chat history is keyed by username. That is correct for a recognised student —
# therapy continuity across visits is the point — but everyone who is *not*
# face-recognised is `guest`, so a single `user:guest` row silently became a
# shared transcript of every anonymous conversation ever held (709 messages,
# 2026-05-11 -> 2026-08-24 on the Pi, with a name from Jul 30 being quoted back
# to a different person on Aug 24). On a CBT robot that leaks one student's
# disclosures into the next student's context window.
#
# Anonymous conversations therefore get an *idle-bounded epoch* instead: the
# same key while the lane stays active, a fresh one once it has been quiet for
# GUEST_IDLE_RESET_S. We key on idle rather than on the per-WebSocket
# `session_id` because WS connections drop every few seconds during long TTS
# playbacks (see the first-turn tracker below) — a per-connection key would
# reset history mid-sentence. The epoch lives in process memory, so a server
# restart also starts a fresh anonymous conversation, which fails closed.
_ANONYMOUS_USERNAMES = frozenset({"", "guest", "unknown"})

# 15 min: far longer than any reconnect storm, far shorter than the gap between
# two people using the robot. Override per-deployment via the environment.
GUEST_IDLE_RESET_S = float(os.environ.get("GUEST_IDLE_RESET_S", str(15 * 60)))

_GUEST_EPOCH_TOKEN: str | None = None
_GUEST_EPOCH_LAST_SEEN: float = 0.0


def is_anonymous(username: str) -> bool:
    """True when ``username`` names nobody in particular."""
    return (username or "").strip().lower() in _ANONYMOUS_USERNAMES


def session_key_for(username: str, *, now: float | None = None) -> str:
    """Return the SQLiteSession key that ``username``'s history belongs in.

    Named users get a stable ``user:<name>`` key so their history persists.
    Anonymous users get ``guest:<epoch>``, shared only for the duration of one
    conversation. ``now`` is injectable for tests.
    """
    global _GUEST_EPOCH_TOKEN, _GUEST_EPOCH_LAST_SEEN

    if not is_anonymous(username):
        return "user:{}".format(username.strip().lower())

    stamp = time.time() if now is None else now
    expired = (stamp - _GUEST_EPOCH_LAST_SEEN) > GUEST_IDLE_RESET_S
    if _GUEST_EPOCH_TOKEN is None or expired:
        _GUEST_EPOCH_TOKEN = uuid.uuid4().hex[:12]
    _GUEST_EPOCH_LAST_SEEN = stamp
    return "guest:{}".format(_GUEST_EPOCH_TOKEN)


def live_anonymous_key() -> str | None:
    """The anonymous epoch key currently in flight, without minting one.

    Returns ``None`` when no anonymous conversation is open. Distinct from
    ``session_key_for`` because callers that are *reading* the epoch (the face
    reco handoff) must not create one as a side effect — doing so would migrate
    a freshly-minted empty row and silently drop what the user just said.
    """
    return None if _GUEST_EPOCH_TOKEN is None else f"guest:{_GUEST_EPOCH_TOKEN}"


def retire_anonymous_epoch() -> None:
    """Close the current anonymous conversation.

    Called once its history has been handed to a named user: the row now
    belongs to that user, so the next stranger must start somewhere else.
    """
    global _GUEST_EPOCH_TOKEN, _GUEST_EPOCH_LAST_SEEN
    _GUEST_EPOCH_TOKEN = None
    _GUEST_EPOCH_LAST_SEEN = 0.0


def get_or_create_session(username: str) -> SQLiteSession:
    return SQLiteSession(session_key_for(username), db_path=_DB_PATH)


def migrate_username(old: str, new: str) -> None:
    """Rename session rows so 'guest' history follows a user after face reco.

    Uses the SDK's public API (add_items / clear_session) rather than raw SQL
    so we stay compatible with any future SDK table-name changes and avoid
    conflicting with the SDK's own file-level locking.
    """
    old_is_anon = is_anonymous(old)
    old_key = live_anonymous_key() if old_is_anon else session_key_for(old)

    if old_key is not None:
        old_sess = SQLiteSession(session_id=old_key, db_path=_DB_PATH)
        new_sess = SQLiteSession(session_key_for(new), db_path=_DB_PATH)

        items = asyncio.run(old_sess.get_items())
        if items:
            asyncio.run(new_sess.add_items(items))
        asyncio.run(old_sess.clear_session())

    if old_is_anon:
        retire_anonymous_epoch()

    # Also migrate prefs rows if they exist
    with _conn() as c:
        c.execute(
            "UPDATE user_prefs SET username = ? WHERE username = ?", (new, old)
        )


def get_camera_consent(username: str) -> bool:
    """Return camera consent flag for ``username``. Default ON (Phase 6).

    First-time callers get a row inserted with ``camera_consent=1`` so the
    persisted state matches the in-memory return value. The default is
    intentionally ON: consent management for Phase 6 lives in the operator
    policy + the audible "stop watching me" trigger + the green-LED capture
    cue, not in a silent default-off knob.
    """
    with _conn() as c:
        row = c.execute(
            "SELECT camera_consent FROM user_prefs WHERE username = ?", (username,)
        ).fetchone()
        if row is None:
            c.execute(
                "INSERT INTO user_prefs (username, camera_consent) VALUES (?, 1)",
                (username,),
            )
            return True
        return bool(row[0])


# ---------------------------------------------------------------------------
# First-turn tracker (Phase 6 camera-consent first-turn announcement).
# ---------------------------------------------------------------------------
#
# `app_ws.py` plays a one-time spoken heads-up the first time a session sees
# audio, telling the user the camera is on and how to disable it. We track
# "have we already announced for this session_id?" in process memory rather
# than a DB column because session_ids are ephemeral per-WebSocket UUIDs:
# they don't survive a server restart, and persisting them would just leak
# rows. The dict is pruned on session close (see ``forget_session``).

import time as _time

_FIRST_TURN_ANNOUNCED: set[str] = set()
# Keyed by username -> last announce timestamp (seconds, monotonic).
# Used to suppress re-announcing the camera heads-up on every WS reconnect
# of the same user. WS connections drop every few seconds during long TTS
# playbacks (5+ s elapsed_s blocks the recv loop on the robot), and each
# reconnect mints a fresh session_id — without this we'd re-greet the
# user "Heads up — my camera is on..." every 10 s.
_LAST_ANNOUNCE_BY_USER: dict[str, float] = {}
_ANNOUNCE_COOLDOWN_S = 300.0  # 5 minutes — long enough to silence reconnects


def is_first_turn(session_id: str, username: str = "") -> bool:
    """True iff the camera-consent heads-up should fire for this engagement.

    Two conditions must both hold:
      1. We haven't fired for this exact session_id yet.
      2. We haven't fired for this username in the last
         ``_ANNOUNCE_COOLDOWN_S`` seconds (suppresses reconnect storms).

    A falsy ``session_id`` is treated as "not first turn" so callers can't
    accidentally fire the announce on an unbound session.
    """
    if not session_id:
        return False
    if session_id in _FIRST_TURN_ANNOUNCED:
        return False
    if username:
        last = _LAST_ANNOUNCE_BY_USER.get(username, 0.0)
        if _time.time() - last < _ANNOUNCE_COOLDOWN_S:
            return False
    return True


def mark_first_turn_announced(session_id: str, username: str = "") -> None:
    """Record that the first-turn camera-consent heads-up has played for
    ``session_id`` and stamp the username's last-announce time.
    Idempotent on repeated calls.
    """
    if not session_id:
        return
    _FIRST_TURN_ANNOUNCED.add(session_id)
    if username:
        _LAST_ANNOUNCE_BY_USER[username] = _time.time()


def forget_session(session_id: str) -> None:
    """Drop ``session_id`` from the first-turn tracker. Called on
    session_close so the in-memory set doesn't leak across long-lived
    server uptime. Idempotent.
    """
    if not session_id:
        return
    _FIRST_TURN_ANNOUNCED.discard(session_id)


def set_camera_consent(username: str, enabled: bool) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO user_prefs (username, camera_consent) VALUES (?, ?) "
            "ON CONFLICT(username) DO UPDATE SET camera_consent=excluded.camera_consent",
            (username, 1 if enabled else 0),
        )


# Phase 11.8: per-user TTS voice profile. Three slots — "girl", "man",
# "neutral". Empty string means "no preference set; use server default".
def get_voice_profile(username: str) -> str:
    """Return the user's chosen voice profile, or "" if not yet set."""
    with _conn() as c:
        row = c.execute(
            "SELECT voice_profile FROM user_prefs WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None:
            return ""
        return (row[0] or "").strip()


def set_voice_profile(username: str, profile: str) -> None:
    """Persist the user's voice profile pick. Empty string clears it."""
    norm = (profile or "").strip().lower()
    with _conn() as c:
        c.execute(
            "INSERT INTO user_prefs (username, voice_profile) VALUES (?, ?) "
            "ON CONFLICT(username) DO UPDATE SET "
            "voice_profile=excluded.voice_profile",
            (username, norm),
        )


def get_proactive_enabled(username: str) -> bool:
    with _conn() as c:
        row = c.execute("SELECT proactive_enabled FROM user_prefs WHERE username = ?", (username,)).fetchone()
        if row is None:
            c.execute("INSERT INTO user_prefs (username, camera_consent, proactive_enabled) VALUES (?, 1, 0)", (username,))
            return False
        return bool(row[0])


def set_proactive_enabled(username: str, enabled: bool) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO user_prefs (username, camera_consent, proactive_enabled) VALUES (?, 1, ?) "
            "ON CONFLICT(username) DO UPDATE SET proactive_enabled=excluded.proactive_enabled",
            (username, 1 if enabled else 0),
        )


def save_recap(username: str, body: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO recaps (username, body) VALUES (?, ?)", (username, body)
        )


def load_recent_recaps(username: str, n: int = 3) -> list[str]:
    with _conn() as c:
        rows = c.execute(
            "SELECT body FROM recaps WHERE username = ? ORDER BY id DESC LIMIT ?",
            (username, n),
        ).fetchall()
        return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Mood log + thought records persistence (ported from nao-therapy).
# ---------------------------------------------------------------------------

def _norm_user(username: str) -> str:
    """Normalize username for mood/thought lookups so writes from any
    code path (mixed-case username from session, lowercased face_id from
    memory preamble) all match the same rows.
    """
    return (username or "").strip().lower()


def log_mood(username: str, mood: str, intensity: int, trigger: str) -> None:
    """Append a mood entry. Caller is the `log_emotion` agent tool."""
    u = _norm_user(username)
    if not u:
        return
    with _conn() as c:
        c.execute(
            "INSERT INTO mood_log (username, mood, intensity, trigger) "
            "VALUES (?, ?, ?, ?)",
            (u, str(mood)[:32], int(intensity), str(trigger)[:200]),
        )


def load_recent_moods(username: str, n: int = 5) -> list[dict]:
    """Most recent mood entries, newest first."""
    u = _norm_user(username)
    if not u:
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT mood, intensity, trigger, created_at FROM mood_log "
            "WHERE username = ? ORDER BY id DESC LIMIT ?",
            (u, n),
        ).fetchall()
        return [
            {"mood": r[0], "intensity": r[1],
             "trigger": r[2], "created_at": r[3]}
            for r in rows
        ]


def log_thought_record(username: str, thought: str, distortion: str,
                        reframe: str = "") -> None:
    """Append a CBT thought record."""
    u = _norm_user(username)
    if not u:
        return
    with _conn() as c:
        c.execute(
            "INSERT INTO thought_records (username, thought, distortion, reframe) "
            "VALUES (?, ?, ?, ?)",
            (u, str(thought)[:500], str(distortion)[:64], str(reframe)[:500]),
        )


def attach_reframe_to_latest_thought(username: str, thought: str,
                                       reframe: str) -> None:
    """Attach a reframe to the most-recent matching thought row."""
    u = _norm_user(username)
    if not u or not reframe:
        return
    with _conn() as c:
        row = c.execute(
            "SELECT id FROM thought_records WHERE username = ? "
            "AND (thought = ? OR thought LIKE ?) "
            "ORDER BY id DESC LIMIT 1",
            (u, thought, f"%{thought[:60]}%"),
        ).fetchone()
        if row is not None:
            c.execute(
                "UPDATE thought_records SET reframe = ? WHERE id = ?",
                (str(reframe)[:500], row[0]),
            )


def load_recent_thought_records(username: str, n: int = 3) -> list[dict]:
    """Newest-first list of thought records."""
    u = _norm_user(username)
    if not u:
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT thought, distortion, reframe, created_at "
            "FROM thought_records WHERE username = ? "
            "ORDER BY id DESC LIMIT ?",
            (u, n),
        ).fetchall()
        return [
            {"thought": r[0], "distortion": r[1],
             "reframe": r[2], "created_at": r[3]}
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Phase 7 — Robot-Side Brain cache sync.
# ---------------------------------------------------------------------------
#
# The robot keeps a small identity/preferences cache (`~/nao_assist/brain.json`)
# capped at 64 KB. On each WS handshake the robot announces its
# ``brain_version`` and we push back any deltas it doesn't have. For Phase 7
# minimum, the deltas are:
#
# * ``last_seen_iso``      — derived from the ``users.updated_at`` epoch.
# * ``display_name``       — from ``users.display_name``.
# * ``last_recap_summary`` — first body in the ``recaps`` table for this user,
#                             truncated to 300 chars.
#
# Recaps key on ``username`` (the human-readable handle the agent loop uses),
# while the cache keys on ``face_id``. We use ``display_name`` as the username
# bridge: when display_name is set, callers like the therapist save recaps
# under that handle, so it's the right key to look up. If no display_name
# exists, recap lookup is skipped (face-only users have no therapist history
# yet by definition).
#
# ``pull_brain_updates`` is read-only and never raises — a corrupt DB or
# missing user just yields ``{}`` so the handshake degrades to "no sync".

_BRAIN_RECAP_TRUNCATE = 300


def _epoch_to_iso(epoch: float | None) -> str | None:
    """Convert a UNIX epoch (REAL column) to an RFC-3339 / ISO-8601 string.

    Returns ``None`` for falsy / invalid inputs so the caller can skip the
    field entirely rather than emit ``"1970-01-01T00:00:00Z"``.
    """
    try:
        ts = float(epoch or 0.0)
    except (TypeError, ValueError):
        return None
    if ts <= 0.0:
        return None
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    # Trim sub-second precision; the brain only needs minute-level "freshness".
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def pull_brain_updates(face_id: str, since_version: int = 2) -> dict:
    """Return the brain-cache delta for ``face_id`` since ``since_version``.

    For Phase 7 minimum, ``since_version`` is reserved for forward-compat
    (the brain schema is locked at v2 today; future versions may carry
    structural rewrites that gate certain fields). We always return the
    current values for the user; the robot is responsible for ignoring
    fields it doesn't recognize.

    Returns:
        ``{}`` if ``face_id`` is empty, the user is unknown, or the row has
        no fields worth syncing.

        Otherwise, ``{"users": {face_id: {...}}, "system_prompt_fragments": {}}``
        per the Phase 7 task map. Only fields that exist for the user are
        included — callers should treat missing keys as "no change".

    Never raises: any DB / I/O failure is caught and returned as ``{}`` so a
    flaky persistence layer can't break the WS handshake.
    """
    fid = (face_id or "").strip().lower()
    if not fid:
        return {}

    # All reads in one short-lived sqlite connection. We open it directly
    # (rather than via ``_conn()``) because the ``users`` table is owned by
    # ``server/memory.py`` — its DDL lives there, not here, and re-issuing
    # the side-table CREATE statements on a hot path would be wasteful.
    user_fields: dict[str, str] = {}
    display_name: str | None = None
    try:
        c = sqlite3.connect(_DB_PATH)
        try:
            row = c.execute(
                "SELECT display_name, updated_at FROM users WHERE face_id = ?",
                (fid,),
            ).fetchone()
        finally:
            c.close()
    except sqlite3.Error:
        return {}

    if row is None:
        return {}

    raw_name, raw_updated_at = row
    if raw_name:
        display_name = str(raw_name).strip() or None
        if display_name:
            user_fields["display_name"] = display_name

    iso = _epoch_to_iso(raw_updated_at)
    if iso:
        user_fields["last_seen_iso"] = iso

    # Recaps key on ``username``. We use ``display_name`` as the bridge —
    # it's the same handle the therapist uses when calling ``save_recap``.
    if display_name:
        try:
            recaps = load_recent_recaps(display_name, n=1)
        except sqlite3.Error:
            recaps = []
        if recaps:
            body = (recaps[0] or "").strip()
            if body:
                if len(body) > _BRAIN_RECAP_TRUNCATE:
                    body = body[: _BRAIN_RECAP_TRUNCATE - 3].rstrip() + "..."
                user_fields["last_recap_summary"] = body

    if not user_fields:
        return {}

    return {
        "users": {fid: user_fields},
        "system_prompt_fragments": {},
    }


# ---------------------------------------------------------------------------
# SAGE-CBT helpers (RQ2 runtime invariant + topology comparison).
# ---------------------------------------------------------------------------

def append_safety_event(
    username: str,
    turn_index: int,
    clause: str,
    severity: str,
    payload: str,
) -> None:
    """Record a single invariant violation. Never raises on bad input."""
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO safety_events "
                "(username, turn_index, clause, severity, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (username, int(turn_index), clause, severity, payload),
            )
    except Exception:
        # Invariant logging is best-effort; never break the response path.
        pass


def append_topology_trace(
    username: str,
    topology: str,
    user_text: str,
    proposed_reply: str,
    final_reply: str,
    verdict: str,
    affect: str,
    invariant_holds: bool,
) -> None:
    """Record one turn tuple per topology run. Never raises on bad input."""
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO topology_trace "
                "(username, topology, user_text, proposed_reply, final_reply, "
                "verdict, affect, invariant_holds) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    username,
                    topology,
                    user_text,
                    proposed_reply,
                    final_reply,
                    verdict,
                    affect,
                    1 if invariant_holds else 0,
                ),
            )
    except Exception:
        pass

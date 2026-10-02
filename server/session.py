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

# Full thought-record columns, added 2026-10-01. The table started as just
# thought + distortion + reframe; existing DBs get the rest by ALTER.
_THOUGHT_RECORD_COLUMNS = (
    ("situation", "TEXT NOT NULL DEFAULT ''"),
    ("emotion", "TEXT NOT NULL DEFAULT ''"),
    ("intensity_before", "INTEGER"),
    ("evidence_for", "TEXT NOT NULL DEFAULT ''"),
    ("evidence_against", "TEXT NOT NULL DEFAULT ''"),
    ("balanced_thought", "TEXT NOT NULL DEFAULT ''"),
    ("intensity_after", "INTEGER"),
)
# DB paths already migrated in this process (tests swap _DB_PATH).
_MIGRATED: set[str] = set()


def _migrate_thought_records(c: sqlite3.Connection) -> None:
    if _DB_PATH in _MIGRATED:
        return
    have = {row[1] for row in c.execute("PRAGMA table_info(thought_records)")}
    for name, decl in _THOUGHT_RECORD_COLUMNS:
        if name not in have:
            try:
                c.execute(f"ALTER TABLE thought_records ADD COLUMN {name} {decl}")
            except sqlite3.OperationalError:
                pass  # another connection added it first
    _MIGRATED.add(_DB_PATH)


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
    _migrate_thought_records(c)
    # Small, student-chosen between-session activities ("a 10-minute walk
    # before your 2pm class"). Written by emotion.assign_homework, reviewed
    # by emotion.review_homework, surfaced by memory.build_context_preamble.
    c.execute(
        "CREATE TABLE IF NOT EXISTS homework ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL, "
        "task TEXT NOT NULL, due_hint TEXT NOT NULL DEFAULT '', "
        "status TEXT NOT NULL DEFAULT 'open', "
        "outcome TEXT NOT NULL DEFAULT '', "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
        "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_homework_owner "
        "ON homework(owner, status)"
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


def therapy_owner(username: str) -> str:
    """The key therapy data (moods, thought records, recaps, homework) is
    stored under.

    Named users keep their plain username, so rows written before this
    helper existed stay theirs. Anonymous users get the same idle-bounded
    ``guest:<epoch>`` key as their chat history: before 2026-10-01 every
    stranger's mood and recap was written under one shared "guest" owner
    and read back to the next stranger.
    """
    if is_anonymous(username):
        return session_key_for(username)
    return username.strip()


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
        # The visit's moods / thought record / homework were filed under
        # the anonymous key; they belong to the student now too.
        if old_is_anon:
            migrate_therapy_owner(old_key, therapy_owner(new))

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


# ---------------------------------------------------------------------------
# Therapy data: recaps, mood log, thought records, homework.
# ---------------------------------------------------------------------------
#
# Every helper here takes an *owner* -- ``therapy_owner(username)`` -- not a
# raw username. A bare anonymous name ("guest", "", "unknown") is refused on
# both write and read: before 2026-10-01 every stranger's mood and recap was
# filed under one shared "guest" owner and read back to the next stranger
# (9 of 12 mood rows and both recaps on the live Pi). Refusing the bare name
# here means a caller that forgets to resolve the owner loses the row
# instead of leaking it, and the old pooled rows are never shown again.

def migrate_therapy_owner(old_owner: str, new_owner: str) -> None:
    """Re-file one owner's therapy rows under another (anonymous visit ->
    the student face recognition just named). Never raises."""
    old, new = _owner_key(old_owner), _owner_key(new_owner)
    old_raw, new_raw = (_owner_key(old_owner, lower=False),
                        _owner_key(new_owner, lower=False))
    if not old or not new or old == new:
        return
    try:
        with _conn() as c:
            for table, col in (("mood_log", "username"),
                               ("thought_records", "username"),
                               ("homework", "owner")):
                c.execute(f"UPDATE {table} SET {col} = ? WHERE {col} = ?",
                          (new, old))
            c.execute("UPDATE recaps SET username = ? WHERE username = ?",
                      (new_raw, old_raw))
    except sqlite3.Error:
        pass


def _owner_key(owner: str, *, lower: bool = True) -> str:
    """The stored form of ``owner``, or "" when it names nobody."""
    if is_anonymous(owner):
        return ""
    o = (owner or "").strip()
    return o.lower() if lower else o


def save_recap(username: str, body: str) -> int | None:
    """Store a session recap. Returns the row id (None if refused)."""
    u = _owner_key(username, lower=False)
    if not u:
        return None
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO recaps (username, body) VALUES (?, ?)", (u, body)
        )
        return int(cur.lastrowid)


def update_recap(recap_id: int, body: str) -> None:
    """Rewrite a recap in place (the same conversation, recapped again)."""
    with _conn() as c:
        c.execute("UPDATE recaps SET body = ? WHERE id = ?", (body, recap_id))


def load_recent_recaps(username: str, n: int = 3) -> list[str]:
    u = _owner_key(username, lower=False)
    if not u:
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT body FROM recaps WHERE username = ? ORDER BY id DESC LIMIT ?",
            (u, n),
        ).fetchall()
        return [r[0] for r in rows]


def _norm_user(username: str) -> str:
    """Normalize an owner for mood/thought/homework rows so writes from any
    code path (mixed-case username from session, lowercased face_id from
    memory preamble) all match the same rows. "" for anonymous names.
    """
    return _owner_key(username)


def _since_clause(since: float | None) -> tuple[str, tuple]:
    """SQL fragment limiting rows to those created at/after epoch ``since``.

    ``created_at`` is SQLite's CURRENT_TIMESTAMP: UTC, 'YYYY-MM-DD HH:MM:SS'.
    """
    if not since:
        return "", ()
    stamp = datetime.fromtimestamp(float(since), tz=timezone.utc)
    return " AND created_at >= ?", (stamp.strftime("%Y-%m-%d %H:%M:%S"),)


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


def load_recent_moods(username: str, n: int = 5, *,
                      since: float | None = None) -> list[dict]:
    """Most recent mood entries, newest first."""
    u = _norm_user(username)
    if not u:
        return []
    extra, args = _since_clause(since)
    with _conn() as c:
        rows = c.execute(
            "SELECT mood, intensity, trigger, created_at FROM mood_log "
            "WHERE username = ?" + extra + " ORDER BY id DESC LIMIT ?",
            (u, *args, n),
        ).fetchall()
        return [
            {"mood": r[0], "intensity": r[1],
             "trigger": r[2], "created_at": r[3]}
            for r in rows
        ]


def _clip_int(value) -> int | None:
    """A 0-10 rating, or None when the student didn't give one."""
    try:
        return max(0, min(10, int(value)))
    except (TypeError, ValueError):
        return None


def log_thought_record(username: str, thought: str, distortion: str,
                        reframe: str = "") -> int | None:
    """Append a CBT thought record. Returns the row id (None if refused)."""
    u = _norm_user(username)
    if not u:
        return None
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO thought_records (username, thought, distortion, reframe) "
            "VALUES (?, ?, ?, ?)",
            (u, str(thought)[:500], str(distortion)[:64], str(reframe)[:500]),
        )
        return int(cur.lastrowid)


def save_full_thought_record(
    username: str, *, thought: str, distortion: str,
    situation: str = "", emotion: str = "",
    intensity_before=None, evidence_for: str = "",
    evidence_against: str = "", balanced_thought: str = "",
    intensity_after=None, record_id: int | None = None,
) -> int | None:
    """Write the whole thought record.

    ``record_id`` is the row ``identify_distortion`` opened at step 2; it is
    completed in place so one exercise is one row. Without it (or when that
    row is gone / belongs to someone else) a new row is inserted.

    ``balanced_thought`` is the one the STUDENT chose or worded, and it is
    also written to the legacy ``reframe`` column so older readers show the
    student's own words rather than the model's first suggestion.
    """
    u = _norm_user(username)
    if not u:
        return None
    vals = {
        "thought": str(thought or "")[:500],
        "distortion": str(distortion or "")[:64],
        "reframe": str(balanced_thought or "")[:500],
        "situation": str(situation or "")[:500],
        "emotion": str(emotion or "")[:64],
        "intensity_before": _clip_int(intensity_before),
        "evidence_for": str(evidence_for or "")[:500],
        "evidence_against": str(evidence_against or "")[:500],
        "balanced_thought": str(balanced_thought or "")[:500],
        "intensity_after": _clip_int(intensity_after),
    }
    with _conn() as c:
        if record_id:
            row = c.execute(
                "SELECT thought FROM thought_records WHERE id = ? AND username = ?",
                (int(record_id), u),
            ).fetchone()
            if row is not None:
                if not vals["thought"]:
                    vals["thought"] = row[0]
                sets = ", ".join(f"{k} = ?" for k in vals)
                c.execute(
                    f"UPDATE thought_records SET {sets} WHERE id = ?",
                    (*vals.values(), int(record_id)),
                )
                return int(record_id)
        cols = ", ".join(["username", *vals])
        marks = ", ".join("?" for _ in range(len(vals) + 1))
        cur = c.execute(
            f"INSERT INTO thought_records ({cols}) VALUES ({marks})",
            (u, *vals.values()),
        )
        return int(cur.lastrowid)


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


_THOUGHT_FIELDS = (
    "thought", "distortion", "reframe", "created_at", "situation", "emotion",
    "intensity_before", "evidence_for", "evidence_against",
    "balanced_thought", "intensity_after",
)


def load_recent_thought_records(username: str, n: int = 3, *,
                                since: float | None = None) -> list[dict]:
    """Newest-first list of thought records."""
    u = _norm_user(username)
    if not u:
        return []
    extra, args = _since_clause(since)
    with _conn() as c:
        rows = c.execute(
            "SELECT " + ", ".join(_THOUGHT_FIELDS) + " "
            "FROM thought_records WHERE username = ?" + extra + " "
            "ORDER BY id DESC LIMIT ?",
            (u, *args, n),
        ).fetchall()
        return [dict(zip(_THOUGHT_FIELDS, r)) for r in rows]


HOMEWORK_STATUSES = ("open", "done", "partly", "not_done", "dropped")
_HOMEWORK_STATUS_ALIASES = {
    "yes": "done", "completed": "done", "complete": "done", "did_it": "done",
    "partial": "partly", "partially": "partly", "some": "partly",
    "no": "not_done", "skipped": "not_done", "didnt": "not_done",
    "didnt_do_it": "not_done", "not_yet": "not_done",
    "not_completed": "not_done", "missed": "not_done",
    "cancelled": "dropped", "canceled": "dropped", "abandoned": "dropped",
}


def add_homework(owner: str, task: str, due_hint: str = "") -> int | None:
    """Record one between-session activity. Returns the row id."""
    u = _norm_user(owner)
    task = (task or "").strip()
    if not u or not task:
        return None
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO homework (owner, task, due_hint) VALUES (?, ?, ?)",
            (u, task[:300], (due_hint or "").strip()[:120]),
        )
        return int(cur.lastrowid)


def load_open_homework(owner: str, n: int = 3) -> list[dict]:
    """Open homework, newest first."""
    u = _norm_user(owner)
    if not u:
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT id, task, due_hint, created_at FROM homework "
            "WHERE owner = ? AND status = 'open' ORDER BY id DESC LIMIT ?",
            (u, n),
        ).fetchall()
    return [{"id": r[0], "task": r[1], "due_hint": r[2], "created_at": r[3]}
            for r in rows]


def load_homework_since(owner: str, since: float | None,
                        n: int = 5) -> list[dict]:
    """Homework assigned or reviewed since ``since``, newest first."""
    u = _norm_user(owner)
    if not u:
        return []
    extra, args = _since_clause(since)
    extra = extra.replace("created_at", "updated_at")
    with _conn() as c:
        rows = c.execute(
            "SELECT id, task, due_hint, status, outcome, created_at, updated_at "
            "FROM homework WHERE owner = ?" + extra + " ORDER BY id DESC LIMIT ?",
            (u, *args, n),
        ).fetchall()
    keys = ("id", "task", "due_hint", "status", "outcome",
            "created_at", "updated_at")
    return [dict(zip(keys, r)) for r in rows]


def review_homework(owner: str, outcome: str, status: str,
                    homework_id: int | None = None) -> dict | None:
    """Close out homework: the given id, else the newest open one.

    Returns the updated row (id, task, status, outcome) or None when there
    was nothing open to review.
    """
    u = _norm_user(owner)
    if not u:
        return None
    norm = (status or "").strip().lower().replace("'", "")
    norm = norm.replace(" ", "_").replace("-", "_")
    norm = _HOMEWORK_STATUS_ALIASES.get(norm, norm)
    if norm not in HOMEWORK_STATUSES or norm == "open":
        norm = "partly"  # reviewed, but the model's word for it was unclear
    with _conn() as c:
        if homework_id:
            row = c.execute(
                "SELECT id, task FROM homework WHERE id = ? AND owner = ?",
                (int(homework_id), u),
            ).fetchone()
        else:
            row = c.execute(
                "SELECT id, task FROM homework WHERE owner = ? AND status = 'open' "
                "ORDER BY id DESC LIMIT 1",
                (u,),
            ).fetchone()
        if row is None:
            return None
        c.execute(
            "UPDATE homework SET status = ?, outcome = ?, "
            "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (norm, (outcome or "").strip()[:300], row[0]),
        )
    return {"id": row[0], "task": row[1], "status": norm,
            "outcome": (outcome or "").strip()[:300]}


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

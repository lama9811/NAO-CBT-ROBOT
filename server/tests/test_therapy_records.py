"""Therapy rows in session.py: full thought records, homework, recaps,
owner keys that refuse the bare anonymous name, and ``since=`` filters."""
from __future__ import annotations

import sqlite3
import time
import uuid

from server import session


def _who() -> str:
    return "student_" + uuid.uuid4().hex[:8]


def _age_rows(table: str, owner_col: str, owner: str, seconds: int) -> None:
    """Backdate every row of ``owner`` in ``table`` by ``seconds``."""
    with sqlite3.connect(session._DB_PATH) as c:
        c.execute(
            f"UPDATE {table} SET created_at = datetime(created_at, ?) "
            f"WHERE {owner_col} = ?",
            (f"-{seconds} seconds", owner.lower()),
        )


# ---------------------------------------------------------------- owner key

def test_bare_anonymous_names_are_refused_on_write_and_read():
    for anon in ("guest", "", "Guest "):
        assert session.log_thought_record(anon, "t", "labeling") is None
        assert session.add_homework(anon, "walk") is None
        assert session.save_recap(anon, "body") is None
        session.log_mood(anon, "sad", 5, "x")
        assert session.load_recent_moods(anon) == []
        assert session.load_recent_thought_records(anon) == []
        assert session.load_open_homework(anon) == []
        assert session.load_recent_recaps(anon) == []


def test_per_visit_guest_owner_is_accepted():
    owner = "guest:" + uuid.uuid4().hex[:8]
    assert session.add_homework(owner, "drink water") is not None
    assert session.load_open_homework(owner)[0]["task"] == "drink water"


def test_owner_lookup_is_case_insensitive():
    who = _who()
    session.add_homework(who.upper(), "stretch")
    assert session.load_open_homework(who)[0]["task"] == "stretch"


# ----------------------------------------------------------- thought records

def test_full_record_completes_the_row_identify_distortion_opened():
    who = _who()
    rid = session.log_thought_record(who, "I'll fail the exam", "fortune-telling")
    out = session.save_full_thought_record(
        who, record_id=rid, thought="", distortion="fortune-telling",
        situation="Got a C on the quiz", emotion="anxious",
        intensity_before=8, evidence_for="quiz grade",
        evidence_against="I passed the last two exams",
        balanced_thought="One quiz doesn't decide the exam", intensity_after=5,
    )
    assert out == rid
    rows = session.load_recent_thought_records(who, n=5)
    assert len(rows) == 1  # one exercise, one row
    r = rows[0]
    assert r["thought"] == "I'll fail the exam"  # kept from step 3
    assert r["situation"] == "Got a C on the quiz"
    assert (r["intensity_before"], r["intensity_after"]) == (8, 5)
    assert r["balanced_thought"] == "One quiz doesn't decide the exam"
    # Legacy readers see the student's words, not the model's suggestion.
    assert r["reframe"] == "One quiz doesn't decide the exam"


def test_full_record_without_an_open_row_inserts_one():
    who = _who()
    rid = session.save_full_thought_record(
        who, thought="Nobody likes me", distortion="mind reading",
        intensity_before=11, intensity_after=-3)
    assert rid
    r = session.load_recent_thought_records(who)[0]
    assert (r["intensity_before"], r["intensity_after"]) == (10, 0)


def test_full_record_cannot_overwrite_someone_elses_row():
    a, b = _who(), _who()
    rid = session.log_thought_record(a, "a's thought", "labeling")
    new_id = session.save_full_thought_record(
        b, record_id=rid, thought="b's thought", distortion="shoulds")
    assert new_id != rid
    assert session.load_recent_thought_records(a)[0]["thought"] == "a's thought"


def test_thought_records_since_filter():
    who = _who()
    session.log_thought_record(who, "old", "labeling")
    _age_rows("thought_records", "username", who, 3600)
    session.log_thought_record(who, "new", "labeling")
    since = time.time() - 60
    got = session.load_recent_thought_records(who, n=5, since=since)
    assert [r["thought"] for r in got] == ["new"]


def test_moods_since_filter():
    who = _who()
    session.log_mood(who, "sad", 7, "old")
    _age_rows("mood_log", "username", who, 3600)
    session.log_mood(who, "calm", 3, "new")
    got = session.load_recent_moods(who, since=time.time() - 60)
    assert [m["mood"] for m in got] == ["calm"]


# ------------------------------------------------------------------ homework

def test_homework_add_review_cycle():
    who = _who()
    hid = session.add_homework(who, "10-minute walk before class", "this week")
    assert session.load_open_homework(who)[0]["id"] == hid
    row = session.review_homework(who, "walked twice", "done")
    assert row == {"id": hid, "task": "10-minute walk before class",
                   "status": "done", "outcome": "walked twice"}
    assert session.load_open_homework(who) == []
    assert session.review_homework(who, "x", "done") is None  # nothing open


def test_homework_status_normalization():
    cases = {
        "Done": "done", "completed": "done", "partially": "partly",
        "didn't": "not_done", "not done": "not_done", "Not-Done": "not_done",
        "skipped": "not_done", "cancelled": "dropped",
        "banana": "partly", "open": "partly",
    }
    for given, expected in cases.items():
        who = _who()
        session.add_homework(who, "task")
        assert session.review_homework(who, "", given)["status"] == expected, given


def test_review_by_id_only_for_own_homework():
    a, b = _who(), _who()
    hid = session.add_homework(a, "a's task")
    assert session.review_homework(b, "", "done", homework_id=hid) is None
    assert session.load_open_homework(a)[0]["id"] == hid


def test_homework_since_uses_updated_at():
    who = _who()
    session.add_homework(who, "old task")
    with sqlite3.connect(session._DB_PATH) as c:
        c.execute("UPDATE homework SET created_at = datetime('now', '-2 days'),"
                  " updated_at = datetime('now', '-2 days') WHERE owner = ?",
                  (who,))
    since = time.time() - 60
    assert session.load_homework_since(who, since) == []
    session.review_homework(who, "did it", "done")  # reviewed this visit
    got = session.load_homework_since(who, since)
    assert [(h["task"], h["status"]) for h in got] == [("old task", "done")]


# -------------------------------------------------------------------- recaps

def test_save_recap_returns_id_and_update_rewrites_it():
    who = _who()
    rid = session.save_recap(who, "first")
    assert isinstance(rid, int)
    session.update_recap(rid, "second")
    assert session.load_recent_recaps(who) == ["second"]


# ----------------------------------------------------------------- migration

def test_migrate_therapy_owner_moves_every_table():
    anon = "guest:" + uuid.uuid4().hex[:8]
    who = _who()
    session.log_mood(anon, "sad", 6, "exam")
    session.log_thought_record(anon, "I'm hopeless", "labeling")
    session.add_homework(anon, "call mom")
    session.save_recap(anon, "visit recap")
    session.migrate_therapy_owner(anon, who)
    assert session.load_recent_moods(who)[0]["mood"] == "sad"
    assert session.load_recent_thought_records(who)[0]["thought"] == "I'm hopeless"
    assert session.load_open_homework(who)[0]["task"] == "call mom"
    assert session.load_recent_recaps(who) == ["visit recap"]
    assert session.load_open_homework(anon) == []


def test_migrate_therapy_owner_ignores_anonymous_targets():
    anon = "guest:" + uuid.uuid4().hex[:8]
    session.add_homework(anon, "keep me")
    session.migrate_therapy_owner(anon, "guest")
    assert session.load_open_homework(anon)[0]["task"] == "keep me"

"""DATA_RETENTION_DAYS: opt-in pruning of old rows at startup."""
import sqlite3
import time

import pytest

from server import privacy


@pytest.fixture
def db(tmp_path, monkeypatch):
    from server import memory, session as s
    path = str(tmp_path / "ret.db")
    monkeypatch.setattr(s, "_DB_PATH", path)
    monkeypatch.setattr(memory, "_DB_PATH", path)
    with s._conn():
        pass
    with memory._conn():
        pass
    return path


NOW = time.mktime((2026, 10, 2, 12, 0, 0, 0, 0, -1))


def _seed(path):
    old = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(NOW - 40 * 86400))
    new = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(NOW - 2 * 86400))
    with sqlite3.connect(path) as c:
        for ts in (old, new):
            c.execute("INSERT INTO mood_log (username, mood, intensity, "
                      "trigger, created_at) VALUES ('a', 'sad', 5, 't', ?)",
                      (ts,))
            c.execute("INSERT INTO recaps (username, body, created_at) "
                      "VALUES ('a', 'r', ?)", (ts,))
        c.execute("INSERT INTO users (face_id, created_at, updated_at) "
                  "VALUES ('a', 0, 0)")
        c.execute("INSERT INTO sessions (face_id, started_at) VALUES ('a', ?)",
                  (NOW - 40 * 86400,))
        c.execute("INSERT INTO sessions (face_id, started_at) VALUES ('a', ?)",
                  (NOW - 86400,))


def _count(path, table):
    with sqlite3.connect(path) as c:
        return c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_off_by_default(db):
    from server import config
    assert config.DATA_RETENTION_DAYS == 0
    _seed(db)
    assert privacy.prune_old_data(now=NOW) == {}
    assert _count(db, "mood_log") == 2


def test_prunes_only_rows_older_than_the_window(db):
    _seed(db)
    counts = privacy.prune_old_data(30, now=NOW)
    assert counts["mood_log"] == 1 and counts["recaps"] == 1
    assert counts["sessions"] == 1
    assert _count(db, "mood_log") == 1
    assert _count(db, "sessions") == 1


def test_missing_tables_are_skipped(tmp_path, monkeypatch):
    from server import session as s
    monkeypatch.setattr(s, "_DB_PATH", str(tmp_path / "empty.db"))
    assert privacy.prune_old_data(30, now=NOW) == {}

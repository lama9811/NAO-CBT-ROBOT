"""Phase 6 — camera-consent default + first-turn announce + migration tests.

These tests cover the surfaces owned by the sibling ``camera-consent`` agent
(see ``docs/PHASE_6_TASK_MAP.md``):

  - ``server/session.py``  default ``camera_consent`` flips 0 → 1; new helper
    ``is_first_turn(session_id)`` returns True on the first call per session.
  - ``server/migrations/0001_camera_default_on.py``  idempotent migration that
    bumps the column default for new rows but leaves existing rows untouched.
  - ``server/app_ws.py``  first-turn announce text (sourced from
    ``config.CAMERA_ANNOUNCE_TEXT``) — tested as a string contract.

The migration module + ``is_first_turn`` helper + announce-text config var
are owned by the sibling worktree and may not have landed yet on
``dev/architecture-rework``. We use ``pytest.importorskip`` and ``hasattr``
guards to keep the file collectable + green either way. When the sibling
agent merges, these tests start running in earnest with no edits required.

All DB writes go to ``tmp_path`` SQLite files — no ``server/nao.db`` writes
ever escape the test sandbox.
"""
from __future__ import annotations

import importlib

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1) Default camera_consent for a brand-new user is 1 (was 0 before Phase 6).
# ─────────────────────────────────────────────────────────────────────────────


def test_new_user_defaults_to_camera_consent_on(tmp_path, monkeypatch):
    """First call to ``get_camera_consent`` for an unknown user should return
    True, and the underlying row should persist as ``camera_consent=1``.

    The test points the session module at a fresh per-test SQLite file so the
    user_prefs table starts empty. We then assert both the return value and
    the literal column value to catch any future refactor that returns True
    by accident while writing 0 to the DB.
    """
    from server import session as s

    db = tmp_path / "phase6_consent.db"
    monkeypatch.setattr(s, "_DB_PATH", str(db))

    # First call — unknown user, should auto-insert with consent on.
    assert s.get_camera_consent("alice") is True

    # Second call — row exists, must still be True (no flip on read).
    assert s.get_camera_consent("alice") is True

    # And the underlying column literally stores 1, not just any truthy value.
    import sqlite3
    with sqlite3.connect(str(db)) as conn:
        row = conn.execute(
            "SELECT camera_consent FROM user_prefs WHERE username = ?",
            ("alice",),
        ).fetchone()
    assert row is not None
    assert row[0] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2) Migration is idempotent — running twice is identical to running once.
# ─────────────────────────────────────────────────────────────────────────────


def _import_migration():
    """Best-effort import of the sibling-owned migration module.

    Returns the loaded module, or skips the test when the sibling worktree
    hasn't merged yet. The expected dotted path is ``server.migrations.0001_camera_default_on``;
    the leading digit makes a plain ``import`` statement awkward, so we use
    ``importlib`` directly.
    """
    try:
        return importlib.import_module("server.migrations.0001_camera_default_on")
    except ModuleNotFoundError:
        try:
            return importlib.import_module(
                "server.migrations.camera_default_on_0001"
            )
        except ModuleNotFoundError:
            pytest.skip(
                "server/migrations/0001_camera_default_on.py not present "
                "(owned by sibling Phase 6 camera-consent agent)"
            )


def test_migration_is_idempotent(tmp_path, monkeypatch):
    """Running the migration a second time must be a no-op.

    The expected pattern (per task map): record migration runs in a tiny
    ``migrations`` table, gate the body of the migration on whether the row
    exists. The post-state of the user_prefs schema must be identical
    between the first and second invocations.
    """
    mig = _import_migration()
    if not hasattr(mig, "run"):
        pytest.skip("migration module missing top-level run() entry point")

    from server import session as s

    db = tmp_path / "phase6_idempotent.db"
    monkeypatch.setattr(s, "_DB_PATH", str(db))
    # Some migrations also read the path off config — keep both in sync so
    # whichever surface they pick still lands in the tmp DB.
    from server import config
    monkeypatch.setattr(config, "SESSION_DB", str(db), raising=False)

    # Bootstrap by touching get_camera_consent — that creates the table.
    s.get_camera_consent("seed")

    # First run + capture user_prefs schema/contents.
    mig.run()
    import sqlite3
    with sqlite3.connect(str(db)) as conn:
        rows_after_first = conn.execute(
            "SELECT username, camera_consent FROM user_prefs ORDER BY username"
        ).fetchall()
        schema_after_first = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'user_prefs'"
        ).fetchone()

    # Second run — must not raise, must not change anything.
    mig.run()
    with sqlite3.connect(str(db)) as conn:
        rows_after_second = conn.execute(
            "SELECT username, camera_consent FROM user_prefs ORDER BY username"
        ).fetchall()
        schema_after_second = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'user_prefs'"
        ).fetchone()

    assert rows_after_first == rows_after_second
    assert schema_after_first == schema_after_second


# ─────────────────────────────────────────────────────────────────────────────
# 3) Existing user with camera_consent=0 is NOT flipped by the migration.
# ─────────────────────────────────────────────────────────────────────────────


def test_existing_user_with_consent_off_stays_off_after_migration(
    tmp_path, monkeypatch
):
    """Operator-policy contract: the migration only changes the default for
    NEW rows. Anyone who explicitly opted out before Phase 6 keeps that
    choice. The migration must NOT do a blanket UPDATE on the column.
    """
    mig = _import_migration()
    if not hasattr(mig, "run"):
        pytest.skip("migration module missing top-level run() entry point")

    from server import session as s
    from server import config

    db = tmp_path / "phase6_existing.db"
    monkeypatch.setattr(s, "_DB_PATH", str(db))
    monkeypatch.setattr(config, "SESSION_DB", str(db), raising=False)

    # Seed an opted-out user BEFORE the migration runs.
    s.set_camera_consent("optedout_user", False)
    assert s.get_camera_consent("optedout_user") is False

    mig.run()

    # And after — still False.
    assert s.get_camera_consent("optedout_user") is False


# ─────────────────────────────────────────────────────────────────────────────
# 4) is_first_turn(session_id): True on first call, False afterwards.
# ─────────────────────────────────────────────────────────────────────────────


def test_is_first_turn_true_then_false(tmp_path, monkeypatch):
    """The helper is used by ``app_ws.py`` to decide whether to inject the
    audible "camera is on" heads-up. After it returns True once for a given
    session, every subsequent call must return False — otherwise the user
    hears the announcement on every turn.
    """
    from server import session as s

    if not hasattr(s, "is_first_turn"):
        pytest.skip(
            "session.is_first_turn not present "
            "(owned by sibling Phase 6 camera-consent agent)"
        )

    db = tmp_path / "phase6_first_turn.db"
    monkeypatch.setattr(s, "_DB_PATH", str(db))

    sess_id = "user:alice"
    # First call: brand-new session, should be True.
    assert s.is_first_turn(sess_id) is True
    # Second call: helper should have flipped state, expect False.
    assert s.is_first_turn(sess_id) is False
    # And idempotently False on every further call.
    assert s.is_first_turn(sess_id) is False


# ─────────────────────────────────────────────────────────────────────────────
# 5) The first-turn announce text matches the contract from the task map.
# ─────────────────────────────────────────────────────────────────────────────


def test_first_turn_announce_text_is_expected_string():
    """The CAMERA_ANNOUNCE_TEXT config default is part of the public contract:
    the NAO speaks it verbatim on the first turn of a vision-on session, so
    any change should be a deliberate edit to both the config + this test.

    Per ``docs/PHASE_6_TASK_MAP.md`` § ``server/config.py additions``:

        CAMERA_ANNOUNCE_TEXT = os.environ.get(
            "CAMERA_ANNOUNCE_TEXT",
            "Heads up — my camera is on for this conversation. "
            "Say 'stop watching me' anytime."
        )
    """
    from server import config

    if not hasattr(config, "CAMERA_ANNOUNCE_TEXT"):
        pytest.skip(
            "config.CAMERA_ANNOUNCE_TEXT not present "
            "(owned by sibling Phase 6 camera-consent agent)"
        )

    # Reworded 2026-09-30: the old "Say 'stop watching me' anytime" was
    # itself a camera-off command, and NAO hears its own speaker.
    expected = "Heads up, my camera is on for this conversation."
    assert config.CAMERA_ANNOUNCE_TEXT == expected

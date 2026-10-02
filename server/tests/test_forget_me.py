"""'Forget me' deletes everything stored about one student, and only them."""
import asyncio
import sqlite3

import pytest

from server import privacy


@pytest.fixture
def db(tmp_path, monkeypatch):
    from server import memory, session as s
    path = str(tmp_path / "forget.db")
    monkeypatch.setattr(s, "_DB_PATH", path)
    monkeypatch.setattr(memory, "_DB_PATH", path)
    with s._conn():
        pass  # create the therapy tables
    with memory._conn():
        pass
    return path


def _seed(path, user, *, homework=True):
    with sqlite3.connect(path) as c:
        c.execute("INSERT INTO mood_log (username, mood, intensity, trigger) "
                  "VALUES (?, 'sad', 6, 'exam')", (user,))
        c.execute("INSERT INTO thought_records (username, thought, distortion) "
                  "VALUES (?, 'I will fail', 'catastrophizing')", (user,))
        c.execute("INSERT INTO recaps (username, body) VALUES (?, 'r')", (user,))
        c.execute("INSERT INTO user_prefs (username) VALUES (?)", (user,))
        c.execute("INSERT INTO safety_events (username, clause, severity) "
                  "VALUES (?, 'crisis_gate', 'keyword')", (user,))
        c.execute("INSERT INTO users (face_id, display_name, created_at, "
                  "updated_at) VALUES (?, ?, 0, 0)", (user.lower(), user))
        c.execute("INSERT INTO sessions (face_id, started_at, summary) "
                  "VALUES (?, 0, 's')", (user.lower(),))
        if homework:
            c.execute("CREATE TABLE IF NOT EXISTS homework (id INTEGER "
                      "PRIMARY KEY, username TEXT, task TEXT, created_at "
                      "TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            c.execute("INSERT INTO homework (username, task) VALUES (?, 'x')",
                      (user,))


def _count(path, table, where="1=1", args=()):
    with sqlite3.connect(path) as c:
        return c.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}",
                         args).fetchone()[0]


def _add_history(user, text):
    from server import session as s
    sess = s.get_or_create_session(user)
    asyncio.run(sess.add_items([{"role": "user", "content": text}]))
    return sess


def test_forget_deletes_one_user_and_leaves_others(db):
    _seed(db, "Alice")
    _seed(db, "Bob")
    _add_history("Alice", "private")
    bob = _add_history("Bob", "keep me")

    counts = privacy.forget_user_data("Alice")

    for table in ("mood_log", "thought_records", "recaps", "user_prefs",
                  "homework"):
        assert _count(db, table, "lower(username)='alice'") == 0, table
        assert _count(db, table, "username='Bob'") == 1, table
    assert _count(db, "users", "face_id='alice'") == 0
    assert _count(db, "sessions", "face_id='alice'") == 0
    assert _count(db, "users", "face_id='bob'") == 1
    # Crisis rows are kept for safety review but no longer name anyone.
    assert _count(db, "safety_events") == 2
    assert _count(db, "safety_events", "username='Alice'") == 0
    # Chat history.
    from server import session as s
    alice = s.get_or_create_session("Alice")
    assert asyncio.run(alice.get_items()) == []
    assert len(asyncio.run(bob.get_items())) == 1
    assert counts["chat_messages"] == 1


def test_forget_is_case_insensitive(db):
    _seed(db, "alice")
    privacy.forget_user_data("ALICE")
    assert _count(db, "mood_log") == 0


def test_forget_without_homework_table(db):
    _seed(db, "Alice", homework=False)
    counts = privacy.forget_user_data("Alice")
    assert "homework" not in counts and "error" not in counts
    assert _count(db, "mood_log") == 0


def test_forget_anonymous_visit(db):
    from server import session as s
    owner = s.therapy_owner("guest")
    _seed(db, owner)
    _add_history("guest", "anon words")
    privacy.forget_user_data("guest")
    assert _count(db, "mood_log") == 0
    assert asyncio.run(s.get_or_create_session("guest").get_items()) == []


def test_forget_clears_conversation_state(db):
    from server import conversation_state
    conv = conversation_state.state_for("Alice")
    conv["cbt_step"] = 3
    privacy.forget_user_data("Alice")
    assert conversation_state.state_for("Alice") == {}


# ───────── voice phrase + confirmation ─────────
@pytest.mark.parametrize("text", [
    "forget me", "Please forget about me.", "delete my data",
    "erase all my conversations", "forget everything you know about me",
])
def test_forget_phrases(text):
    assert privacy.detect_forget_request(text)


@pytest.mark.parametrize("text", [
    "don't forget me", "never forget me okay", "I forgot my homework",
    "what is the weather", "",
    "yesterday my friend told me a long story about how she wanted to "
    "delete my data from her phone",
])
def test_not_forget_phrases(text):
    assert not privacy.detect_forget_request(text)


def test_two_step_flow():
    conv = {}
    assert privacy.handle_forget_turn(conv, "forget me", now=100) == "ask"
    assert privacy.handle_forget_turn(conv, "yes please", now=110) == "confirm"
    assert "forget_pending" not in conv


def test_anything_but_yes_cancels():
    conv = {}
    privacy.handle_forget_turn(conv, "delete my data", now=100)
    assert privacy.handle_forget_turn(conv, "no wait", now=105) == "cancel"
    assert privacy.handle_forget_turn(conv, "yes", now=106) is None


def test_late_yes_does_not_delete():
    conv = {}
    privacy.handle_forget_turn(conv, "forget me", now=100)
    late = 100 + privacy.FORGET_CONFIRM_WINDOW_S + 1
    assert privacy.handle_forget_turn(conv, "yes", now=late) is None


def test_app_ws_emit_forget_confirm_deletes(db, monkeypatch):
    from server import app_ws
    _seed(db, "Alice")
    said = []

    async def fake_speak(ws, sess, text, phase, phase_ms):
        said.append(text)
        return True

    async def fake_send(ws, payload):
        pass

    monkeypatch.setattr(app_ws, "_speak_line", fake_speak)
    monkeypatch.setattr(app_ws, "_send_json", fake_send)
    sess = app_ws._Session("Alice")
    asyncio.run(app_ws._emit_forget(None, sess, "ask", {}))
    assert _count(db, "mood_log") == 1
    asyncio.run(app_ws._emit_forget(None, sess, "confirm", {}))
    assert _count(db, "mood_log") == 0
    assert said == [privacy.FORGET_CONFIRM_PROMPT, privacy.FORGET_DONE_REPLY]


def test_forget_me_tool_requires_confirmation(db):
    from server.tools import privacy_tools
    _seed(db, "Alice")

    class Ctx:
        context = {"username": "Alice"}

    assert privacy_tools._forget_me_impl(Ctx(), False).startswith("not_confirmed")
    assert _count(db, "mood_log") == 1
    assert privacy_tools._forget_me_impl(Ctx(), True).startswith("deleted")
    assert _count(db, "mood_log") == 0

"""Anonymous chat history must not leak between people.

Root cause of the bug this pins (found 2026-08-24): ``get_or_create_session``
keyed the Agents SDK ``SQLiteSession`` by ``user:<username>``. Everyone who is
not face-recognised is ``guest``, so a single ``user:guest`` row accumulated
*every* anonymous conversation, forever — 709 messages spanning 2026-05-11 to
2026-08-24 on the live Pi. NAO quoted a name introduced on Jul 30 back to a
different person on Aug 24, 25 days later. On a CBT robot that means one
student's disclosures sit in the model's context while the next student talks.

The fix scopes anonymous history to an *idle-bounded epoch*: a fresh key is
minted once the anonymous lane has been quiet for ``GUEST_IDLE_RESET_S``.

The idle window is load-bearing, not decoration. ``session.py`` documents that
WS connections "drop every few seconds during long TTS playbacks" and that each
reconnect mints a fresh ``session_id`` — so keying anonymous history to the
per-connection UUID would shred a conversation mid-sentence. The epoch must
survive reconnect storms and still reset between two different people.

Named users are deliberately untouched: therapy continuity across visits is a
feature, and ``migrate_username`` hands the anonymous row over once face reco
puts a real name to the voice.
"""
from __future__ import annotations

import asyncio

import pytest

from server import session as S


@pytest.fixture(autouse=True)
def _fresh_epoch(monkeypatch):
    """Each test starts with no anonymous epoch in flight."""
    monkeypatch.setattr(S, "_GUEST_EPOCH_TOKEN", None, raising=False)
    monkeypatch.setattr(S, "_GUEST_EPOCH_LAST_SEEN", 0.0, raising=False)


def test_anonymous_reconnect_within_window_keeps_same_key():
    """A WS reconnect seconds later is the same person mid-sentence."""
    first = S.session_key_for("guest", now=1000.0)
    reconnect = S.session_key_for("guest", now=1004.0)

    assert reconnect == first


def test_anonymous_key_resets_after_idle_gap():
    """The next person to walk up gets a clean slate."""
    first = S.session_key_for("guest", now=1000.0)
    later = S.session_key_for("guest", now=1000.0 + S.GUEST_IDLE_RESET_S + 1)

    assert later != first


def test_anonymous_epoch_is_extended_by_activity():
    """Idle is measured from last activity, not from when the epoch opened.

    A 40-minute conversation with steady turns must stay one conversation even
    though its total length exceeds the idle window.
    """
    start = S.session_key_for("guest", now=1000.0)
    step = S.GUEST_IDLE_RESET_S * 0.5
    mid = S.session_key_for("guest", now=1000.0 + step)
    end = S.session_key_for("guest", now=1000.0 + step * 2)

    assert mid == start
    assert end == start


def test_anonymous_key_is_never_the_shared_guest_row():
    """The `user:guest` row is what leaked; nothing may write to it again."""
    key = S.session_key_for("guest", now=1000.0)

    assert key != "user:guest"


@pytest.mark.parametrize("anon", ["guest", "", "unknown", "GUEST"])
def test_all_anonymous_spellings_scope_to_the_epoch(anon):
    """`app_ws._ANONYMOUS_USERNAMES` treats these as the same non-identity."""
    key = S.session_key_for(anon, now=1000.0)

    assert key != "user:{}".format(anon)
    assert key.startswith("guest:")


def test_named_user_key_is_stable_across_months():
    """Therapy continuity for a recognised student is intentional."""
    first = S.session_key_for("alice", now=1000.0)
    much_later = S.session_key_for("alice", now=1000.0 + 60 * 86400)

    assert first == much_later == "user:alice"


def test_named_user_key_is_case_insensitive():
    """`aayush` / `Aayush` / `ayush` fragmented into three rows on the Pi."""
    assert S.session_key_for("Aayush") == S.session_key_for("aayush")


def test_two_anonymous_people_never_share_history():
    """End-to-end statement of the bug, in one assertion.

    Person A talks, leaves. The lane goes quiet past the window. Person B walks
    up. B must not be handed A's transcript.
    """
    person_a = S.session_key_for("guest", now=1000.0)
    person_b = S.session_key_for(
        "guest", now=1000.0 + S.GUEST_IDLE_RESET_S + 60,
    )

    assert person_a != person_b


def test_second_stranger_cannot_read_the_first_strangers_transcript(
    tmp_path, monkeypatch,
):
    """The bug, stated end-to-end against a real SQLiteSession.

    This is the assertion that would have caught the live incident: a student
    disclosed something as `guest`, and 25 days later NAO recited it to whoever
    walked up next. Key-level tests above pin the mechanism; this one pins the
    consequence, so a future refactor that reintroduces a shared row fails here
    even if the key scheme changes shape.
    """
    monkeypatch.setattr(S, "_DB_PATH", str(tmp_path / "nao.db"))

    person_a = S.get_or_create_session("guest")
    asyncio.run(person_a.add_items(
        [{"role": "user", "content": "my name is Bob and I'm failing calculus"}]
    ))

    # A leaves; the robot sits idle past the reset window; B walks up.
    monkeypatch.setattr(
        S, "_GUEST_EPOCH_LAST_SEEN",
        S._GUEST_EPOCH_LAST_SEEN - (S.GUEST_IDLE_RESET_S + 60),
    )
    person_b = S.get_or_create_session("guest")

    transcript_b = str(asyncio.run(person_b.get_items()))
    assert "Bob" not in transcript_b
    assert "calculus" not in transcript_b


def test_face_reco_handoff_carries_the_live_anonymous_epoch(
    tmp_path, monkeypatch,
):
    """Scoping anonymous history must not break the handoff it feeds.

    Someone talks before being recognised, then face reco puts a name to them.
    Everything they said as `guest` has to follow them into `user:<name>` —
    that is what `migrate_username` exists for.
    """
    monkeypatch.setattr(S, "_DB_PATH", str(tmp_path / "nao.db"))

    anon = S.get_or_create_session("guest")
    asyncio.run(anon.add_items([{"role": "user", "content": "finals are rough"}]))

    S.migrate_username("guest", "alice")

    items = asyncio.run(S.get_or_create_session("alice").get_items())
    assert any("finals are rough" in str(i.get("content", "")) for i in items)


def test_anonymous_epoch_is_retired_after_handoff(tmp_path, monkeypatch):
    """The next stranger must not land in the epoch we just gave away.

    Without retiring it, the epoch is still live and inside its idle window, so
    the next anonymous turn would reopen the row that now belongs to Alice.
    """
    monkeypatch.setattr(S, "_DB_PATH", str(tmp_path / "nao.db"))

    before = S.session_key_for("guest")
    S.migrate_username("guest", "alice")
    after = S.session_key_for("guest")

    assert after != before

"""Anonymous ("guest") sessions must not inherit or leak a voice profile.

Every WS session opens as ``guest`` and is only renamed once face reco or the
name prompt resolves an identity — so the greeting, camera announce and name
prompt are all synthesized under ``username="guest"``. Because
``user_prefs.voice_profile`` is keyed by username, one stranger saying "man
voice" during their guest phase used to pin the *man* voice onto the shared
``guest`` row forever. Every later session then opened in that voice and
audibly flipped to the identified user's voice a few seconds in.

These tests lock in the two halves of the fix:
  1. a voice pick made while anonymous applies to that session only;
  2. profile resolution for an anonymous session ignores any stored ``guest``
     row (including legacy poisoned ones) and uses the server default.
"""
from __future__ import annotations


def test_guest_voice_pick_is_session_scoped(tmp_path, monkeypatch):
    """"man voice" while anonymous takes effect now, but never persists."""
    from server import app_ws
    from server import session as s

    monkeypatch.setattr(s, "_DB_PATH", str(tmp_path / "voice_guest.db"))

    sess = app_ws._Session(username="guest")
    app_ws._persist_voice_profile(sess, "man")

    # Applies to the live session...
    assert sess.voice_profile_override == "man"
    # ...but leaves no durable trace for the next stranger.
    assert s.get_voice_profile("guest") == ""


def test_identified_user_voice_pick_still_persists(tmp_path, monkeypatch):
    """The anonymous carve-out must not break real per-user preferences."""
    from server import app_ws
    from server import session as s

    monkeypatch.setattr(s, "_DB_PATH", str(tmp_path / "voice_named.db"))

    sess = app_ws._Session(username="mia")
    app_ws._persist_voice_profile(sess, "man")

    assert sess.voice_profile_override == "man"
    assert s.get_voice_profile("mia") == "man"


def test_anonymous_resolution_ignores_stored_guest_row(tmp_path, monkeypatch):
    """A legacy poisoned ``guest`` row must not colour the opening lines."""
    from server import app_ws, config
    from server import session as s

    monkeypatch.setattr(s, "_DB_PATH", str(tmp_path / "voice_legacy.db"))
    s.set_voice_profile("guest", "man")  # what today's live DB looks like

    resolved = app_ws._resolve_voice_profile("guest", None)

    assert resolved == config.ELEVENLABS_DEFAULT_PROFILE


def test_identified_user_resolution_reads_stored_profile(tmp_path, monkeypatch):
    from server import app_ws
    from server import session as s

    monkeypatch.setattr(s, "_DB_PATH", str(tmp_path / "voice_read.db"))
    s.set_voice_profile("mia", "man")

    assert app_ws._resolve_voice_profile("mia", None) == "man"


def test_session_override_beats_stored_profile(tmp_path, monkeypatch):
    """An in-session switch wins over the persisted pick for this session."""
    from server import app_ws
    from server import session as s

    monkeypatch.setattr(s, "_DB_PATH", str(tmp_path / "voice_override.db"))
    s.set_voice_profile("mia", "man")

    assert app_ws._resolve_voice_profile("mia", "neutral") == "neutral"

"""NAO must not call a new speaker by the previous person's name.

Incident, 2026-09-30: NAO matched Mingma's face to the learned face "Mason"
at a weak 0.646, and because identity was fixed at wake it called everyone
Mason for the rest of the conversation. The robot now re-scans the face
after each turn toward a voice and reports `speaker_identified`.
"""
import asyncio
import sys
import types

import pytest

from server import app_ws


def _sess(name="guest"):
    s = app_ws._Session(name)
    app_ws._IDENTIFIED_USERS.pop(s.session_id, None)
    return s


def _apply(sess, **data):
    asyncio.run(app_ws._apply_speaker_identified(sess, data))


def test_unknown_face_drops_the_previous_name():
    sess = _sess("mason")
    app_ws._IDENTIFIED_USERS[sess.session_id] = {"name": "Mason",
                                                 "recognized": True}
    _apply(sess, name=None, recognized=False, face_visible=True)
    assert sess.username == "guest"
    ident = app_ws._IDENTIFIED_USERS[sess.session_id]
    assert ident["name"] is None and ident["recognized"] is False


def test_recognised_new_speaker_takes_over():
    sess = _sess("mason")
    app_ws._IDENTIFIED_USERS[sess.session_id] = {"name": "Mason",
                                                 "recognized": True}
    _apply(sess, name="Mingma", recognized=True, face_visible=True)
    assert sess.username == "mingma"
    assert app_ws._IDENTIFIED_USERS[sess.session_id]["name"] == "Mingma"


def test_no_face_in_view_changes_nothing():
    sess = _sess("mason")
    app_ws._IDENTIFIED_USERS[sess.session_id] = {"name": "Mason",
                                                 "recognized": True}
    _apply(sess, name=None, recognized=False, face_visible=False)
    assert sess.username == "mason"


def test_same_person_again_is_a_noop():
    sess = _sess("mason")
    app_ws._IDENTIFIED_USERS[sess.session_id] = {"name": "Mason",
                                                 "recognized": True,
                                                 "greeted": True}
    _apply(sess, name="mason", recognized=True, face_visible=True)
    assert sess.username == "mason"
    assert app_ws._IDENTIFIED_USERS[sess.session_id]["greeted"] is True


def test_control_frame_is_routed():
    sess = _sess("mason")
    app_ws._IDENTIFIED_USERS[sess.session_id] = {"name": "Mason",
                                                 "recognized": True}

    class WS:
        async def send_text(self, t):
            pass

    asyncio.run(app_ws._ingest_control(WS(), sess, {
        "type": "control", "subtype": "speaker_identified",
        "data": {"name": None, "face_visible": True}}))
    assert sess.username == "guest"


# ─────────────────────────── robot: weak matches ────────────────────────────
def _face_naoqi():
    if "nao" not in sys.path:
        sys.path.insert(0, "nao")
    return pytest.importorskip("utils.face_naoqi")


def _record(score, name):
    shape = [0, 0.1, 0.0, 0.2, 0.2]
    return [shape, [7, score, name, [], [], [], [], []]]


def test_weak_face_match_is_unknown(monkeypatch):
    fn = _face_naoqi()
    monkeypatch.delenv("FACE_RECO_MIN_SCORE", raising=False)
    face = fn._parse_face_record(_record(0.646, "Mason"))
    assert face["name"] == ""


def test_strong_face_match_keeps_name(monkeypatch):
    fn = _face_naoqi()
    monkeypatch.delenv("FACE_RECO_MIN_SCORE", raising=False)
    face = fn._parse_face_record(_record(0.82, "Mingma"))
    assert face["name"] == "Mingma"


def test_threshold_is_tunable(monkeypatch):
    fn = _face_naoqi()
    monkeypatch.setenv("FACE_RECO_MIN_SCORE", "0.6")
    assert fn._parse_face_record(_record(0.646, "Mason"))["name"] == "Mason"

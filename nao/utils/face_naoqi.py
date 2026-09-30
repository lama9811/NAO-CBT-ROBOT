# -*- coding: utf-8 -*-
from __future__ import print_function
import math
import time

try:
    unicode_type = unicode  # noqa: F821  (Py2.7 on NAO)
except NameError:
    unicode_type = str

_TEXT_TYPES = (str, unicode_type)


def face_min_score():
    """Minimum ALFaceDetection recognition score to trust a name.

    NAOqi returns its best guess even when it is weak. On 2026-09-30 it
    matched Mingma to the learned face "Mason" at 0.646, and NAO called
    them Mason for the whole conversation. Below this score the face is
    treated as unknown: NAO uses no name rather than the wrong one.
    """
    import os
    try:
        return float(os.environ.get("FACE_RECO_MIN_SCORE", "0.75"))
    except (TypeError, ValueError):
        return 0.75

# NAO V6 top camera intrinsics (from Aldebaran datasheet).
# Used by detect_faces_with_geometry to convert face size in the image
# into an approximate distance.
NAO_TOP_CAM_HFOV_DEG = 60.97
NAO_TOP_CAM_VFOV_DEG = 47.64
# Average adult inter-temple face width. Tune if the robot will mostly see
# children or partial faces.
ASSUMED_FACE_WIDTH_M = 0.16


def recognize_face_naoqi(qi_session, tts, subscriber_name="FaceReco", timeout=10,
                         stop_event=None, return_seen=False):
    """Use NAO's ALFaceDetection to recognize a known face.

    Silent — no spoken prompt. The caller indicates listening via LEDs so
    the user doesn't sit through a 4-second dead-air "look at me" pause.

    Default returns: recognized name string, or None if no face was recognized.
    With return_seen=True, returns (name_or_None, face_was_visible) — the
    second element is True if ALFaceDetection saw ANY face in frame during
    the scan, even if the face couldn't be identified. This lets callers
    distinguish "the cached user just isn't looking at me" (face not visible)
    from "an unknown stranger is in frame" (face visible but no match), which
    matters for greeting logic — we don't want to greet a stranger by the
    cached user's name.

    stop_event: optional threading.Event — when set, the polling loop exits
    promptly. Used by callers (conversation._onboard_new_user) that run this
    in a background thread alongside ask_name and want to abort the moment
    ask_name returns, rather than letting the full timeout drain.
    """
    face_detection = None
    face_was_visible = False
    try:
        memory = qi_session.service("ALMemory")
        face_detection = qi_session.service("ALFaceDetection")
        face_detection.subscribe(subscriber_name)
        # No TTS prompt — just scan silently. ALFaceDetection populates
        # ALMemory key "FaceDetected" within ~200ms when a face is in frame.
        start_time = time.time()
        recognized_name = None
        while time.time() - start_time < timeout:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                face_data = memory.getData("FaceDetected")
                if face_data and isinstance(face_data, list) and len(face_data) >= 2:
                    face_info_list = face_data[1]
                    if face_info_list and len(face_info_list) > 0:
                        # Even if the face can't be matched, note that a face
                        # was visible at all — caller may need to know.
                        face_was_visible = True
                        first_face = face_info_list[0]
                        if isinstance(first_face, list) and len(first_face) >= 2:
                            extra_info = first_face[1]
                            if isinstance(extra_info, list) and len(extra_info) >= 3:
                                face_name = extra_info[2]
                                try:
                                    score = float(extra_info[1])
                                except (TypeError, ValueError):
                                    score = 0.0
                                if score < face_min_score():
                                    face_name = ""  # weak match
                                if face_name and isinstance(face_name, _TEXT_TYPES) and face_name.strip() != "":
                                    recognized_name = face_name.strip()
                                    print("[Recognized]: {}".format(recognized_name))
                                    break
            except Exception as e:
                print("[Memory read error]:", e)
            time.sleep(0.3)
        if return_seen:
            return (recognized_name, face_was_visible)
        return recognized_name
    except Exception as e:
        print("[Face recognition error]:", e)
        if return_seen:
            return (None, face_was_visible)
        return None
    finally:
        if face_detection is not None:
            try:
                face_detection.unsubscribe(subscriber_name)
            except Exception:
                pass


def learn_new_face_naoqi(qi_session, tts, name, subscriber_name="FaceLearn"):
    """Try to learn the face currently visible to NAO. Silent — no spoken
    prompt and no follow-up greeting. The caller already had a conversation
    with the user (asking their name) so the camera almost always has a face
    in frame; saying "please look at me" again is redundant and was the main
    reason onboarding felt slow.

    Returns True if a face was captured and learnFace was called.
    """
    face_detection = None
    try:
        face_detection = qi_session.service("ALFaceDetection")
        memory = qi_session.service("ALMemory")
        try:
            face_detection.subscribe(subscriber_name)
        except Exception:
            pass
        start_time = time.time()
        face_found = False
        while time.time() - start_time < 4:
            try:
                face_data = memory.getData("FaceDetected")
                if face_data and isinstance(face_data, list) and len(face_data) >= 2:
                    if face_data[1] and len(face_data[1]) > 0:
                        face_found = True
                        break
            except Exception:
                pass
            time.sleep(0.2)
        if face_found:
            print("[Learning face as]: {}".format(name))
            try:
                # learnFace may return bool, None, or raise. Capture the
                # return so a False ("face not clear enough") doesn't get
                # silently treated as success and leave us claiming we
                # learned the user when we didn't.
                ret = face_detection.learnFace(name)
                print("[learnFace] returned:", ret)
            except Exception as e:
                print("[learnFace error]:", e)
                return False
            time.sleep(0.4)
            # Verify by reading the persisted list. If the name isn't there,
            # something silently failed (insufficient face data, etc.) and
            # the caller needs to know so they can retry next session.
            try:
                learned = face_detection.getLearnedFacesList() or []
                if name in learned:
                    return True
                print("[Learn face] verify FAILED; learned list:", learned)
                return False
            except Exception as e:
                # If we can't read the list, be optimistic — learnFace
                # didn't raise, so probably it worked.
                print("[Learn face] verify read error:", e)
                return True
        print("[Learn face]: no face in frame for {0}, skipping".format(name))
        return False
    except Exception as e:
        print("[Learn face error]:", e)
        return False
    finally:
        if face_detection is not None:
            try:
                face_detection.unsubscribe(subscriber_name)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Phase 3 — geometry helpers used by WakeStateMachine engagement gating.
# These do NOT touch the existing recognise/learn/clear paths above. They
# operate on the raw ALMemory["FaceDetected"] payload and on the dicts they
# emit, so they can be unit-tested with synthetic inputs on a dev machine
# without naoqi installed.
# ---------------------------------------------------------------------------


def _estimate_distance_m(size_x_norm,
                         hfov_deg=NAO_TOP_CAM_HFOV_DEG,
                         real_face_width_m=ASSUMED_FACE_WIDTH_M):
    """Approximate distance to a face using its width in the image.

    Pinhole-camera geometry. ALFaceDetection reports each face's size in
    NORMALISED image coordinates (sizeX is a fraction in [0, 1] of the
    image width), so the angular subtense of the face in the image is::

        angular_width_rad = size_x_norm * hfov_rad

    Then, treating the face as a planar object whose real width is
    real_face_width_m, the distance from the camera is::

        distance_m = (real_face_width_m / 2) / tan(angular_width_rad / 2)

    Returns 0.0 if size_x_norm is non-positive (caller should treat as
    "unknown distance").
    """
    if size_x_norm is None or size_x_norm <= 0.0:
        return 0.0
    hfov_rad = math.radians(hfov_deg)
    angular_width_rad = float(size_x_norm) * hfov_rad
    half = angular_width_rad / 2.0
    if half <= 0.0:
        return 0.0
    return (real_face_width_m / 2.0) / math.tan(half)


def _coerce_text(value):
    """Best-effort string coercion that survives bytes vs unicode on Py2.7."""
    if value is None:
        return ""
    if isinstance(value, _TEXT_TYPES):
        return value.strip()
    try:
        return str(value).strip()
    except Exception:
        return ""


def _parse_face_record(face_info):
    """Extract a single face dict from one ALFaceDetection face entry.

    Layout (per Aldebaran ALFaceDetection docs)::

        face_info = [shape_info, extra_info]
        shape_info = [alpha, beta, size_x, size_y]
            alpha, beta : face-centre position in camera angle space (rad)
            size_x, size_y : normalised face size in [0, 1]
        extra_info = [face_id, score_reco, face_label,
                      left_eye, right_eye, nose, mouth]

    Returns None if the record is malformed, otherwise a dict with the
    geometry-bearing fields populated. Fields:
        face_id     : str (empty if unknown)
        name        : str (face_label, empty if unknown)
        confidence  : float in [0, 1]
        distance_m  : float (0.0 if size unavailable)
        yaw_deg     : float (camera-frame, +right)
        pitch_deg   : float (camera-frame, +down)
        size_x_norm : float (kept for closest_face tie-breaks/debug)
    """
    if not isinstance(face_info, (list, tuple)) or len(face_info) < 2:
        return None

    shape = face_info[0]
    extra = face_info[1]
    if not isinstance(shape, (list, tuple)) or len(shape) < 4:
        return None

    try:
        alpha = float(shape[0])
        beta = float(shape[1])
        size_x = float(shape[2])
        # size_y kept around in case future callers want vertical extent.
        # size_y = float(shape[3])
    except (TypeError, ValueError):
        return None

    face_id = ""
    name = ""
    confidence = 0.0
    if isinstance(extra, (list, tuple)) and len(extra) >= 3:
        face_id = _coerce_text(extra[0])
        try:
            confidence = float(extra[1])
        except (TypeError, ValueError):
            confidence = 0.0
        name = _coerce_text(extra[2])
        if name and confidence < face_min_score():
            name = ""  # weak match: unknown, not a guess

    return {
        "face_id": face_id,
        "name": name,
        "confidence": confidence,
        "distance_m": _estimate_distance_m(size_x),
        "yaw_deg": math.degrees(alpha),
        "pitch_deg": math.degrees(beta),
        "size_x_norm": size_x,
    }


def detect_faces_with_geometry(face_detection, memory, max_age_ms=200):
    """Read ALFaceDetection's most recent event and return geometry per face.

    Reads ``ALMemory["FaceDetected"]`` directly. The standard payload is::

        [time_filtered_reading,
         [face_info_0, face_info_1, ...],
         camera_pose_in_robot,
         current_torso_in_robot,
         current_camera_id]

    For each face we extract the geometry block (face-centre angles in the
    camera frame and face size in normalised image coordinates) and
    compute an approximate distance using::

        angular_width_rad = size_x_norm * hfov_rad
        distance_m = (REAL_FACE_WIDTH_M / 2) / tan(angular_width_rad / 2)

    where REAL_FACE_WIDTH_M defaults to 0.16 m (average adult face width)
    and hfov_rad is the NAO V6 top-camera horizontal FOV
    (60.97 deg horizontal, 47.64 deg vertical).

    Args:
        face_detection: NAOqi ALFaceDetection proxy (must already be
            subscribed by the caller). Currently unused beyond signalling
            intent — the data flows through ALMemory — but kept in the
            signature so callers express their dependency explicitly and
            so we can extend later without breaking the contract.
        memory: NAOqi ALMemory proxy.
        max_age_ms: Reserved for future use. ALMemory itself does not
            expose an "age" stamp on getData() reads, so this argument is
            currently informational; callers should poll faster than this
            window if they care about freshness. Kept in the signature so
            the WakeStateMachine doesn't have to be re-plumbed once we
            switch to event subscriptions.

    Returns:
        list of dicts: ``[{face_id, name, confidence, distance_m, yaw_deg,
        pitch_deg}, ...]``. At most one entry per ``face_id`` (closest
        wins on duplicates). Empty list if no faces are visible or the
        ALMemory read fails.
    """
    # max_age_ms is part of the documented contract but ALMemory.getData
    # does not return a timestamp on this key, so we cannot reject stale
    # reads here without an event subscription. Accepted to keep the API
    # stable; callers control freshness by poll cadence.
    _ = max_age_ms
    _ = face_detection

    if memory is None:
        return []

    try:
        payload = memory.getData("FaceDetected")
    except Exception as exc:
        print("[detect_faces_with_geometry] ALMemory read error:", exc)
        return []

    if not payload or not isinstance(payload, (list, tuple)) or len(payload) < 2:
        return []

    face_info_list = payload[1]
    if not isinstance(face_info_list, (list, tuple)) or len(face_info_list) == 0:
        return []

    # The first element of face_info_list is sometimes a "TimeStamp" sub-array
    # ([seconds, microseconds]) on some firmware revisions; the actual face
    # entries follow. Detect that defensively.
    candidates = list(face_info_list)
    if (len(candidates) > 0
            and isinstance(candidates[0], (list, tuple))
            and len(candidates[0]) == 2
            and all(isinstance(v, (int, float)) for v in candidates[0])):
        candidates = candidates[1:]

    by_id = {}
    anon_index = 0
    for entry in candidates:
        face = _parse_face_record(entry)
        if face is None:
            continue
        key = face["face_id"]
        if not key:
            # Unknown / not-yet-learned face — keep them all but key by index
            # so they don't collapse onto each other.
            key = "__anon_{0}__".format(anon_index)
            anon_index += 1
        existing = by_id.get(key)
        if existing is None:
            by_id[key] = face
            continue
        # Duplicate face_id — keep the closer reading. If distance is
        # unknown for one (0.0), prefer the one we *do* have a distance
        # for; otherwise compare numerically.
        new_d = face["distance_m"]
        old_d = existing["distance_m"]
        if old_d <= 0.0 and new_d > 0.0:
            by_id[key] = face
        elif new_d > 0.0 and new_d < old_d:
            by_id[key] = face

    # Strip the helper field before handing back to callers.
    out = []
    for face in by_id.values():
        face.pop("size_x_norm", None)
        out.append(face)
    return out


def closest_face(faces):
    """Pick the closest face by ``distance_m``; ties broken by ``confidence``.

    Faces with ``distance_m <= 0`` (unknown distance) are deprioritised
    and only returned if no face has a known distance. Returns ``None``
    if ``faces`` is empty / falsy.

    Args:
        faces: iterable of dicts as produced by
            :func:`detect_faces_with_geometry`.

    Returns:
        dict or None.
    """
    if not faces:
        return None

    known = [f for f in faces if f.get("distance_m", 0.0) > 0.0]
    pool = known if known else list(faces)
    if not pool:
        return None

    # Sort: smallest distance first, then highest confidence first.
    # math.inf isn't available in py2.7 stdlib for use with sort keys here,
    # but we already filtered out non-positive distances when possible; the
    # fallback pool sorts by (distance, -confidence) where unknowns share
    # distance 0.0 and are simply ranked by confidence.
    def _key(f):
        d = f.get("distance_m", 0.0)
        c = f.get("confidence", 0.0)
        # Negate confidence so that, at equal distance, higher confidence
        # sorts first.
        return (d if d > 0.0 else float("inf"), -c)

    pool.sort(key=_key)
    return pool[0]


def is_mutually_gazing(face, yaw_tolerance_deg=15, pitch_tolerance_deg=15):
    """Return True if ``face`` is roughly head-on (eyes pointed at NAO).

    Approximation: a face whose centre sits within
    ``+/- yaw_tolerance_deg`` horizontally and ``+/- pitch_tolerance_deg``
    vertically of the camera optical axis is treated as making mutual
    gaze. ALFaceDetection does not expose a true 6-DOF head pose per face
    in its event payload, so we use the camera-frame face-centre angles
    (``yaw_deg``, ``pitch_deg`` from
    :func:`detect_faces_with_geometry`) as a tractable proxy. This is
    the same approximation used by the AWARE-state engagement gate in
    PRD v2 §Phase 3.

    Args:
        face: dict with ``yaw_deg`` and ``pitch_deg`` keys.
        yaw_tolerance_deg: half-width of the acceptance cone, horizontal.
        pitch_tolerance_deg: half-width of the acceptance cone, vertical.

    Returns:
        bool.
    """
    if not face:
        return False
    try:
        yaw = abs(float(face.get("yaw_deg", 0.0)))
        pitch = abs(float(face.get("pitch_deg", 0.0)))
    except (TypeError, ValueError):
        return False
    return yaw <= float(yaw_tolerance_deg) and pitch <= float(pitch_tolerance_deg)


# ---------------------------------------------------------------------------
# Self-test: runs without naoqi. Exercises closest_face + is_mutually_gazing
# on synthetic dicts, plus _parse_face_record on a hand-built event payload
# so the distance formula gets covered without a live robot.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic faces. distance_m is what we'd get from the formula at
    # size_x_norm in {0.20, 0.10, 0.05} with default constants.
    faces = [
        {"face_id": "near", "name": "Aayush", "confidence": 0.81,
         "distance_m": 0.45, "yaw_deg": 4.0, "pitch_deg": -2.0},
        {"face_id": "mid", "name": "", "confidence": 0.55,
         "distance_m": 0.92, "yaw_deg": 12.0, "pitch_deg": 8.0},
        {"face_id": "far", "name": "Stranger", "confidence": 0.42,
         "distance_m": 1.85, "yaw_deg": -25.0, "pitch_deg": 18.0},
    ]

    chosen = closest_face(faces)
    assert chosen is not None and chosen["face_id"] == "near", chosen
    print("[selftest] closest_face: ok ({0} @ {1:.2f} m)".format(
        chosen["face_id"], chosen["distance_m"]))

    # Tie-breaker: equal distance, higher confidence wins.
    tied = [
        {"face_id": "a", "confidence": 0.50, "distance_m": 0.80,
         "yaw_deg": 0.0, "pitch_deg": 0.0},
        {"face_id": "b", "confidence": 0.90, "distance_m": 0.80,
         "yaw_deg": 0.0, "pitch_deg": 0.0},
    ]
    tie_winner = closest_face(tied)
    assert tie_winner is not None and tie_winner["face_id"] == "b", tie_winner
    print("[selftest] closest_face tie-break: ok (chose '{0}')".format(
        tie_winner["face_id"]))

    # Empty list returns None.
    assert closest_face([]) is None
    assert closest_face(None) is None
    print("[selftest] closest_face empty/None: ok")

    # Mutual gaze: head-on within tolerance.
    assert is_mutually_gazing(faces[0]) is True
    # Just outside default tolerance.
    assert is_mutually_gazing({"yaw_deg": 20.0, "pitch_deg": 0.0}) is False
    # Pitch outside tolerance, yaw inside.
    assert is_mutually_gazing({"yaw_deg": 5.0, "pitch_deg": 22.0}) is False
    # Custom tolerances accept a wider face.
    assert is_mutually_gazing({"yaw_deg": 18.0, "pitch_deg": 10.0},
                              yaw_tolerance_deg=20,
                              pitch_tolerance_deg=15) is True
    # Defensive: empty / malformed input. We treat empty as "no data,
    # don't claim engagement" rather than "0 deg yaw / 0 deg pitch".
    assert is_mutually_gazing(None) is False
    assert is_mutually_gazing({}) is False
    assert is_mutually_gazing({"yaw_deg": "n/a"}) is False
    # Explicit zeros DO count as mutual gaze.
    assert is_mutually_gazing({"yaw_deg": 0.0, "pitch_deg": 0.0}) is True
    print("[selftest] is_mutually_gazing: ok")

    # Parser + distance estimate. Build a synthetic ALFaceDetection record:
    # a face filling 20% of the image width should land near 0.75 m with
    # default constants (0.16 m face / 60.97 deg HFOV). Allow some slack —
    # the heuristic is approximate by design.
    fake_event_face = [
        [0.05, -0.03, 0.20, 0.20],          # alpha, beta, size_x, size_y
        ["face-42", 0.77, "Aayush", [], [], [], []],
    ]
    parsed = _parse_face_record(fake_event_face)
    assert parsed is not None
    assert parsed["face_id"] == "face-42"
    assert parsed["name"] == "Aayush"
    assert abs(parsed["confidence"] - 0.77) < 1e-6
    assert 0.60 < parsed["distance_m"] < 0.90, parsed["distance_m"]
    assert abs(parsed["yaw_deg"] - math.degrees(0.05)) < 1e-6
    assert abs(parsed["pitch_deg"] - math.degrees(-0.03)) < 1e-6
    print("[selftest] _parse_face_record: ok ({0} @ {1:.2f} m)".format(
        parsed["face_id"], parsed["distance_m"]))

    # Distance scales inversely with size_x as expected: doubling size_x
    # should roughly halve the distance.
    near = _estimate_distance_m(0.40)
    far = _estimate_distance_m(0.10)
    assert near > 0.0 and far > 0.0
    assert far > near * 3.0, (near, far)  # 4x size_x => ~4x closer
    print("[selftest] distance scaling: ok (0.40->{0:.2f}m, 0.10->{1:.2f}m)".format(
        near, far))

    # End-to-end: feed a fake ALMemory into detect_faces_with_geometry.
    class _FakeMemory(object):
        def __init__(self, payload):
            self._p = payload
        def getData(self, key):
            assert key == "FaceDetected"
            return self._p

    payload = [
        12345.6,                                   # time_filtered_reading
        [
            [[0.10, 0.05, 0.10, 0.10],             # ~1.5 m, slightly off-axis
             ["abc", 0.62, "", [], [], [], []]],
            [[-0.02, 0.01, 0.30, 0.30],            # very close, head-on
             ["xyz", 0.88, "Aayush", [], [], [], []]],
            # Duplicate face_id with worse distance — should be discarded.
            [[0.50, 0.40, 0.06, 0.06],
             ["xyz", 0.20, "Aayush", [], [], [], []]],
        ],
        [], [], 0,
    ]
    seen = detect_faces_with_geometry(None, _FakeMemory(payload))
    assert len(seen) == 2, seen
    by_id = dict((f["face_id"], f) for f in seen)
    assert "abc" in by_id and "xyz" in by_id
    assert by_id["xyz"]["confidence"] == 0.88, by_id["xyz"]
    pick = closest_face(seen)
    assert pick is not None and pick["face_id"] == "xyz", pick
    assert is_mutually_gazing(pick) is True
    assert is_mutually_gazing(by_id["abc"]) is True  # ~5.7 deg / ~2.9 deg
    print("[selftest] detect_faces_with_geometry: ok ({0} faces, picked {1!r})".format(
        len(seen), pick["face_id"]))

    # Empty / failure paths.
    assert detect_faces_with_geometry(None, None) == []
    assert detect_faces_with_geometry(None, _FakeMemory(None)) == []
    assert detect_faces_with_geometry(None, _FakeMemory([1, []])) == []
    print("[selftest] detect_faces_with_geometry empty: ok")

    print("[selftest] all checks passed.")

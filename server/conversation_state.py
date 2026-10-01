"""State that has to outlive a single agent turn.

The agent ``ctx`` dict is rebuilt on every turn (``_legacy_helpers``), so
anything an agent stored in it -- the CBT step, which coach was running --
was gone by the next sentence, and a five-step thought record could never
get past step 1. This module keeps one small dict per conversation, keyed
the same way as chat history (``session.session_key_for``), and hands the
*same* dict to every turn as ``ctx["conv"]``.

Kept in process memory on purpose: it is working state for one
conversation, not a record. A restart starts clean, and an entry nobody
has touched for ``CONV_STATE_IDLE_S`` is dropped, so a new visitor never
inherits the last one's place in an exercise.

Well-known keys (anything else is allowed):

* ``lane`` / ``lane_ts`` -- the support agent that answered last and when
  (see ``set_lane`` / ``active_lane``).
* ``cbt_step`` -- the thought-record step the CBT coach is on.
* ``crisis_followup`` -- set after a crisis reply; the next turn checks in.
"""
from __future__ import annotations

import os
import threading
import time

from server import session

# How long a quiet conversation keeps its state. Matches the anonymous
# history epoch so the two reset together.
CONV_STATE_IDLE_S = float(os.environ.get(
    "CONV_STATE_IDLE_S", str(session.GUEST_IDLE_RESET_S)))

# How long the therapy lane holds after the last support reply.
THERAPY_LANE_TTL_S = float(os.environ.get("THERAPY_LANE_TTL_S", "600"))

# Agent names that make up the support ("therapy") lane.
SUPPORT_AGENTS = frozenset({
    "therapist", "cbt_coach", "grounding_coach", "mi_coach",
})

_lock = threading.Lock()
_states: dict[str, dict] = {}
_seen: dict[str, float] = {}


def _prune(now: float) -> None:
    for key in [k for k, t in _seen.items() if now - t > CONV_STATE_IDLE_S]:
        _states.pop(key, None)
        _seen.pop(key, None)


def state_for(username: str, *, now: float | None = None) -> dict:
    """The persistent state dict for ``username``'s current conversation."""
    stamp = time.time() if now is None else now
    key = session.session_key_for(username, now=stamp)
    with _lock:
        _prune(stamp)
        _seen[key] = stamp
        return _states.setdefault(key, {})


def clear(username: str) -> None:
    """Forget the conversation state for ``username`` (end of session)."""
    key = session.session_key_for(username)
    with _lock:
        _states.pop(key, None)
        _seen.pop(key, None)


def set_lane(state: dict, agent_name: str | None, *,
             now: float | None = None) -> None:
    """Record which agent answered. Support agents open/extend the lane."""
    if agent_name in SUPPORT_AGENTS:
        state["lane"] = agent_name
        state["lane_ts"] = time.time() if now is None else now


def leave_lane(state: dict) -> None:
    state.pop("lane", None)
    state.pop("lane_ts", None)


def active_lane(state: dict, *, now: float | None = None) -> str | None:
    """The support agent to resume with, or None when the lane has lapsed."""
    lane = state.get("lane")
    if not lane:
        return None
    stamp = time.time() if now is None else now
    if stamp - float(state.get("lane_ts") or 0.0) > THERAPY_LANE_TTL_S:
        leave_lane(state)
        return None
    return lane


def _reset_for_tests() -> None:
    with _lock:
        _states.clear()
        _seen.clear()

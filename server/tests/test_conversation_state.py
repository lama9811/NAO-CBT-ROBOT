"""State that outlives one agent turn, and per-visit therapy ownership."""
from server import conversation_state as cs
from server import session


def setup_function(_fn):
    cs._reset_for_tests()
    session.retire_anonymous_epoch()


def test_same_dict_across_turns_of_one_conversation():
    a = cs.state_for("Alice", now=1000.0)
    a["cbt_step"] = "3"
    assert cs.state_for("Alice", now=1010.0)["cbt_step"] == "3"


def test_named_users_do_not_share_state():
    cs.state_for("Alice", now=1000.0)["cbt_step"] = "4"
    assert "cbt_step" not in cs.state_for("Bob", now=1000.0)


def test_state_dropped_after_idle():
    cs.state_for("Alice", now=1000.0)["cbt_step"] = "4"
    later = 1000.0 + cs.CONV_STATE_IDLE_S + 1
    assert "cbt_step" not in cs.state_for("Alice", now=later)


def test_next_anonymous_visitor_starts_clean():
    cs.state_for("guest", now=1000.0)["cbt_step"] = "2"
    later = 1000.0 + session.GUEST_IDLE_RESET_S + 1
    assert "cbt_step" not in cs.state_for("guest", now=later)


def test_lane_opens_only_for_support_agents_and_lapses():
    st = {}
    cs.set_lane(st, "pure_chat", now=0.0)
    assert cs.active_lane(st, now=1.0) is None
    cs.set_lane(st, "cbt_coach", now=0.0)
    assert cs.active_lane(st, now=10.0) == "cbt_coach"
    assert cs.active_lane(st, now=cs.THERAPY_LANE_TTL_S + 1) is None
    assert "lane" not in st


def test_therapy_owner_named_vs_anonymous():
    assert session.therapy_owner("Mingma") == "Mingma"
    owner = session.therapy_owner("guest")
    assert owner.startswith("guest:")
    assert owner != "guest"

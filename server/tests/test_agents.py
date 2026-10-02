from server.agents import pick_initial_agent


def test_hint_chat_picks_chat_agent():
    assert pick_initial_agent("alice", "chat").name == "chat"


def test_hint_morgan_picks_chatbot():
    assert pick_initial_agent("alice", "morgan").name == "chatbot"


def test_hint_therapy_picks_therapist():
    assert pick_initial_agent("alice", "therapy").name == "therapist"


def test_hint_skills_picks_skills():
    assert pick_initial_agent("alice", "skills").name == "skills"


def test_no_hint_returns_router():
    assert pick_initial_agent("alice", None).name == "router"


def test_no_hint_general_chat_uses_fast_chat():
    agent = pick_initial_agent(
        "alice", None,
        "What do you think about parallel universes?",
    )
    assert agent.name == "chat"


def test_no_hint_emotional_turn_goes_straight_to_therapist():
    # Python already made the router's call here; the extra hop only
    # added a second sequential model call (see test_therapy_lane.py).
    agent = pick_initial_agent(
        "alice", None,
        "I'm anxious and overwhelmed about my exam tomorrow.",
    )
    assert agent.name == "therapist"


def test_no_hint_emotional_turn_with_school_words_still_uses_router():
    agent = pick_initial_agent(
        "alice", None,
        "I'm so stressed about my classes.",
    )
    assert agent.name == "router"


def test_no_hint_morgan_turn_still_uses_router():
    agent = pick_initial_agent(
        "alice", None,
        "What classes does Morgan offer for computer science?",
    )
    assert agent.name == "router"


def test_therapist_injects_recaps(monkeypatch):
    from server.agents import therapist as t
    monkeypatch.setattr(t.session, "load_recent_recaps", lambda u, n=3: ["past talk"])
    a = t.build_therapist_agent("alice")
    # Instructions are now a callable so memory updates land per turn.
    rendered = a.instructions(None, a) if callable(a.instructions) else a.instructions
    assert "past talk" in rendered

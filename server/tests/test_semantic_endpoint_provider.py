# -*- coding: utf-8 -*-
"""`semantic_endpoint` must stay provider-agnostic.

It runs on EVERY turn, and it used to call `from openai import OpenAI`
directly. That silently pinned the endpointing check to OpenAI on a stack
whose agents are otherwise all Claude -- and no value of
`SEMANTIC_ENDPOINT_MODEL` could move it, because the module never consulted
the dispatcher. These tests fail if anyone reintroduces a direct client.

`pytest-asyncio` is not installed here, so coroutines are driven with
`asyncio.run()` rather than an async test function (an
`@pytest.mark.asyncio` test would silently skip).
"""
import asyncio
import time

from server import semantic_endpoint as se


class _Recorder:
    """Stands in for `llm_compat.chat` and records how it was called."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def __call__(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return self.reply


def _fresh_cache():
    # The module caches verdicts by transcript; unique text per assertion
    # keeps one test from answering another's question.
    return "probe-%.9f" % time.time()


class TestGoesThroughLlmCompat:
    def test_no_direct_provider_client_remains(self):
        # The failure mode this whole module guards against.
        assert not hasattr(se, "_get_client"), (
            "_get_client is back -- semantic_endpoint is pinned to one provider"
        )

    def test_call_routes_through_llm_compat(self, monkeypatch):
        rec = _Recorder("yes")
        monkeypatch.setattr(se.llm_compat, "chat", rec)
        assert se._call_llm("what time is it") is True
        assert len(rec.calls) == 1

    def test_configured_model_is_the_one_used(self, monkeypatch):
        rec = _Recorder("yes")
        monkeypatch.setattr(se.llm_compat, "chat", rec)
        monkeypatch.setattr(se, "_MODEL", "claude-haiku-4-5")
        se._call_llm("anything")
        assert rec.calls[0]["model"] == "claude-haiku-4-5"

    def test_an_openai_id_still_works(self, monkeypatch):
        # Provider-agnostic means both directions, not "Claude instead".
        rec = _Recorder("no")
        monkeypatch.setattr(se.llm_compat, "chat", rec)
        monkeypatch.setattr(se, "_MODEL", "gpt-4o-mini")
        assert se._call_llm("I need") is False
        assert rec.calls[0]["model"] == "gpt-4o-mini"

    def test_system_and_user_roles_are_sent(self, monkeypatch):
        rec = _Recorder("yes")
        monkeypatch.setattr(se.llm_compat, "chat", rec)
        se._call_llm("tell me a joke")
        roles = [m["role"] for m in rec.calls[0]["messages"]]
        assert roles == ["system", "user"]
        assert rec.calls[0]["messages"][1]["content"] == "tell me a joke"

    def test_token_budget_leaves_room_for_a_word(self, monkeypatch):
        # max_tokens=1 is enough for one OpenAI BPE token but Anthropic can
        # spend it on leading whitespace and return nothing usable.
        rec = _Recorder("yes")
        monkeypatch.setattr(se.llm_compat, "chat", rec)
        se._call_llm("anything")
        assert rec.calls[0]["max_tokens"] > 1


class TestReplyParsing:
    def _verdict(self, monkeypatch, reply):
        monkeypatch.setattr(se.llm_compat, "chat", _Recorder(reply))
        return se._call_llm(_fresh_cache())

    def test_plain_words(self, monkeypatch):
        assert self._verdict(monkeypatch, "yes") is True
        assert self._verdict(monkeypatch, "no") is False

    def test_case_and_punctuation(self, monkeypatch):
        for reply in ("Yes", "YES", "Yes.", " yes \n", "Yes, complete"):
            assert self._verdict(monkeypatch, reply) is True, reply
        for reply in ("No", "NO", "no.", " no \n"):
            assert self._verdict(monkeypatch, reply) is False, reply

    def test_empty_reply_is_not_complete(self, monkeypatch):
        # Must not raise on an empty or whitespace-only completion.
        assert self._verdict(monkeypatch, "") is False
        assert self._verdict(monkeypatch, "   ") is False


class TestFailsOpen:
    """A dead provider must never strand a turn.

    `is_complete_thought` returning True means "treat it as finished and
    answer now". On an outage that is the safe direction: Nao replies to a
    possibly-partial sentence instead of waiting forever in silence.
    """

    def test_provider_error_returns_complete(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("provider down")
        monkeypatch.setattr(se.llm_compat, "chat", boom)
        assert asyncio.run(se.is_complete_thought(_fresh_cache())) is True

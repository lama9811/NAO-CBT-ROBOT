"""OpenAI tracing must be off unless asked for.

Until 2026-09-30 OPENAI_AGENTS_TRACE was read by nothing and the Agents SDK
uploaded every turn's trace to OpenAI, even though the brain runs on Claude.
"""
import importlib


def test_tracing_disabled_by_default(monkeypatch):
    monkeypatch.delenv("OPENAI_AGENTS_TRACE", raising=False)
    from server import config
    importlib.reload(config)
    assert config.OPENAI_AGENTS_TRACE is False
    from agents import set_tracing_disabled
    from agents.tracing import get_trace_provider
    set_tracing_disabled(not config.OPENAI_AGENTS_TRACE)
    assert get_trace_provider()._disabled is True


def test_app_applies_the_switch_at_import():
    from agents.tracing import get_trace_provider
    from server import app_ws  # noqa: F401
    assert get_trace_provider()._disabled is True

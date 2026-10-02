"""Session summaries go through llm_compat, not a direct OpenAI client."""
import pathlib
import time

from server import memory


def test_memory_has_no_direct_openai_client():
    src = pathlib.Path(memory.__file__).read_text()
    assert "from openai import" not in src and "import openai" not in src
    assert not hasattr(memory, "_client")


def test_summary_uses_llm_compat(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "_DB_PATH", str(tmp_path / "m.db"))
    calls = []

    def fake_chat(**kw):
        calls.append(kw)
        return " Talked about exams. "

    monkeypatch.setattr(memory.llm_compat, "chat", fake_chat)
    sid = memory.start_session("ann")
    memory.summarize_session_async(sid, ["user: exams are hard"])
    for _ in range(100):
        if memory.recent_sessions("ann")[0]["summary"]:
            break
        time.sleep(0.02)
    assert memory.recent_sessions("ann")[0]["summary"] == "Talked about exams."
    assert calls[0]["model"] == memory._SUMMARY_MODEL

"""Markdown must not reach the speech synthesizer.

The CS Navigator backend formats its answers for a web UI, so NAO was handed
raw Markdown and voiced the punctuation. The examples below are its real
output, captured from the live Cloud Run service.
"""
from __future__ import annotations

import pytest

from server.tts_text import to_speakable


def test_bold_is_spoken_as_words():
    """Real CS Navigator answer — the prerequisite lookup."""
    got = to_speakable(
        'The prerequisite for **COSC 220 - Data Structures and Algorithms** is '
        '**COSC 112 - Introduction to Computer Science II** with a grade of "C" or higher.'
    )

    assert "*" not in got
    assert got.startswith("The prerequisite for COSC 220 - Data Structures")
    assert "COSC 112 - Introduction to Computer Science II" in got


def test_bulleted_faculty_list_gets_sentence_breaks():
    """Real CS Navigator answer — the faculty lookup, which is a list."""
    got = to_speakable(
        "The Computer Science Department has a dedicated faculty. "
        "Here are some of the professors:\n\n"
        "*   **Dr. Amjad Ali** - Professor (Cybersecurity, AI)\n"
        "*   **Dr. Radhouane Chouchane** - Associate Professor\n"
    )

    assert "*" not in got
    assert "Dr. Amjad Ali - Professor (Cybersecurity, AI)." in got
    assert "Dr. Radhouane Chouchane - Associate Professor." in got


def test_headings_quotes_and_rules_are_dropped():
    got = to_speakable("## Advising\n\n> Note this\n\n---\n\nSee an advisor.")

    assert "#" not in got and ">" not in got and "---" not in got
    assert "Advising" in got and "See an advisor." in got


def test_links_keep_the_label_and_drop_the_url():
    got = to_speakable("See [the catalog](https://example.edu/catalog/cosc) for details.")

    assert got == "See the catalog for details."


def test_code_spans_are_spoken_plainly():
    assert to_speakable("Run `git pull` first.") == "Run git pull first."


@pytest.mark.parametrize(
    "text",
    [
        "The prerequisite is COSC 112 with a grade of C or higher.",
        "Multiply 2 * 3 to get six.",
        "The flag is snake_case_name in the config.",
    ],
)
def test_plain_text_is_returned_unchanged(text):
    """Arithmetic and identifiers must survive — they are not emphasis."""
    assert to_speakable(text) == text


def test_empty_and_none_are_safe():
    assert to_speakable("") == ""
    assert to_speakable(None) is None


def test_synth_strips_markdown_before_it_reaches_a_provider(monkeypatch):
    """The audio path is what matters — assert on what TTS actually receives."""
    from server import app_ws, openai_tts

    seen: list[str] = []
    monkeypatch.setattr(app_ws, "_eleven", None)
    monkeypatch.setattr(
        openai_tts, "synthesize", lambda text: seen.append(text) or b"mp3"
    )

    app_ws._synth_for("mia", "The prerequisite is **COSC 112**.")

    assert seen == ["The prerequisite is COSC 112."]

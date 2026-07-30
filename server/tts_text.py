"""Turn written formatting into something a speech synthesizer can voice.

The CS Navigator backend answers in Markdown — it was built for a web UI:

    The prerequisite for **COSC 220 - Data Structures** is **COSC 112**...
    *   **Dr. Amjad Ali** - Professor (Cybersecurity, AI)

Nothing stripped that before the text reached TTS, so NAO would voice the
asterisks and run a bulleted list together as one breathless sentence. Agents
also emit Markdown on their own from time to time, so this runs on everything
headed for synthesis rather than only on the RAG path.

`to_speakable` is deliberately conservative: it removes formatting and gives
list items sentence boundaries so the voice pauses between them. It does not
rewrite words, reorder anything, or summarize — the text NAO says stays the
text the agent wrote.
"""
from __future__ import annotations

import re

# ``**bold**`` / ``__bold__`` → the words inside.
_BOLD = re.compile(r"(\*\*|__)(.+?)\1", re.DOTALL)
# ``*italic*`` / ``_italic_`` → the words inside. Requires a non-space next to
# the marker so ``2 * 3`` and snake_case identifiers survive untouched.
_ITALIC = re.compile(r"(?<!\w)([*_])(?=\S)(.+?)(?<=\S)\1(?!\w)", re.DOTALL)
# `code` / ```fenced``` → the contents, spoken plainly.
_CODE_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.DOTALL)
_CODE_INLINE = re.compile(r"`([^`]+)`")
# [label](https://…) → label. Nobody wants a URL read aloud.
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# Leading #, >, and horizontal rules.
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_QUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_RULE = re.compile(r"^\s{0,3}([-*_])\s*(\1\s*){2,}$", re.MULTILINE)
# Bullet markers at the start of a line: "* ", "- ", "+ ", "• ".
_BULLET = re.compile(r"^\s*[*+\-•]\s+", re.MULTILINE)
# Leftover emphasis characters from unbalanced markup. Only strip runs that
# touch a word — an asterisk standing alone between spaces is arithmetic
# ("2 * 3"), not formatting, and must be left for the voice to say.
_STRAY = re.compile(r"\*+(?=\w)|(?<=\w)\*+|_{2,}(?=\w)|(?<=\w)_{2,}")
_SPACES = re.compile(r"[ \t]{2,}")
_BLANKS = re.compile(r"\n{3,}")

# A line that already ends like a sentence needs no help; anything else gets a
# period so the synthesizer pauses instead of running items together.
_ENDS_SENTENCE = re.compile(r"[.!?:;,]$")


def to_speakable(text: str) -> str:
    """Return `text` with Markdown formatting removed, safe to hand to TTS.

    Returns the input unchanged when it carries no formatting, so the common
    case costs one regex sweep and nothing else.
    """
    if not text:
        return text

    out = _CODE_FENCE.sub(r"\1", text)
    out = _CODE_INLINE.sub(r"\1", out)
    out = _LINK.sub(r"\1", out)
    out = _RULE.sub("", out)
    out = _HEADING.sub("", out)
    out = _QUOTE.sub("", out)
    out = _BOLD.sub(r"\2", out)
    out = _ITALIC.sub(r"\2", out)

    # Bullets become their own sentences so the voice breathes between them.
    lines = []
    for raw_line in out.split("\n"):
        was_bullet = bool(_BULLET.match(raw_line))
        line = _BULLET.sub("", raw_line).strip()
        if was_bullet and line and not _ENDS_SENTENCE.search(line):
            line += "."
        lines.append(line)
    out = "\n".join(lines)

    out = _STRAY.sub("", out)
    out = _SPACES.sub(" ", out)
    out = _BLANKS.sub("\n\n", out)
    return out.strip()

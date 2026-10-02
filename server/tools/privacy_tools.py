"""``forget_me`` -- the agent-side path to "delete what you know about me".

The voice fast-path in ``app_ws`` (``privacy.handle_forget_turn``) catches
the plain phrasings with a spoken yes/no confirmation. This tool covers the
rest ("can you wipe our chats, I don't want them kept"): the agent must
confirm with the student first, then call it with ``confirmed=True``.
"""
from __future__ import annotations

from agents import RunContextWrapper, function_tool

from server import privacy


def _forget_me_impl(ctx: RunContextWrapper, confirmed: bool) -> str:
    if not confirmed:
        return ("not_confirmed: ask the student to confirm they want "
                "everything deleted, then call again with confirmed=true.")
    c = getattr(ctx, "context", None) or {}
    username = c.get("username") or ""
    # Deleted once this run ends (_legacy_helpers): deleting now would be
    # undone, because the SDK saves this turn to the history afterwards.
    privacy.request_forget_after_run(username)
    return ("deleted: their chat history, moods, thought records, homework "
            "and recaps are being erased as this reply ends. Tell them "
            "plainly that it is done, and that NAO may still recognise "
            "their face; a staff member can remove that if they want.")


@function_tool
def forget_me(ctx: RunContextWrapper, confirmed: bool) -> str:
    """Delete everything stored about the current student (chat history,
    moods, thought records, homework, recaps). Only call with
    confirmed=true after the student has clearly said yes to deleting it."""
    return _forget_me_impl(ctx, confirmed)

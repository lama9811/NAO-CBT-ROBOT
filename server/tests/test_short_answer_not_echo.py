"""Short human answers must not be swallowed by the self-echo guard.

`_is_self_echo` compared the normalised transcript to the normalised last
reply with a raw substring test (`nt in nl or nl in nt`). Normalisation strips
punctuation but keeps the text as one string, so a one-word answer matched
whenever its letters appeared *anywhere* inside the previous reply:

    LAST_REPLY = "You're at work now? How's that going..."
    transcript = "No."      -> "no" is inside "now" -> rejected as self_echo

Observed live: the user answered "No." and the turn was dropped, while the
same word was accepted one turn later when the previous reply happened not to
contain those letters. From the user's side NAO simply ignores them at random.

The guard must still catch genuine echo -- NAO hearing its own reply through
its speaker -- so these tests pin both directions.
"""
from __future__ import annotations

import importlib


L = importlib.import_module("server._legacy_helpers")
A = importlib.import_module("server.app_ws")


# ───────── false positives: real short answers ─────────

def test_no_is_not_an_echo_of_a_reply_containing_now():
    L.LAST_REPLY["u"] = "You're at work now? How's that going with classes starting?"
    assert L._is_self_echo("u", "No.") is False


def test_common_short_answers_survive_a_long_reply():
    L.LAST_REPLY["u"] = (
        "I know that's a lot to take on, and nothing about it is simple. "
        "Do you want to talk through the okay parts first?"
    )
    for answer in ("No.", "Ok.", "Okay.", "Yes.", "Sure.", "Hi."):
        assert L._is_self_echo("u", answer) is False, answer


def test_substring_guard_ignores_single_word_answers():
    A._reset_reply_chunks("u", "Not right now, but I can help you know where to start.")
    assert A._is_substring_or_sentence_echo("u", "No.") is False


# ───────── true positives: genuine echo must still be caught ─────────

def test_exact_echo_of_a_short_reply_is_still_rejected():
    L.LAST_REPLY["u"] = "Okay."
    assert L._is_self_echo("u", "Okay.") is True


def test_partial_echo_of_a_long_reply_is_still_rejected():
    L.LAST_REPLY["u"] = "Alright, Mia. Good luck with work and everything coming up."
    assert L._is_self_echo("u", "Good luck with work") is True


def test_two_word_echo_still_caught_by_token_overlap():
    L.LAST_REPLY["u"] = "Take care."
    assert L._is_self_echo("u", "Take care") is True


def test_substring_guard_still_catches_a_real_sentence_echo():
    A._reset_reply_chunks("u", "Cool. Take care, Mia. Talk to you soon.")
    assert A._is_substring_or_sentence_echo("u", "Take care, Mia.") is True

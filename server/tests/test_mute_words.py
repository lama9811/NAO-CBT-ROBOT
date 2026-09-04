# -*- coding: utf-8 -*-
"""Mute / unmute command matching.

Pure text logic, so these run without the server, an event loop, or any
API key. `pytest-asyncio` is not installed in this repo, so nothing here
is a coroutine on purpose.
"""
from server import mute_words


class TestMute:
    def test_bare_word(self):
        assert mute_words.classify("mute") == "mute"

    def test_addressed(self):
        assert mute_words.classify("Nao mute") == "mute"
        assert mute_words.classify("nao, mute") == "mute"
        assert mute_words.classify("hey nao mute") == "mute"

    def test_stt_mishears_nao(self):
        # Deepgram routinely returns "now"/"no" for "NAO"; rejecting those
        # would make the feature look broken on the robot.
        assert mute_words.classify("now mute") == "mute"
        assert mute_words.classify("no mute") == "mute"

    def test_natural_phrasings(self):
        assert mute_words.classify("be quiet") == "mute"
        assert mute_words.classify("stop talking") == "mute"

    def test_case_and_punctuation(self):
        assert mute_words.classify("  MUTE!!  ") == "mute"
        assert mute_words.classify("Nao... Mute.") == "mute"


class TestUnmute:
    def test_unmute_wins_over_mute(self):
        # "mute" is a substring of "unmute" -- the ordering bug this guards.
        assert mute_words.classify("unmute") == "unmute"
        assert mute_words.classify("nao unmute") == "unmute"

    def test_variants(self):
        assert mute_words.classify("un mute") == "unmute"
        assert mute_words.classify("you can talk") == "unmute"
        assert mute_words.classify("start talking") == "unmute"


class TestNoFalsePositives:
    def test_word_inside_another_word(self):
        assert mute_words.classify("commuter") is None
        assert mute_words.classify("commute to campus") is None

    def test_sentence_merely_mentioning_it(self):
        # The length cap is what stops these. Silencing the robot because
        # someone said "mute" in passing reads as a crash.
        assert mute_words.classify("I had to mute my laptop in class") is None
        assert mute_words.classify(
            "my professor told me to be quiet during the exam") is None

    def test_ordinary_speech(self):
        assert mute_words.classify("how are you today") is None
        assert mute_words.classify("what classes should I take") is None
        assert mute_words.classify("") is None
        assert mute_words.classify(None) is None


class TestSelfTrigger:
    def test_detects_own_speech(self):
        # NAO hears itself while the mic is open during TTS.
        assert mute_words.is_self_trigger("mute", "Okay, I'll mute myself.")
        assert mute_words.is_self_trigger("unmute", "I am unmuted now.")

    def test_ignores_unrelated_reply(self):
        assert not mute_words.is_self_trigger(
            "mute", "COSC 220 requires COSC 112 with a C or higher.")

    def test_empty(self):
        assert not mute_words.is_self_trigger("mute", "")
        assert not mute_words.is_self_trigger("", "mute")


class TestEchoStrippedCommands:
    """The failure seen the first time this ran on the real robot.

    The mic stays open during TTS, so the user's command arrives glued to
    the tail of Nao's own echoed reply as one long transcript. `classify`
    caps a command at four words, so it returned None and the command was
    silently lost.
    """

    # Verbatim from logs/server.log, 2026-09-04 16:29:22 -- the user said
    # "Nao, mute" and STT rendered it "Now on mute" on the end of Nao's
    # previous reply.
    REAL_TRANSCRIPT = (
        "Well. Thanks for asking. How are things going with you? Now on mute."
    )
    REAL_REPLY = (
        "I'm doing well, thanks for asking. How are things going with you?"
    )

    def test_the_real_robot_failure_now_matches(self):
        assert mute_words.classify(self.REAL_TRANSCRIPT) is None
        assert mute_words.classify_with_echo(
            self.REAL_TRANSCRIPT, self.REAL_REPLY) == "mute"

    def test_strip_leaves_only_the_user_words(self):
        assert mute_words.strip_leading_echo(
            self.REAL_TRANSCRIPT, self.REAL_REPLY) == "now on mute"

    def test_unmute_survives_the_same_treatment(self):
        assert mute_words.classify_with_echo(
            "How are things going with you? Unmute.",
            "How are things going with you?") == "unmute"

    def test_no_echo_source_is_a_passthrough(self):
        assert mute_words.classify_with_echo("Nao mute", "") == "mute"
        assert mute_words.classify_with_echo("tell me about classes", "") is None

    def test_short_incidental_overlap_does_not_strip(self):
        # Only two leading tokens overlap, under min_stripped -- the
        # transcript must come back untouched so a normal sentence cannot be
        # whittled down into a command.
        t = "how are you going to mute the whole department"
        assert mute_words.strip_leading_echo(t, "how are we doing") == t
        assert mute_words.classify_with_echo(t, "how are we doing") is None

    def test_pure_self_echo_yields_no_command(self):
        # Whole transcript is Nao's own voice: nothing is left after the
        # strip, so there is no command to act on.
        reply = "Sure, I can put myself on mute whenever you like"
        assert mute_words.classify_with_echo(reply, reply) is None

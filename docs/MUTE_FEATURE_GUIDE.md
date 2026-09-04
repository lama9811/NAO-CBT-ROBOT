# Building the spoken mute feature

A build-it-yourself walkthrough. Goal: say **"Nao, mute"** and the robot stops
speaking; say **"unmute"** and it starts again.

Written after building it and watching it fail on the real robot, so the
mistakes are included on purpose. Skipping to the finished design hides the
part that actually matters.

---

## Step 0 — Understand why this is hard

The naive version is two lines: match "mute", stop sending audio. It does not
work, for one reason.

**NAO is deaf while it speaks.** `nao/ws_client.py` closes the mic gate the
moment TTS starts:

```python
self.audio_streamer.gate(True)   # calls unsubscribe() on the audio device
```

That exists to stop NAO transcribing its own voice and answering itself. But
mute is only useful *while NAO is talking* — which is exactly when it cannot
hear you.

So the real problem is: **let NAO hear "mute" while speaking, without
re-opening the self-conversation loop the gate was built to prevent.**

Everything below follows from that.

---

## Step 1 — Write the matcher first, with no I/O

Put pure logic in its own module. It unit-tests in milliseconds, with no
WebSocket, no event loop, and no robot.

`server/mute_words.py`:

```python
_MUTE_PHRASES   = ("mute", "be quiet", "stop talking", "quiet please")
_UNMUTE_PHRASES = ("unmute", "un mute", "you can talk", "speak again", ...)

_MAX_COMMAND_WORDS = 4

def classify(transcript: str) -> str | None:
    norm = _strip_address(_normalize(transcript))
    if _matches(norm, _UNMUTE_PHRASES):   # unmute FIRST -- "mute" is a
        return "unmute"                   # substring of "unmute"
    if _matches(norm, _MUTE_PHRASES):
        return "mute"
    return None
```

Three decisions worth copying:

- **Whole-word matching.** Substring matching fires "mute" on "commuter".
- **A four-word cap.** Otherwise *"I had to mute my laptop"* silences the robot.
- **Address-term stripping that accepts `now` / `no` / `neo`.** STT does not
  return "NAO". It returns those. Rejecting them makes the feature look broken.

Asymmetric risk drives all of it: a false positive silences NAO mid-sentence
for a word you never said, which reads as a crash. A false negative just means
you repeat yourself. **Be strict.**

---

## Step 2 — Keep the mic open during TTS

`nao/ws_client.py`. Remember: robot code is **Python 2.7** — no f-strings, no
type hints.

```python
self._mute_listen_during_tts = (
    os.environ.get("MUTE_LISTEN_DURING_TTS", "1") == "1"
)

def _close_mic_gate(self):
    if self._mute_listen_during_tts:
        self.log.debug("mic_gate_left_open_for_mute_listen")
        return                      # <-- do NOT gate
    self.audio_streamer.gate(True)
```

Put it behind an env var. You want to A/B this against the old behaviour
without a code change.

---

## Step 3 — Give that audio a side channel, not the turn pipeline

This is the step that keeps Step 2 from being a disaster.

Audio arriving during TTS is still **dropped from the turn pipeline** — it is
mostly NAO's own voice. Do not weaken that. Instead *copy* it to a path that
runs only the keyword matcher.

`server/app_ws.py`, in `_ingest_frame`:

```python
if in_tts_window or turn_running:
    if MUTE_LISTEN_ENABLED:
        _pcm = base64.b64decode(frame.get("data") or "")
        if _pcm:
            await _feed_mute_listener(ws, sess, _pcm)
    return True          # still dropped from the turn pipeline
```

`_feed_mute_listener` buffers ~1.6 s (52000 bytes at 16 kHz mono s16), runs
STT, and classifies. Nothing from that buffer can become a turn, reach an
agent, or be written to memory — so the self-conversation loop still cannot
form.

Cap the buffer (`_MUTE_BUF_MAX_BYTES`) and allow one STT call in flight per
session, or a long reply grows it without bound.

---

## Step 4 — Stop NAO muting itself

Mic open means NAO hears its own voice. If the sentence it is speaking right
now contains "mute" ("I'll mute myself now"), that transcript is NAO, not you.

```python
def is_self_trigger(command: str, spoken_text: str) -> bool:
    ...
```

Track what NAO is saying (`sess.speaking_text`, set in `_send_audio_chunk`)
and suppress the trigger when it matches. Err toward suppressing: a false
positive here ignores one command; a false negative lets the robot silence
itself at random.

---

## Step 5 — One choke point for all outbound speech

Do not sprinkle `if muted` around. Funnel **every** speech path through one
function, or mute gets defeated by whichever reply path happens to run —
crisis, motion ack, onboarding, greeting, camera announcement, streamed reply.

```python
async def _send_audio_chunk(ws, sess, frame, force=False) -> bool:
    if getattr(sess, "muted", False) and not force:
        logger.info("tts_suppressed_muted", ...)
        return False
    sess.speaking_text = str(frame.get("text") or "")
    await _send_json(ws, frame)
    return True
```

Two deliberate choices:

- **Audio only is withheld.** The turn still runs — NAO listens, thinks, and
  records memory — so "unmute" lands mid-conversation instead of after a reset.
- **`force=True` exists for the 988 crisis reply only.** A spoken "mute" must
  not be able to silence a hotline referral.

Verify you got them all:

```bash
grep -n "_audio_chunk_frame" server/app_ws.py | grep -v "_send_audio_chunk"
```

Anything listed is a hole. (The camera announcement was one.)

---

## Step 6 — Cut audio the robot has already queued

Server-side suppression is not enough. The reply was synthesized before the
command arrived, so several MP3 chunks are already in the robot's player.

Server sends a control frame:

```python
await _send_json(ws, _control_frame("mute", muted=True))
```

Robot dispatches it in `_on_control`:

```python
elif sub == "mute":
    self._on_mute(data)
```

And `_on_mute` stops **three** things — miss any one and NAO keeps making
noise:

```python
self.tts_player.stop()          # queued reply audio
self._announcer.stop(...)       # "hmm, one sec" filler -- separate stream
self._stop_speaking_gestures()  # or it waves in silence
```

---

## Step 7 — Order matters in the turn pipeline

In `_process_turn`, check the mute command **before** the reject filter:

```python
if await _handle_mute_command(ws, sess, transcript):
    return

reason = legacy.transcript_reject_reason(...)   # drops 1-2 word utterances
```

Commands are one or two words, and the reject filter treats utterances that
short as noise. Placed after it, the command never arrives.

---

## Step 8 — The bug you will actually hit

Everything above was built, had **24 passing tests**, and failed on the robot.

Every test fed the matcher a clean `"Nao mute"`. The real transcript, from
`logs/server.log` on 2026-09-04:

```
"Well. Thanks for asking. How are things going with you? Now on mute."
 \____________ NAO's own echoed reply _______________/ \____ you ____/
```

The open mic means NAO's voice and yours land in **one buffer** and arrive as
one 13-word transcript. The four-word cap rejected it. Zero mute hits.

The mechanism that makes the feature possible is the same one that defeats it.

**Fix:** subtract NAO's known words from the front, then apply the unchanged
strict matcher to the remainder.

```python
def strip_leading_echo(transcript, spoken_text, min_stripped=3):
    """Walk from the left, dropping tokens that appear in what NAO just said."""
```

```
'Well. Thanks for asking. How are things going with you? Now on mute.'
  − last reply
  = 'now on mute'  → strip address → 'on mute' → 2 words → MUTE
```

`min_stripped` matters: without it, a normal sentence that merely opens with a
common word gets whittled down into a command.

`classify_with_echo()` tries the plain transcript first, so the retry can only
turn a `None` into a match — never the reverse.

---

## Step 9 — Test from real logs, not from imagination

The single most useful lesson here.

```python
# Verbatim from logs/server.log, 2026-09-04 16:29:22
REAL_TRANSCRIPT = (
    "Well. Thanks for asking. How are things going with you? Now on mute."
)

def test_the_real_robot_failure_now_matches(self):
    assert mute_words.classify(REAL_TRANSCRIPT) is None
    assert mute_words.classify_with_echo(REAL_TRANSCRIPT, REAL_REPLY) == "mute"
```

Note `pytest-asyncio` is **not installed** — `@pytest.mark.asyncio` tests
silently skip. Drive coroutines with `asyncio.run()`; see
`server/tests/test_mute_integration.py` for the `FakeWS` pattern.

---

## Step 10 — Know what else the open mic breaks

Keeping the mic open during TTS is not a local change. It makes NAO hear
itself constantly, and anything that acts on a transcript becomes reachable by
NAO's own voice.

Real consequence, same session: NAO spoke its camera announcement — which
contains the words *"say stop watching me anytime"* — heard itself, matched
the `disable_camera` trigger, and **turned its own camera off.**

Two guards were needed:

1. `_rebind_username()` — the echo store is keyed by username, and a session
   renames itself (`guest` → learned name) mid-conversation. That orphaned
   everything NAO had said and blinded the guard for exactly one turn.
2. `_is_system_line_echo()` — a **stateless** check on NAO's fixed lines
   (announcements, prompts, hotline reply). Never valid user input, regardless
   of session state.

The second requires a **≥5 token floor**. Without it, `"stop watching me"`
overlaps the announcement 100% and gets eaten — the guard would swallow the
very command the announcement tells people to say.

---

## Deploying

```
robot code  → rsync   →  ./run.sh     (never git)
server code → git push →  Pi pulls    (never rsync)
```

Use `./run.sh`. It clears `.pyc` (Python 2 prefers stale bytecode, so edits
silently no-op) and excludes `nao.log` (a bare `rsync --delete` deletes the
robot's live log file).

---

## Env knobs

| Var | Default | Effect |
|---|---|---|
| `MUTE_LISTEN_ENABLED` | `1` | server-side side-channel listener |
| `MUTE_LISTEN_DURING_TTS` | `1` | robot keeps mic open while speaking |

Set either to `0` to restore deaf-while-speaking behaviour.

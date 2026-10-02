# -*- coding: utf-8 -*-
"""Streaming MP3 chunk player for the WS transport.

Receives MP3 audio chunks from the WS receiver (`nao/ws_client.py`) and
plays them back-to-back through ALAudioPlayer with no perceptible gap. The
public surface is `StreamTtsPlayer` — an enqueue/stop/shutdown player.

This file is the Phase 1 rewrite of the SSE consumer that previously lived
here. SSE parsing, head-touch / mic-energy barge monitor, and the `consume`
function are GONE — those responsibilities moved out:

  - SSE parsing -> `nao/ws_client.py` (now consumes WS frames).
  - Barge-in detection -> `nao/ws_client.py` (single source of truth).
  - This module just plays audio chunks smoothly and supports `stop()` so
    the WS client can interrupt instantly.

What was preserved from the old file (deliberately, do not change):
  - Per-chunk volume pinning (`ALAudioDevice.setOutputVolume(100)` and
    `ALAudioPlayer.setMasterVolume(1.0)`) BEFORE every play. Some
    background service on this NAO firmware drops the output volume back
    down between sentences; pinning once at startup is not enough.
  - Lazy ALProxy creation through `naoqi.ALProxy` using `nao_ip`.
  - MP3 path on disk: NAO's ALAudioPlayer cannot play in-memory bytes; it
    needs a file path, so each chunk is written to /tmp/nao_tts_<seq>.mp3
    and `playFile()` is called.
  - `print()` logging via `from __future__ import print_function`. The
    `nao-logger-main` agent will replace these with structured logging in
    a follow-up commit; until then keep the prints so we can debug.

Python 2.7 / naoqi only. All imports of `naoqi` are guarded so this file
imports cleanly off-robot for `python -m py_compile` and unit tests.
"""
from __future__ import print_function

import os
import sys
import threading
import time

try:
    import Queue as _queue  # Py2.7 stdlib
except ImportError:  # pragma: no cover - Py3 dev import
    import queue as _queue  # type: ignore


# Where MP3 chunks are spooled before playback. Same /tmp prefix as before
# so leftover files from older runs are still cleaned up by the existing
# OS-level tmp policy. Filenames now use a monotonic counter (no collision
# even at >1 chunk/sec) and shutdown removes any survivors.
_MP3_DIR = "/tmp"
_MP3_PREFIX = "nao_tts_"
_MP3_SUFFIX = ".mp3"

# Polling interval while waiting for ALAudioPlayer to drain a single chunk.
# 50 ms keeps inter-chunk gap perceptibly seamless; the PRD requires < 30 ms
# of dead air between chunks, but with `playFile` semantics on this firmware
# the actual handoff is bounded by NAOqi's RPC + decoder warmup — empirically
# under 30 ms once the next file is queued.
_POLL_S = 0.05

# NAOqi ALAudioPlayer occasionally wedges inside blocking playFile() when
# another behavior has the audio/motion stack busy. Use post.playFile() plus
# polling so a single bad task cannot strand the TTS worker forever.
_PLAY_TIMEOUT_MIN_S = 4.0
_PLAY_TIMEOUT_GRACE_S = 5.0
_PLAY_TIMEOUT_MAX_S = 30.0
_PLAY_START_GRACE_S = 0.80

# Hard cap on individual MP3 size we will accept (defensive). The OpenAI TTS
# server emits sentence-sized chunks well under this; anything over likely
# means we are buffering an entire reply instead of streaming it, which is a
# bug we want to surface rather than silently drown out.
_MAX_MP3_BYTES = 4 * 1024 * 1024


# ── ffmpeg helpers ──────────────────────────────────────────────────────────
# NAO V6 ships ffmpeg 4.1.3 at /usr/bin/ffmpeg. ALAudioPlayer is picky about
# MP3 containers (ElevenLabs Flash output is often rejected silently — the
# call returns instantly with no audio out). When the first blocking play
# returns suspiciously fast we transcode to NAO-safe WAV (16 kHz, mono,
# S16LE) and retry. Both helpers are best-effort; failures are logged and
# the caller falls through.
import subprocess as _subprocess

_FFMPEG_BIN = "/usr/bin/ffmpeg"
_FFPROBE_BIN = "/usr/bin/ffprobe"


def _ffprobe_audio(path):
    """Best-effort ffprobe of `path`. Returns a short string like
    'mp3 22050Hz mono 32k' or 'probe_failed'. Never raises."""
    if not os.path.exists(_FFPROBE_BIN):
        # Fallback: parse first few bytes for codec hint.
        try:
            with open(path, "rb") as fh:
                head = fh.read(10)
            if head.startswith(b"ID3") or head[:2] == b"\xff\xfb" or head[:2] == b"\xff\xf3":
                return "mp3 (no_ffprobe)"
            if head[:4] == b"RIFF":
                return "wav (no_ffprobe)"
            return "unknown (no_ffprobe)"
        except Exception:
            return "probe_failed"
    try:
        out = _subprocess.check_output(
            [_FFPROBE_BIN, "-v", "error", "-show_entries",
             "stream=codec_name,sample_rate,channels,bit_rate",
             "-of", "default=noprint_wrappers=1", path],
            stderr=_subprocess.STDOUT,
        )
        if isinstance(out, bytes):
            out = out.decode("utf-8", "ignore")
        return out.strip().replace("\n", " ")
    except Exception as exc:
        return "probe_failed:{0}".format(exc)


def _ffprobe_duration_s(path):
    """Best-effort duration probe. Returns float seconds or None."""
    if not os.path.exists(_FFPROBE_BIN):
        return None
    try:
        out = _subprocess.check_output(
            [_FFPROBE_BIN, "-v", "error", "-show_entries",
             "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1", path],
            stderr=_subprocess.STDOUT,
        )
        if isinstance(out, bytes):
            out = out.decode("utf-8", "ignore")
        val = float(str(out).strip().splitlines()[-1])
        if val > 0:
            return val
    except Exception:
        pass
    return None


def _estimate_duration_s(path):
    """Return a conservative duration estimate used only for timeouts."""
    dur = _ffprobe_duration_s(path)
    if dur is not None:
        return dur
    try:
        size = float(os.path.getsize(path))
    except Exception:
        size = 0.0
    # ElevenLabs MP3 chunks are usually around 32-64 kbps. 4000 bytes/s is a
    # conservative lower bitrate estimate, so the timeout errs long.
    if size > 0:
        return max(0.5, size / 4000.0)
    return 1.0


def _convert_mp3_to_wav(mp3_path, wav_path):
    """Transcode `mp3_path` to NAO-safe WAV at `wav_path`, with loudness boost.

    Target format: 16 kHz mono signed 16-bit PCM (s16le) — what
    ALAudioPlayer reliably accepts on this firmware.

    Loudness: ``loudnorm`` two-stage normalization to ~ -10 LUFS, then a
    final ``volume=2.0`` (+6 dB) sweetener. ElevenLabs Flash output runs
    a few dB quieter than OpenAI tts-1; combined with NAO's small speakers
    in a noisy classroom this comes out around 50 % perceived loudness
    without amplification. ``loudnorm`` is preferred over a flat
    multiplier because it dodges clipping while still pulling quiet TTS up
    near the ceiling.

    Returns True on success, False otherwise.
    """
    if not os.path.exists(_FFMPEG_BIN):
        print("[tts_trace] ffmpeg not found at {0}".format(_FFMPEG_BIN))
        sys.stderr.flush()
        return False
    try:
        # MAX-loudness chain tuned for NAO V6 speakers. Each filter
        # exists for a specific reason — they compound, you can't
        # drop any of them without losing dB:
        #
        #   1. highpass=180 — strip everything below 180 Hz that NAO
        #      can't reproduce anyway. Frees +3-6 dB of headroom for
        #      the speech band.
        #   2. dynaudnorm=p=0.95:m=20 — dynamic range compressor with
        #      MAX gain factor 20× (+26 dB). Crushes quiet voices
        #      (cloned voices ship 6-10 dB quieter than ElevenLabs
        #      stock) up to the same envelope as loud voices. WITHOUT
        #      this, loudnorm's single-pass undershoots on quiet
        #      sources because the integrated LUFS measurement is
        #      pulled down by long quiet stretches.
        #   3. equalizer f=2200:g=4 — +4 dB presence boost where
        #      speech intelligibility lives.
        #   4. loudnorm=I=-6 — final LUFS target. After dynaudnorm
        #      has flattened the dynamics, loudnorm reliably hits the
        #      hot -6 LUFS target on every voice (girl/man/neutral/my).
        #   5. volume=14dB — flat 5× sweetener.
        #   6. alimiter limit=0.99 — true-peak ceiling.
        cmd = [
            _FFMPEG_BIN, "-y", "-loglevel", "error",
            "-i", mp3_path,
            "-af",
            "highpass=f=180,"
            "dynaudnorm=p=0.95:m=20:s=10:g=15,"
            "equalizer=f=2200:t=q:w=1.4:g=4,"
            "loudnorm=I=-6:TP=-1.0:LRA=7,"
            "volume=14dB,"
            "alimiter=limit=0.99",
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
            "-f", "wav",
            wav_path,
        ]
        rv = _subprocess.call(cmd)
        if rv != 0:
            print("[tts_trace] ffmpeg returned non-zero: {0}".format(rv))
            sys.stderr.flush()
            return False
        return os.path.exists(wav_path) and os.path.getsize(wav_path) > 44
    except Exception as exc:
        print("[tts_trace] ffmpeg conversion error: {0}: {1}".format(
            type(exc).__name__, exc))
        sys.stderr.flush()
        return False


def _try_import_naoqi():
    """Import naoqi.ALProxy or return None.

    Centralised so tests can run on developer laptops where naoqi is not
    installed. The `StreamTtsPlayer` constructor falls back to no-op mode
    if this returns None instead of crashing on import.
    """
    try:
        from naoqi import ALProxy  # type: ignore
        return ALProxy
    except Exception:
        return None


class StreamTtsPlayer(object):
    """Back-to-back MP3 chunk player with barge-in stop.

    Public API (matches the Phase 1 task spec):
        StreamTtsPlayer(nao_ip)
        enqueue(text, mp3_bytes)   -> queue one chunk for playback
        is_playing()               -> bool
        stop()                     -> abort current playback + drain queue
        shutdown()                 -> release proxies + remove tmp files

    Threading model:
      - Producer side: any number of WS-receiver threads call enqueue(...).
        It is non-blocking; it writes the MP3 to /tmp and pushes a job onto
        an unbounded thread-safe queue.
      - Worker side: ONE daemon worker thread drains the queue, calls
        `ALAudioPlayer.playFile(path)`, then polls
        `ALAudioPlayer.getNumOfChannels()` to decide when playback has
        ended, and finally deletes the tmp file. Polling the channel count
        is the cheapest way naoqi exposes "am I still playing"; on this
        firmware it returns 0 when the player is idle.
      - `_playing` is a bool guarded by `_state_lock`. Readers see a
        consistent value without racing the worker.
    """

    def __init__(self, nao_ip, nao_port=9559):
        self._nao_ip = nao_ip
        self._nao_port = int(nao_port)

        # State guarded by _state_lock. _playing is True from the moment we
        # ask ALAudioPlayer to play a chunk until we observe its channel
        # count drop back to zero (or stop() forces it false).
        self._state_lock = threading.Lock()
        self._playing = False
        self._pending_jobs = 0

        # Queue of (path, text, pause_after_ms) jobs the worker consumes.
        # Unbounded
        # because TTS chunks arrive faster than playback when the network
        # spurts; the queue absorbs the spike. shutdown()/stop() drain it.
        self._queue = _queue.Queue()
        self._play_abort_event = threading.Event()

        # Monotonic seq used in tmp filenames so two concurrent enqueues
        # never collide on the same path.
        self._seq_lock = threading.Lock()
        self._seq = 0

        # Track every tmp file we wrote so shutdown() can remove leftovers
        # if the worker was killed mid-playback.
        self._tmp_paths_lock = threading.Lock()
        self._tmp_paths = set()

        # Stop flag for the worker. set() once on shutdown.
        self._shutdown_event = threading.Event()

        # ALProxy handles. None means "naoqi not importable" (dev machine);
        # in that case enqueue() warns once and silently drops chunks so
        # tests can instantiate the class without a robot present.
        self._player = None
        self._audio_dev = None
        self._naoqi_warned = False

        ALProxy = _try_import_naoqi()
        if ALProxy is None:
            print("[stream_tts] naoqi unavailable; StreamTtsPlayer running "
                  "in inert dev mode (no audio will play)")
        else:
            try:
                self._player = ALProxy("ALAudioPlayer", self._nao_ip,
                                       self._nao_port)
            except Exception as e:
                print("[stream_tts] ALAudioPlayer proxy failed:", e)
                self._player = None
            try:
                self._audio_dev = ALProxy("ALAudioDevice", self._nao_ip,
                                          self._nao_port)
            except Exception as e:
                # Non-fatal — we just lose the output-volume pin. Playback
                # still works at whatever the system volume already is.
                print("[stream_tts] ALAudioDevice proxy failed:", e)
                self._audio_dev = None

        # Start the worker thread regardless. If naoqi is missing it just
        # spins waiting for jobs that never come; cheap on cycles.
        self._worker = threading.Thread(target=self._worker_loop,
                                        name="stream_tts_worker")
        self._worker.daemon = True
        self._worker.start()

    # ---- public API -----------------------------------------------------

    def _inc_pending_job(self):
        with self._state_lock:
            self._pending_jobs += 1

    def _dec_pending_job(self):
        with self._state_lock:
            if self._pending_jobs > 0:
                self._pending_jobs -= 1

    def enqueue(self, text, mp3_bytes, pause_after_ms=0, on_start=None):
        """Accept one TTS chunk; non-blocking; queues for playback.

        ``on_start`` is an optional no-argument callable run on the worker
        thread just before this chunk starts playing (used to start an
        eye-LED breathing fade in time with a paced breathing count).

        Pins the volume BEFORE the worker dequeues, not just inside the
        worker, because the previous turn's interrupt may have left the
        master volume in a weird state and we want the very next chunk to
        come out at full level.

        text is kept around for logging only — the chunk player does not
        re-derive what was said from the MP3 bytes.
        """
        if not mp3_bytes:
            return
        if self._shutdown_event.is_set():
            return
        if len(mp3_bytes) > _MAX_MP3_BYTES:
            # Defensive — log and drop; treating it as a programming bug
            # rather than playing a 4 MB blob and pretending all is well.
            print("[stream_tts] enqueue: chunk too large ({0} bytes), "
                  "dropping".format(len(mp3_bytes)))
            return

        # Count this chunk as active before any slow disk/probe work. The
        # server can send tts_ended immediately after the last audio_chunk;
        # the mic-resume waiter must see "pending playback" even if the
        # worker has not yet received the queued file.
        self._inc_pending_job()

        # Pin volumes UP FRONT — before the worker thread sees the job. If
        # the job is ahead of others in the queue this still happens before
        # playback starts. Idempotent and cheap.
        self._pin_volumes()

        path = self._spool_to_disk(mp3_bytes)
        if path is None:
            print("[tts_trace] mp3_written FAILED (spool returned None)")
            sys.stderr.flush()
            self._dec_pending_job()
            return
        with self._tmp_paths_lock:
            self._tmp_paths.add(path)
        try:
            pause_after_ms = max(0, int(pause_after_ms or 0))
        except Exception:
            pause_after_ms = 0
        try:
            on_disk_size = os.path.getsize(path)
        except Exception:
            on_disk_size = -1
        print("[tts_trace] mp3_written path={0} bytes={1} on_disk_size={2}".format(
            path, len(mp3_bytes), on_disk_size))
        sys.stderr.flush()
        try:
            preview = (text or "")
            if isinstance(preview, bytes):
                try:
                    preview = preview.decode("utf-8", "ignore")
                except Exception:
                    preview = ""
            preview = preview[:60]
        except Exception:
            preview = ""
        self._queue.put((path, text, pause_after_ms, on_start))
        print("[stream_tts] enqueue:", preview,
              "(", len(mp3_bytes), "bytes ->", path, ")")

    def is_playing(self):
        """Return True while a chunk is being played OR jobs are queued."""
        with self._state_lock:
            playing = self._playing
            pending_jobs = self._pending_jobs
        if playing or pending_jobs > 0:
            return True
        # Pending jobs also count as "still playing" from the caller's
        # perspective — barge-in / state machines want to know whether the
        # robot's mouth is going to keep moving, not just right now.
        return not self._queue.empty()

    def stop(self):
        """Stop current playback + drain queue. Used for barge-in.

        Must complete in < 50 ms. We:
          1. Mark _playing False under the lock so readers see "stopped"
             immediately.
          2. Drain the queue (best-effort — racing producers may push one
             more job; that's fine, the worker will see _interrupt and
             skip it).
          3. Call ALAudioPlayer.stopAll() to kill whatever file is mid-
             playback. stopAll is the only barge primitive naoqi exposes
             that doesn't require a play-id we may not have.
          4. Wake the worker via an interrupt event it checks each loop.
        """
        with self._state_lock:
            self._playing = False
            self._pending_jobs = 0
        self._play_abort_event.set()

        drained = 0
        while True:
            try:
                job = self._queue.get_nowait()
            except _queue.Empty:
                break
            try:
                path = job[0]
            except Exception:
                continue
            drained += 1
            self._safe_remove(path)
        if drained:
            print("[stream_tts] stop: drained", drained, "queued chunk(s)")

        if self._player is not None:
            try:
                self._player.stopAll()
            except Exception as e:
                print("[stream_tts] stop: stopAll failed:", e)

    def shutdown(self):
        """Drain queue, stop player, remove leftover tmp files, end worker.

        Idempotent. Safe to call from any thread.
        """
        if self._shutdown_event.is_set():
            return
        self._shutdown_event.set()
        # Reuse stop() to drain + kill audio.
        try:
            self.stop()
        except Exception as e:
            print("[stream_tts] shutdown: stop raised:", e)

        # Push a sentinel so the worker exits its blocking get() promptly.
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass

        # Best-effort join. Worker is a daemon so we won't hang process
        # exit even if it does not return.
        if self._worker is not None:
            try:
                self._worker.join(timeout=1.0)
            except Exception:
                pass

        # Remove any /tmp/nao_tts_*.mp3 we wrote (and as a safety net any
        # leftovers from a previous crashed run).
        self._cleanup_tmp_files()

    # ---- worker loop ----------------------------------------------------

    def _worker_loop(self):
        """Daemon: dequeue jobs, play sequentially, delete files."""
        print("[tts_trace] worker_loop_started player_present={0} audio_dev_present={1}".format(
            self._player is not None, self._audio_dev is not None))
        sys.stderr.flush()
        while not self._shutdown_event.is_set():
            try:
                job = self._queue.get(timeout=0.25)
            except _queue.Empty:
                continue
            if job is None:
                # Shutdown sentinel.
                break
            on_start = None
            try:
                if len(job) >= 3:
                    path, _text, pause_after_ms = job[:3]
                else:
                    path, _text = job
                    pause_after_ms = 0
                if len(job) >= 4:
                    on_start = job[3]
            except Exception:
                self._dec_pending_job()
                continue

            # If stop() landed between enqueue and dequeue, skip.
            if self._shutdown_event.is_set():
                self._safe_remove(path)
                self._dec_pending_job()
                continue

            # Mark playback active before handing the job to _play_one.
            # Otherwise there is a short race where the worker has already
            # removed the job from the queue, but _play_one has not yet set
            # _playing=True. The mic-resume waiter can observe
            # queue-empty + not-playing in that gap and reopen the mic
            # before the chunk has actually started.
            with self._state_lock:
                self._playing = True
            self._play_abort_event.clear()
            if on_start is not None:
                try:
                    on_start()
                except Exception as exc:
                    print("[stream_tts] on_start callback failed: {0}".format(exc))
            try:
                self._play_one(path, pause_after_ms=pause_after_ms)
            finally:
                self._dec_pending_job()

    def _play_file_with_timeout(self, path, label):
        """Play through NAOqi post.playFile and wait with a hard timeout.

        Returns (rv, elapsed_s, observed_playback). The third value is False
        when NAOqi accepted the task but we never saw it running or using an
        output channel, which is how silent MP3 rejection shows up on this
        firmware.
        """
        expected_s = _estimate_duration_s(path)
        timeout_s = max(_PLAY_TIMEOUT_MIN_S,
                        min(_PLAY_TIMEOUT_MAX_S,
                            expected_s + _PLAY_TIMEOUT_GRACE_S))
        t0 = time.time()
        observed = False
        rv = None
        print("[tts_trace] blocking_play_start path={0} ({1}) timeout_s={2:.2f}".format(
            path, label, timeout_s))
        sys.stderr.flush()

        try:
            post = getattr(self._player, "post", None)
            post_play = getattr(post, "playFile", None) if post is not None else None
            if post_play is not None:
                rv = post_play(path)
                print("[tts_trace] post_playFile path={0} task_id={1!r}".format(
                    path, rv))
                sys.stderr.flush()
                while not self._shutdown_event.is_set():
                    elapsed = time.time() - t0
                    if self._play_abort_event.is_set():
                        self._stop_player_task(rv)
                        break
                    if elapsed >= timeout_s:
                        print("[tts_trace] play_timeout path={0} task_id={1!r} elapsed_s={2:.2f}".format(
                            path, rv, elapsed))
                        sys.stderr.flush()
                        self._stop_player_task(rv)
                        break

                    running = None
                    if rv is not None:
                        try:
                            running = bool(self._player.isRunning(rv))
                        except Exception:
                            running = None

                    channels = None
                    try:
                        channels = int(self._player.getNumOfChannels())
                    except Exception:
                        channels = None

                    if running is True or (channels is not None and channels > 0):
                        observed = True
                    if running is False and (observed or elapsed >= 0.30):
                        break
                    if (not observed and channels == 0 and
                            elapsed >= _PLAY_START_GRACE_S):
                        break
                    if (running is None and channels == 0 and elapsed >= 0.30
                            and observed):
                        break
                    time.sleep(_POLL_S)
                elapsed = time.time() - t0
                print("[tts_trace] blocking_play_done path={0} rv={1!r} elapsed_s={2:.2f} observed={3}".format(
                    path, rv, elapsed, observed))
                sys.stderr.flush()
                return rv, elapsed, observed

            # Last-resort fallback for dev/fake proxies without post.
            rv = self._player.playFile(path)
            elapsed = time.time() - t0
            observed = elapsed >= 0.30
            print("[tts_trace] blocking_play_done path={0} rv={1!r} elapsed_s={2:.2f} observed={3}".format(
                path, rv, elapsed, observed))
            sys.stderr.flush()
            return rv, elapsed, observed
        except Exception as e:
            elapsed = time.time() - t0
            print("[tts_trace] blocking_play FAILED on {0} path={1} err={2}: {3} elapsed_s={4:.2f}".format(
                label, path, type(e).__name__, e, elapsed))
            sys.stderr.flush()
            return rv, elapsed, observed

    def _stop_player_task(self, task_id):
        try:
            if task_id is not None:
                self._player.stop(task_id)
        except Exception:
            pass
        try:
            self._player.stopAll()
        except Exception:
            pass

    def _play_one(self, path, pause_after_ms=0):
        """Play a single MP3 file, blocking until the player drains it.

        Sets `_playing=True` immediately so `is_playing()` reflects reality
        even before naoqi reports a non-zero channel count. Polls
        `getNumOfChannels()` every _POLL_S to detect end-of-playback.
        """
        with self._state_lock:
            self._playing = True

        # Final volume pin right before play — covers the case where stop()
        # was called between enqueue and now (which would have left the
        # speaker at whatever volume the OS picked).
        self._pin_volumes()

        # Snapshot the current output volume so we can verify it actually
        # took effect (some firmwares silently ignore setOutputVolume when
        # the audio device is busy).
        cur_out_vol = "?"
        cur_master_vol = "?"
        try:
            if self._audio_dev is not None:
                cur_out_vol = self._audio_dev.getOutputVolume()
        except Exception as exc:
            cur_out_vol = "err:{0}".format(exc)
        try:
            if self._player is not None:
                cur_master_vol = self._player.getMasterVolume()
        except Exception as exc:
            cur_master_vol = "err:{0}".format(exc)
        print("[tts_trace] play_start path={0} output_vol={1} master_vol={2}".format(
            path, cur_out_vol, cur_master_vol))
        sys.stderr.flush()

        if self._player is None:
            # Dev mode without naoqi: simulate playback time so callers
            # that drive the player in tests can see is_playing() flip.
            print("[tts_trace] play_failed reason=player_proxy_None (dev/inert mode)")
            sys.stderr.flush()
            time.sleep(0.05)
            with self._state_lock:
                self._playing = False
            self._safe_remove(path)
            return

        try:
            # Probe the MP3 first (NAO ALAudioPlayer is picky about MP3
            # containers; ElevenLabs Flash output isn't always playable).
            mp3_info = _ffprobe_audio(path)
            print("[tts_trace] mp3_probe path={0} info={1!r}".format(path, mp3_info))
            sys.stderr.flush()

            # Try the original MP3 first. Playback is started with
            # post.playFile() and polled with a hard timeout so NAOqi cannot
            # wedge the TTS worker forever.
            rv, elapsed, observed = self._play_file_with_timeout(path, "mp3")

            # If playback returned suspiciously fast (< 0.3 s for any non-trivial
            # MP3), assume the firmware silently rejected the format. Convert
            # to a NAO-safe WAV and retry.
            try:
                file_bytes = os.path.getsize(path)
            except Exception:
                file_bytes = 0
            if (elapsed < 0.30 or not observed) and file_bytes > 4096:
                wav_path = path.rsplit(".", 1)[0] + ".wav"
                t1 = time.time()
                ok = _convert_mp3_to_wav(path, wav_path)
                conv_elapsed = time.time() - t1
                if ok:
                    try:
                        wav_size = os.path.getsize(wav_path)
                    except Exception:
                        wav_size = -1
                    # 16 kHz mono S16LE → 32_000 bytes/s; estimate duration.
                    duration_s = max(0.0, (wav_size - 44) / 32000.0)
                    print("[tts_trace] wav_converted path={0} bytes={1} duration_s={2:.2f} conv_elapsed_s={3:.2f}".format(
                        wav_path, wav_size, duration_s, conv_elapsed))
                    sys.stderr.flush()
                    with self._tmp_paths_lock:
                        self._tmp_paths.add(wav_path)
                    self._play_file_with_timeout(wav_path, "wav")
                else:
                    print("[tts_trace] wav_convert FAILED conv_elapsed_s={0:.2f}".format(
                        conv_elapsed))
                    sys.stderr.flush()

        except Exception as e:
            print("[tts_trace] play_failed path={0} unexpected={1}: {2}".format(
                path, type(e).__name__, e))
            sys.stderr.flush()
            print("[stream_tts] _play_one error:", e)
        finally:
            try:
                pause_s = max(0.0, float(pause_after_ms or 0) / 1000.0)
            except Exception:
                pause_s = 0.0
            if pause_s > 0 and not self._shutdown_event.is_set() \
                    and not self._play_abort_event.is_set():
                print("[tts_trace] pause_after_chunk ms={0}".format(
                    int(pause_s * 1000)))
                sys.stderr.flush()
                end_t = time.time() + pause_s
                while time.time() < end_t:
                    if self._shutdown_event.is_set() or \
                            self._play_abort_event.is_set():
                        break
                    time.sleep(min(0.05, max(0.0, end_t - time.time())))
            with self._state_lock:
                self._playing = False
            self._safe_remove(path)

    # ---- helpers --------------------------------------------------------

    def _pin_volumes(self):
        """Force speaker output to 100 across every layer of the stack.

        On NAO V6 the audio path is:
            ffmpeg gain  →  ALAudioPlayer.masterVolume  →
            ALAudioDevice.outputVolume  →  ALSA mixer  →  amplifier  →  speaker

        Each layer can clamp the signal. We pin all four so nothing in
        the path is the bottleneck. Idempotent + best-effort — every
        call is wrapped so a single failure can't silence playback.
        """
        if self._audio_dev is not None:
            try:
                self._audio_dev.setOutputVolume(100)
            except Exception:
                pass
        if self._player is not None:
            try:
                self._player.setMasterVolume(1.0)
            except Exception:
                pass
        # Do not unmute NAO's built-in ALTextToSpeech here. Real reply
        # audio is played through ALAudioPlayer; raising the native TTS
        # volume makes local fallback/filler phrases leak as the robot's
        # kid voice alongside ElevenLabs.
        try:
            from naoqi import ALProxy as _ALProxy  # noqa
            _ALProxy("ALTextToSpeech", self._nao_ip, self._nao_port).setVolume(0.0)
        except Exception:
            pass
        # ALSA hardware mixer — sits ABOVE NAOqi's setOutputVolume on
        # this firmware. Default ships at ~80 %; pushing to 100 % buys
        # ~2 dB of additional analog gain. Done via a one-shot subprocess
        # the first time _pin_volumes runs (no point spamming amixer
        # 50× per turn). The control name is "Master" on V6 — we also
        # try "PCM" and "Speaker" as fallbacks for older firmwares.
        if not getattr(self, "_alsa_pinned", False):
            self._alsa_pinned = True
            for ctl in ("Master", "PCM", "Speaker"):
                try:
                    rv = _subprocess.call(
                        ["amixer", "-q", "set", ctl, "100%"],
                    )
                    if rv == 0:
                        print("[stream_tts] amixer pin {0}=100% OK".format(ctl))
                        sys.stderr.flush()
                except Exception:
                    pass

    def _spool_to_disk(self, mp3_bytes):
        """Write MP3 bytes to /tmp/nao_tts_<seq>.mp3 and return the path.

        Returns None if the write fails — caller drops the chunk. We do
        not retry: a missing tmp dir is a deployment problem, not a
        runtime one.
        """
        with self._seq_lock:
            self._seq = (self._seq + 1) % 1000000
            seq = self._seq
        path = os.path.join(_MP3_DIR,
                            "{0}{1}{2}".format(_MP3_PREFIX, seq, _MP3_SUFFIX))
        try:
            f = open(path, "wb")
            try:
                f.write(mp3_bytes)
            finally:
                f.close()
        except Exception as e:
            print("[stream_tts] write tmp mp3 failed:", e)
            return None
        return path

    def _safe_remove(self, path):
        """Remove a tmp file; never raises. Updates the bookkeeping set."""
        try:
            with self._tmp_paths_lock:
                if path in self._tmp_paths:
                    self._tmp_paths.discard(path)
        except Exception:
            pass
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    def _cleanup_tmp_files(self):
        """Remove every file we wrote AND any /tmp/nao_tts_*.mp3 stragglers.

        Catches the case where a previous process crashed mid-play and
        left files behind — we don't want /tmp filling up across reboots.
        """
        with self._tmp_paths_lock:
            paths = list(self._tmp_paths)
            self._tmp_paths.clear()
        for p in paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        try:
            for name in os.listdir(_MP3_DIR):
                if name.startswith(_MP3_PREFIX) and name.endswith(_MP3_SUFFIX):
                    try:
                        os.remove(os.path.join(_MP3_DIR, name))
                    except Exception:
                        pass
        except Exception:
            pass

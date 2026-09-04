# Nao-OpenAI-Morgan-Assist

## Project Overview

NAO humanoid robot assistant for Morgan State University, built on the **OpenAI Agents SDK** with multi-agent routing (router + chat, chatbot, skills, therapist with CBT/grounding/MI sub-agents). Multimodal emotion detection via camera vision.

**The stack is no longer all-OpenAI** (changed 2026-07-28):

| Layer | Provider | Detail |
|---|---|---|
| **Ears** | Deepgram | `nova-3` batch REST, retrying once on `nova-2` when the first pass returns empty. **No OpenAI fallback** as of 2026-07-30 — see below. ElevenLabs Scribe is off (`USE_ELEVENLABS_STT=0`); ElevenLabs is TTS-only now. |
| **Brain** | Anthropic | Haiku 4.5 for general chat/router/skills/embodied (fast, ~1s); Sonnet 5 for therapist + chatbot (depth); Opus 5 for crisis/safety. Chat moved off Sonnet 5 to Haiku on 2026-07-29 — Sonnet measured ~3.5s/reply (19.6s on tool-heavy turns), too slow for real-time. |
| **Voice** | ElevenLabs | `eleven_flash_v2_5`, falls back to OpenAI `tts-1` |
| **Safety + CBT tools + vision** | *configurable* | `CRISIS_MODEL` / `VISION_MODEL` — OpenAI by code default, but the live Mac + Pi `.env` now point them at Claude (`claude-opus-5` / `claude-sonnet-5`) as of 2026-07-29 |

`OPENAI_API_KEY` is still **mandatory**: `config.py:17` reads it with
`os.environ[...]`, so a missing key is a `KeyError` at import and the server
never boots — and the fallback paths, vision, and safety all still need it.

**Two indirection layers make provider a config choice, not a code edit:**

- `server/model_factory.py` — for agents on the Agents SDK. `resolve_model()`
  passes OpenAI ids through as strings and wraps `claude-*` ids in
  `LitellmModel`. Any `*_MODEL` env var can name either provider.
- `server/llm_compat.py` — for the call sites that use a provider client
  *directly* and never touch the SDK (crisis classifier in `safety.py`,
  `identify_distortion` / `suggest_reframe` / vision in `tools/emotion.py`,
  rollups in `memory_rollup.py`, and the endpointing check in
  `semantic_endpoint.py`). `chat()` takes OpenAI-shaped messages and
  dispatches on model name. **If you add a direct client call, route it
  through here or it silently pins that feature to one provider.**

  That warning is not theoretical: `semantic_endpoint.py` did
  `from openai import OpenAI` and ran on **every turn**, so the stack was
  still calling OpenAI once per turn long after the agents moved to Claude —
  and no value of `SEMANTIC_ENDPOINT_MODEL` could change it, because the
  module never consulted the dispatcher. Fixed 2026-09-04; it now defaults to
  `claude-haiku-4-5` (12/12 on the endpointing cases, median 557 ms, versus a
  miss on "what time is it" from `gpt-4.1-nano`). Regression test:
  `server/tests/test_semantic_endpoint_provider.py` asserts `_get_client` is
  gone. To find any that remain:
  `grep -rn "from openai import\|import openai" server --include="*.py"`.

Gotcha: `llm_compat` drops `temperature` for Anthropic on purpose. Sampling
params are removed on current Claude models — Opus 5 returns
`400 temperature is deprecated for this model`.

Pinecone is **gone** — RAG is now an HTTP proxy to Cloud Run
(`server/tools/cs_navigator.py`, `vertex_search.py`). No embeddings anywhere.

## Repo Layout

```
Nao-OpenAI-Morgan-Assist/
├── nao/         NAO-side Python 2.7 code — copy this to the robot
├── server/      Python 3.11+ Flask server + Agents SDK graph
├── docs/        Design specs, implementation plans, reference docs
├── README.md
├── CLAUDE.md
├── LICENSE
└── pytest.ini
```

## Architecture

**The live path is FastAPI + WebSocket** (`USE_WS=1` in `.env` → `server/app_ws.py`,
served by uvicorn). `server/server.py` (Flask `POST /turn`) and
`nao/conversation.py` are the **legacy** path — still present, not what runs.
Check `USE_WS` before assuming which file you're debugging. `realtime_proxy.py`
is unreachable under `USE_WS=1`.

```
NAO Robot (Python 2.7 / naoqi SDK) — everything under nao/
  nao/main.py           -> Wake loop + wake state machine
  nao/wake_state.py     -> Face + touch gates (NO voice wake word)
  nao/ws_client.py      -> WebSocket client: streams PCM, plays TTS, runs actions
  nao/audio_module.py   -> Mic capture (fragment recorder; MIC_CHANNEL selects mic)

Server (Python 3.11+) — everything under server/
  server/app_ws.py        LIVE: FastAPI + WebSocket /ws/{username}
  server/server.py        LEGACY: Flask POST /turn (USE_WS=0 only)
  server/safety.py        Pre-dispatch crisis gate (keyword + LLM, 988 hotline)
  server/session.py       SQLiteSession wrapper + camera consent + therapy recaps
  server/agents/          Agent graph (router, chat, chatbot, skills, therapist, cbt_coach, grounding_coach)
  server/tools/           Tool modules (nao_actions, pinecone_search, emotion, skills_tools)
```

## Key Files

### NAO side (Python 2.7, `nao/`)
| File | Purpose |
|------|---------|
| `nao/main.py` | Entry; wake -> conversation loop |
| `nao/wake_listener.py` | Wake phrase detection + `extract_hint()` |
| `nao/conversation.py` | Single mode loop (record, POST, speak, dispatch actions) |
| `nao/audio_handler.py` | Mic recording with VAD |
| `nao/processing_announcer.py` | Background "please wait" speaker |
| `nao/config.py` | Env-driven NAO IP/SERVER IP |
| `nao/utils/camera_capture.py` | JPEG capture including `snap_quick()` for per-turn vision |
| `nao/utils/nao_execute.py` | Dispatches `{name, args}` actions from server to naoqi calls |
| `nao/utils/face_naoqi.py` | Face recognition/learning via ALFaceDetection |
| `nao/utils/ask_name_utils.py` | Ask user for name via audio round-trip |
| `nao/utils/exit_detection.py` | Regex-based exit intent |
| `nao/utils/name_utils.py` | Extract name from speech |
| `nao/utils/speech.py` | Phrase pools + expressive TTS |

### Server side (Python 3.11+)
| File | Purpose |
|------|---------|
| `server/server.py` | Flask `POST /turn` + `GET /health` |
| `server/config.py` | Env config (per-agent models, IPs, SQLite path) |
| `server/model_factory.py` | `resolve_model()` — OpenAI string vs Claude `LitellmModel`, for Agents SDK agents |
| `server/llm_compat.py` | `chat()` — provider-agnostic direct calls (system/JSON/image translation) |
| `server/safety.py` | `crisis_check()` + hardcoded 988 hotline reply (keyword gate, LLM only on soft-trigger match) |
| `server/session.py` | SQLiteSession + `{get,set}_camera_consent` + `{save,load}_recap` |
| `server/agents/router.py` | Triage agent with handoffs |
| `server/agents/chat.py` | General chat + NAO actions |
| `server/agents/chatbot.py` | Morgan CS RAG |
| `server/agents/skills.py` | Time/weather/timers/todos |
| `server/agents/therapist.py` | Empathetic + CBT/grounding handoffs |
| `server/agents/cbt_coach.py` | Thought record walker |
| `server/agents/grounding_coach.py` | 5-4-3-2-1, box breathing, body scan |
| `server/tools/nao_actions.py` | 18 NAO action tools (append to `actions_queue`) |
| `server/tools/pinecone_search.py` | RAG tool |
| `server/tools/emotion.py` | `observe_face`, `log_emotion`, `identify_distortion`, `suggest_reframe`, `set_camera_consent`, `recap_session` |
| `server/tools/skills_tools.py` | Utility tools |

## Development Guidelines

- **NAO-side** (everything in `nao/`): **Python 2.7 compatible**. `from __future__ import print_function`, `str.format()`, no f-strings, no type hints. On the robot, copy `nao/` contents to `/home/nao/nao_assist/` and run `python /home/nao/nao_assist/main.py`.
- **Server-side** (`server/`): **Python 3.11+**. Modern idioms fine.
- IPs/ports read from env or `config.py` — never hardcode.
- Agents SDK: `openai-agents>=0.0.5` (currently 0.13.6).
- `pytest.ini` at repo root pins rootdir so the SDK's `agents` module doesn't get shadowed by `server/agents/`.
- NAO action tools append `{name, args}` records to a context-scoped `actions_queue`; after `Runner.run()` returns, the queue is read out and sent to NAO in the response JSON. NAO-side `utils/nao_execute.py` dispatches them.
- Crisis gate runs **before** the agent sees the user message. Agent cannot override.
- Camera consent persists in `user_prefs` table; therapist tool `set_camera_consent` toggles it; NAO honors the `suppress_image` flag in responses.

## Obsidian Vault

Knowledge vault for this codebase at `~/Documents/Obsidian Vault/Nao-OpenAI-Morgan-Assist/wiki/`. Read `wiki/index.md` first for context. Pattern: `raw/` (immutable) + `wiki/` (LLM-maintained).

## NAO Robot — Connection

- **IP: unknown — re-resolve every session.** `.100` was right on 2026-08-24 and
  was already wrong by 2026-09-03. The lease has moved `.127` → `.123`
  (2026-07-15 → 07-28), `.123` → `.100` (by 08-24, when the **Pi took over the
  robot's old `.123`**), and off `.100` since. A stale `ssh nao` lands you on the
  Pi, and a stale `NAO_IP` makes `run.sh` rsync Python 2.7 robot code onto the
  server. Do not trust any number written here or in `.env`.
- **Fastest way to get the real IP: press NAO's chest button once — it speaks its
  own address aloud.** No network access needed, and it doubles as a power check:
  silence means the robot is off or flat, which is its own answer. `dscacheutil -q
  host -a name nao.local | grep ip_address` works only when mDNS is up (it was
  not on 2026-09-03).
- **Don't trust `.env`'s `NAO_IP` either.** On 2026-09-03 it read `172.20.95.101`,
  which was a **WNC Corporation** device (a phone), not the robot — see the MAC
  vendor trick under "Debugging the robot".
- **Hostname:** `nao.local` (mDNS fallback)
- **User:** `nao`
- **Password:** stored in `.env` as `NAO_PASSWORD` (do NOT commit the password; `.env` is gitignored). Only needed for the initial key push — passwordless SSH is now configured (see below).
- **OS:** Aldebaran RT Linux, kernel 4.4.185, x86_64; **Python 2.7.15** at `/usr/bin/python`.

### SSH

Passwordless key auth is **set up and confirmed working (2026-07-15)** — an ed25519 key (`~/.ssh/id_ed25519`) is installed on the robot and a `Host nao` alias is in `~/.ssh/config`. Just use:

```bash
ssh nao                   # key auth, no password -- but see the warning below
ssh nao@<current-ip>      # explicit host; get the IP from the chest button
ssh nao@nao.local         # elsewhere, if mDNS resolves
```

`~/.ssh/config` entry:

```
Host nao
  HostName <robot's current IP>    # STALE BY DEFAULT -- update on every lease move
  User nao
  IdentityFile ~/.ssh/id_ed25519
```

**The `Host nao` alias pins a hard-coded `HostName`, so it goes stale exactly
when the lease moves — and then `ssh nao` connects to whatever now owns that
address.** On 2026-09-03 the alias still pointed at `.100` while the robot was
elsewhere entirely. Worse, on 2026-08-24 the Pi had taken the robot's old
`.123`, so `ssh nao` silently landed on the **server**. Re-check `HostName`
against the robot's actual IP before trusting the alias, and never assume a
successful `ssh nao` reached the robot — confirm with `hostname` (the robot is
not `naoserver`).

To re-provision the key on a fresh machine: `ssh-copy-id nao@<current-ip>` (uses `NAO_PASSWORD` from `.env` once). VS Code Remote-SSH picks up the `Host nao` alias automatically.

### Making the IP static

Best path: file a ticket with Morgan IT giving them the NAO's WiFi MAC address (`ifconfig wlan0 | grep ether` on the robot) and request a DHCP reservation for the robot. That survives firmware updates and doesn't require touching the robot.

Fallback: configure a fixed IP via Choregraphe (Settings → Network → "Use a fixed IP address") or via `connmanctl` on the robot directly.

## The three computers

The robot is a body, not a brain. Every turn is: NAO records audio → ships it
over WiFi to a **server** → server calls OpenAI → speech + actions come back.
The server runs on one of two machines, and *which one* is the single biggest
source of confusion:

| | Runs | Address | When |
|---|---|---|---|
| **NAO robot** | `nao/` (Python 2.7) | *DHCP — ask the robot (chest button)* | always |
| **Raspberry Pi 4** | `server/` via systemd `nao-server` | `172.20.95.123` | production / demos |
| **Your laptop** | `server/` via `./run.sh` | your LAN IP | development |

The Pi is the always-on brain so the robot works with nobody's laptop around
(see README "Always-on deployment"). `./run.sh` stands your laptop in for it.

- **Robot code** reaches the robot by **rsync** (`./run.sh`) — never git.
- **Server code** reaches the Pi by **git push → Pi `git pull`** — never rsync.
  **There are two remotes and only one of them deploys.** The Pi pulls
  `lama9811/NAO-CBT-ROBOT`; a second copy lives at `github.com/nasomv/NAO-CBT-ROBOT`
  and is what a clone on the Mac may have as `origin`. Pushing to `nasomv`
  deploys **nothing** — auto-deploy never sees it. On 2026-09-03 `nasomv/main`
  was 6 commits behind (`c38e1a0` vs `875badb`, ~25 days) while looking healthy.
  Check `git remote -v` before assuming a push shipped, and reconcile with
  `git fetch https://github.com/lama9811/NAO-CBT-ROBOT.git main`.
  The Pi's `origin` is **`github.com/lama9811/NAO-CBT-ROBOT`** (moved there
  2026-08-24 from `lama9811/nao-sagecbt`, which was itself repointed from a
  collaborator's fork `theaayushstha1/...` on 2026-07-29. The `nao-sagecbt`
  repo is now **gone** — it 404s — so NAO-CBT-ROBOT is the only remote copy of
  the history; it carries all 319 commits). It tracks `main`, so push to `main` (`git push origin HEAD:main`) then
  `ssh naoserver 'cd ~/nao-sagecbt && git pull && sudo systemctl restart nao-server'`.
  **Auto-deploy is set up (2026-07-29):** a systemd timer `nao-autodeploy.timer`
  on the Pi runs `/home/nao/auto-deploy.sh` every 2 min — it `git fetch`es and,
  only when `main` moved, fast-forwards and restarts `nao-server` (logs to
  `~/auto-deploy.log`). So a plain push to `main` reaches the Pi within ~2 min
  with no manual step. A local `git deploy` alias (`.git/config`) pushes AND
  triggers the same script immediately for instant deploys. Pause with
  `ssh naoserver 'sudo systemctl stop nao-autodeploy.timer'`. GitHub webhooks
  can't reach the Pi (no public address behind Morgan's NAT) — hence the Pi
  polls out rather than being pushed to.
- **`.env` reaches the Pi by neither.** It is gitignored, so every key and model
  setting must be recreated there by hand. A Pi that pulled fine but behaves
  like an older build is almost always an out-of-date `.env`. As of 2026-07-29
  the Pi's `.env` was reconciled to Mac parity and **runs the Claude stack**
  (see below). Deliberately NOT synced: `NAO_SHARED_SECRET` (kept the Pi's — it
  pairs with the robot), `FFMPEG_BIN` (`/usr/bin/ffmpeg` on the Pi), and the
  Pi's own API keys. New deps must be pip-installed on the Pi by hand (e.g.
  `litellm`, added 2026-07-29 for the Anthropic `LitellmModel` wrapping).

### Pi access (working as of 2026-07-29)

Passwordless SSH key auth to the Pi is **set up and confirmed working**. Use:

```bash
ssh naoserver               # key auth, no password (alias in ~/.ssh/config)
ssh nao@172.20.95.123       # equivalent, explicit host
```

`~/.ssh/config` has a `Host naoserver` block (`HostName 172.20.95.123`, `User
nao`, `IdentityFile ~/.ssh/id_ed25519`). The same ed25519 key used for the robot
is installed on the Pi. Runs Ubuntu 24.04.4 LTS (aarch64), Python 3.12.

Its hostname is `naoserver`, and it runs `avahi-daemon`, so
`dscacheutil -q host -a name naoserver.local` finds it without scanning — much
faster than a subnet sweep, and it survives lease changes. `PI_IP` is also
recorded in both `.env` files (informational; nothing reads it yet).

**How the key finally got installed (2026-07-29).** The earlier attempts failed
because `cmdline.txt` pins the cloud-init instance-id on the kernel command line
(`ds=nocloud;i=<id>`), which **overrides `meta-data`** — so bumping `meta-data`
alone never marked a new instance and per-instance modules (including the one
that installs `ssh_authorized_keys`) never re-ran. Fix, from the SD card in a
Mac: bump the id in **both** `cmdline.txt` and `meta-data`, and add a
`bootcmd`/`runcmd` in `user-data` that runs `/boot/firmware/nao-fixssh.sh` — an
idempotent script (still on the card) that generates host keys, installs the
key, and starts `ssh.socket`. Gotchas learned: `systemctl restart ssh` from
`bootcmd` hangs boot (do service starts in `runcmd`, timeout-guarded); Ubuntu
uses **socket activation** so you must start `ssh.socket`, not just `ssh.service`;
`enable_ssh: true` in `user-data` is a Raspberry Pi Imager-ism that does nothing.
The script logs to `nao-provision.log` on the FAT boot partition (readable on a
Mac) — read it there if SSH ever fails to come up again.

Note `ssh`/`ssh-copy-id` password prompts **cannot** be answered from a
non-interactive shell (including tool-run commands) — they fail instantly with
`Permission denied` having never prompted. Run those in a real terminal (not
needed now that key auth works).

## Running it (development)

```bash
./run.sh              # deploy nao/ + clear .pyc + start server + relaunch robot + tail
./run.sh deploy-only  # rsync only
./run.sh server-only  # server only
./run.sh stop         # kill server + robot main.py
```

**Robot → server pointing (updated 2026-07-29).** On power-up a Choregraphe
`nao-therapy-autostart` behavior runs `/home/nao/launch_nao_assist.sh`, which
now **resolves `naoserver.local` by mDNS at launch** (5 attempts, 3s apart,
falling back to the last-known `172.20.95.123`) and starts `main.py` against
whatever it finds — so the robot autostarts against the **Pi** and survives a
DHCP lease change without an edit. Before 2026-08-24 it hard-set
`SERVER_IP=172.20.95.126`; when the Pi's lease moved to `.123`, the robot woke,
greeted, and silently had nowhere to send audio. The robot resolves mDNS via
`mdns4_minimal` in `/etc/nsswitch.conf`; `getent`/`avahi-resolve` do not exist
on this firmware, so the launcher uses
`python -c "import socket; socket.gethostbyname(...)"`. (Before,
the launcher set no `SERVER_IP` and `main.py` fell back to `nao/config.py`'s
stale default `172.20.95.106`, a dead host — the robot woke and silently did
nothing. `nao/config.py:23` still defaults to `.106`; the launcher override is
what fixes it.) The robot's `NAO_SHARED_SECRET` (embedded in the launcher) was
aligned to the Pi's `.env` so WS auth succeeds — **both must match** or the
robot connects and is rejected. Backups: `launch_nao_assist.sh.bak.20260729`.
Note `start_nao_assistant.sh` is a stale legacy launcher (points at `.106`) —
not the active one.

To relaunch `main.py` over SSH so it survives the session closing, use
`setsid bash /home/nao/launch_nao_assist.sh </dev/null >/tmp/nao_launch.log 2>&1 &`
(a plain `nohup … &` got SIGHUP'd and died). Verify with
`ssh nao 'pgrep -af "[m]ain\.py"'` and tail `/home/nao/nao_assist.log`.

**For development** (point the robot at your laptop instead of the Pi), run
`./run.sh` — it kills the running `main.py` and relaunches pointed at your LAN
IP with the laptop's secret.

Do **not** hand-roll the rsync. `run.sh` also clears `.pyc` (Python 2 prefers
stale bytecode, which makes edits silently no-op) and excludes `nao.log` —
a bare `rsync --delete nao/ ...` **deletes the robot's live log file**.

### Robot-side env knobs (forwarded by `run.sh`)

| Var | Default | Notes |
|---|---|---|
| `MIC_CHANNEL` | `left` | Which of NAO's 4 mics to record. Re-measured 2026-07-30 on one 4-channel recording of a single source: LEFT rms 10003, RIGHT 10067, FRONT 6614, REAR 6147 — LEFT/RIGHT are ~1.5× the other two, so the default is right. `left\|right\|front\|rear\|all`. |
| `ENGAGE_POSTURE` | `Sit` (code) | Posture on wake; `.env` currently overrides to `Stand`. `Sit` saves battery and mic noise; `none` disables. |
| `SPEAKING_GESTURES` | `1` | `0` stops arm micro-gestures during TTS. |
| `BOOT_GREETING` | `1` | NAO says `BOOT_GREETING_TEXT` and waves once it's ready to talk (after mic/TTS/LEDs/wake gates are built, just before `wsm.start()`). Latched once per process so the crash-retry loop doesn't re-greet. `0` disables. |
| `BOOT_GREETING_TEXT` | `Hello, I'm NAO. How can I help you?` | Reword without a code change. Spoken by **native** NAOqi TTS, which is unmuted and re-muted to 0.0 in a `finally` — if that restore ever breaks, the kid voice leaks into every ElevenLabs reply. |
| `SOUND_LOCALIZER` | `1` | `0` stops NAO aiming its head at whoever just spoke. **Currently `0` on the robot.** `sound_localize.py` has no TTS gating, so while NAO talks its own speaker is the loudest source in the room and it turns to face itself — it looks away precisely while answering. Turning it off leaves the face tracker holding eye contact. The real fix is to gate it during TTS so off-axis speakers still get a head-turn. |
| `FACE_TRACKER` | `1` | `0` stops ALTracker following a face with the camera. Left **on** — this is what keeps NAO looking at the user. |

**The robot's launcher currently sets** `SPEAKING_GESTURES=0 BOOT_GREETING=0
SOUND_LOCALIZER=0 FACE_TRACKER=1` (added 2026-08-24 on user request: no
greeting/wave, still hands while talking, eyes on the user). These live only in
`/home/nao/launch_nao_assist.sh` — **not in git**, so a reflash loses them.
Backups on the robot: `launch_nao_assist.sh.bak.*`.

**Mic capture gain IS adjustable, is not persistent, and has already failed
once** (2026-09-04). `ALAudioDevice.setInputVolume` does not exist on this
firmware — that part of the old note stands, and the `Can't find method`
line at boot is cosmetic. But the ALSA controls `Numeric Left mics` /
`Numeric Right mics` (range 0-88, 0.5 dB per step, 24 = 0.00 dB) do work,
they are **not persisted across reboot**, and on 2026-09-04 they were found
at **24/88 = 0.00 dB**. The effect: measured capture RMS **421** against a
speech baseline of ~**10000**, so Deepgram returned empty on every single
turn and NAO heard nothing — while every log looked healthy and the key,
model and network all tested fine. Restore with:

```bash
ssh nao 'amixer -c 0 sset "Numeric Left mics" 75; amixer -c 0 sset "Numeric Right mics" 75'   # +25.50 dB
```

**Gain is applied when the capture stream OPENS**, so setting it while
`main.py` runs changes nothing until the process restarts — that cost an
hour on 2026-09-04 when the mixer read 75 and the audio was still silent.
The launcher now restores it on every boot, but
`/home/nao/launch_nao_assist.sh` **is not in git**, so a reflash loses it
again. Backup: `launch_nao_assist.sh.bak.20260904`.

Diagnose by measuring, not guessing: copy
`/home/nao/recordings/_stream/stream.wav` off the robot and check per-second
RMS. Speech is ~10000; a flat ~400 floor is a dead mic, not a quiet room.
The old advice — "don't fix the mic gain, measure first" — was right about
measuring and wrong that there is nothing to fix.

### Server-side gotchas

- **CS Navigator was wired but unreachable until 2026-07-30.** `chatbot.py`
  imports `cs_navigator_search` (falling back to `vertex_search` only if that
  import fails — it doesn't), but `CS_NAVIGATOR_URL` was empty on the Pi, so
  every Morgan CS question returned the tool's polite *"I couldn't reach the CS
  Navigator just now"* instead of an answer. It degrades quietly, so nothing
  looked broken. Now set to
  `https://csnavigator-backend-900141432581.us-central1.run.app` (public
  `POST /chat/guest`, no token needed; the sibling `csnavigator-adk` service is
  403/private and isn't used). Verified live: prerequisites → "COSC 220 requires
  COSC 112 with a C or higher", 342 ms; faculty lookup 5.3 s.
- **Everything TTS speaks goes through `server/tts_text.py:to_speakable()`.**
  CS Navigator answers in Markdown (`**COSC 220**`, `*   Dr. Ali - Professor`)
  because it was built for a web UI, and agents emit Markdown too. Without
  stripping, NAO voices the asterisks and runs bulleted lists together with no
  pause. `_synth_for` normalizes before any provider sees the text. It only
  removes formatting — it never rewrites, reorders, or summarizes — and leaves
  standalone `*` alone so "2 * 3" still reads as arithmetic.
- **`FFMPEG_BIN`** must point at a real ffmpeg or `OPENAI_TTS_GAIN_DB` /
  `ELEVENLABS_TTS_GAIN_DB` are *silently skipped* and NAO is barely audible.
  No Homebrew on this Mac; the path in `.env`
  (`/opt/miniconda3/envs/nao-sagecbt/bin/ffmpeg`) is from a machine that had
  conda. **Verified absent on this Mac 2026-09-03** — no conda, no ffmpeg on
  PATH — so that value is dead here. Check `logs/server.log` for
  `ffmpeg unavailable`, and `[ -x "$FFMPEG_BIN" ]` before trusting it.
- **The Mac's `.venv` was rebuilt 2026-09-03** and now imports the server
  cleanly (`from server import app_ws`). Before that there was no `.venv` and no
  conda, so `run.sh` — which pins `.venv/bin/python` — could not start at all.
  The **system** `python3` still lacks `dotenv`/`fastapi`/`agents`: always go
  through `.venv/bin/python`, including for pytest, or you get
  `No module named 'fastapi'` and conclude the tree is broken when it isn't.
  `requirements.txt` now pins `openai-agents==0.13.6` + `openai<3` on purpose:
  the old `>=0.0.5` resolves to 0.22.0, which wants `openai>=3`, which litellm
  forbids — and the loser is `litellm`, so every `claude-*` model silently
  degrades to OpenAI at import.
- `USE_DEEPGRAM=1` / `USE_ELEVENLABS_TTS=1` in `.env` are **no-ops** without
  their keys/voice IDs — everything falls through to OpenAI. Docs record that
  Deepgram was benchmarked and rejected (slow on the CS network).
- `pytest-asyncio` is **not installed**, so every `@pytest.mark.asyncio` test
  **silently skips**. Drive async tests with `asyncio.run()` instead.
- ~32 tests fail on a clean tree (pre-existing). Diff against a stash before
  blaming your change.
- **`ELEVENLABS_STT_MODEL` must be `scribe_v1`.** `config.py` defaults it to
  `scribe_v2_realtime`, which the REST endpoint rejects with
  `400 unsupported_model`. The realtime *WebSocket* additionally 403s without a
  realtime entitlement, so `USE_ELEVENLABS_STT_REALTIME` defaults to `0` and the
  batch REST path is what actually runs.
- **`USE_DEEPGRAM=1` / `USE_ELEVENLABS_TTS=1` are no-ops without their keys** —
  everything falls through to OpenAI. Read the boot log, not the flag.
- **ElevenLabs keys are permission-scoped — a REST test passing does NOT mean
  the live path works** (learned 2026-07-29). The server uses the **WebSocket
  streaming** TTS and **Scribe STT**, which need the `speech_to_text` and
  streaming scopes. A key scoped for REST `text_to_speech` only returns `200` on
  `POST /v1/text-to-speech` yet 401s (`invalid_api_key`) on the WS stream and STT
  — so both silently fall back to OpenAI (`elevenlabs_synth_returned_none`,
  `[transcribe] elevenlabs returned empty; falling back to whisper`) and every
  reply wastes ~2s on the failed attempt first. Verify a new key against the
  paths the server actually uses: `synthesize_stream` (WS) and `POST
  /v1/speech-to-text`, not just REST synth.
- **Dead env vars are a recurring trap.** A `*_MODEL` var only works if
  something reads `config.X`. `CBT_MODEL` / `GROUNDING_MODEL` existed for months
  and were read by nothing (both coaches read `THERAPIST_MODEL`) — setting them
  looked like it worked. Before trusting a knob:
  `grep -rn "config\.YOUR_VAR" server --include="*.py"`.
- **`CRISIS_MODEL` is badly named** — it drives the crisis classifier *and*
  `identify_distortion` / `suggest_reframe` (the clinical core of the CBT flow)
  *and* memory rollups. Changing it to tune CBT also changes the safety gate.
  Vision is separate (`VISION_MODEL`).
- **`SAFETY_MODEL_CLAUDE` / `SAGE_SAFETY_PROVIDER` do nothing at
  `SAGE_TOPOLOGY=passthrough`** (the default). They configure
  `topologies/safety_agent.py`, which only runs under the SAGE research
  topologies. The live gate is `safety.py` + `CRISIS_MODEL`.

### API keys — the failure that looks like a broken robot

**Check which slot each key is in before debugging anything else.** On
2026-09-03 the robot "stopped responding" because the **ElevenLabs key had been
pasted into `OPENAI_API_KEY`**, and the real OpenAI key was absent. Nothing in
the logs says "wrong variable" — you get `invalid_api_key` from OpenAI and a
silent fallback everywhere else.

Tell the keys apart by prefix:

| Provider | Prefix | Note |
|---|---|---|
| OpenAI | `sk-` / `sk-proj-` | **hyphen** |
| ElevenLabs | `sk_` | **underscore** — this is the one that gets mis-pasted |
| Anthropic | `sk-ant-api03-` | |
| Deepgram | 40 hex chars, no prefix | |

Why one swapped key killed the whole robot: every agent pointed at OpenAI
models, so the brain 401'd on every turn; and `ELEVENLABS_API_KEY` being empty
sent TTS to the OpenAI fallback, which 401'd too. Deepgram still worked, so
**NAO heard you and then went silent** — the exact symptom. Verify keys against
the live endpoints rather than trusting the file; a 401 body names the provider
that rejected it.

**A valid ElevenLabs key is not enough — a voice ID is also required.**
`elevenlabs_tts.is_available()` (`server/elevenlabs_tts.py`) returns `False`
when `_resolve_default_voice_id()` is empty, so with `ELEVENLABS_VOICE_GIRL`
unset the whole provider is skipped and TTS silently falls back to OpenAI. Keys
are also permission-scoped: a key without `voices_read` **cannot list voices**
(401), so you cannot discover an ID from the API — the working IDs are recorded
in `.env`, and losing them means losing the voice.

`OPENAI_AGENTS_TRACE=1` uploads traces to OpenAI on every turn. With a dead or
absent OpenAI key that is a wasted round-trip per turn; set it to `0` on any
deploy not using OpenAI.

### Known bugs

- **CBT classifier could not say "no distortion" — FIXED 2026-07-30 (`ebe5930`).**
  The prompt said *"Choose exactly ONE from"* the ten labels in `_DISTORTIONS`
  with no none-of-these option, so a balanced thought got a label anyway:
  *"I studied hard, I did well, and I feel good about it."* →
  `magnification/minimization`, while the model's own explanation read
  *"there's no distortion here."* It knew, and the prompt left it no way to say
  so — NAO told students their balanced thinking was a cognitive distortion.
  `NO_DISTORTION = "none"` is now an allowed answer, and the prompt states that
  a thought can be sad, worried, or negative about a genuinely bad situation
  and still not be distorted (grief and disappointment were the cases most at
  risk of being labelled). Downstream, `none` is a real answer, not a missing
  one: nothing is written to `thought_records` (it would read back next session
  as a distortion the student never had), `suggest_reframe` returns `[]` rather
  than asking for alternatives to a thought exhibiting "none", and the coach
  prompt says to acknowledge it warmly and skip the reframe step.
  `_is_no_distortion()` normalizes `None` / `no distortion` / `n/a` / `""`.
  Verified against live `claude-opus-5`: three healthy thoughts → `none`;
  catastrophizing and mind-reading still named correctly.
  Tests: `server/tests/test_no_distortion.py`.
- **Whisper fallback hallucinated whole sentences — FIXED 2026-07-30.** STT fell
  through to OpenAI whenever Deepgram returned empty, and Whisper never returns
  empty on weak audio: it invents. On the robot's own mic recording the same
  clip gave Deepgram `''` and Whisper `'それではまた。'`; live it produced
  "Hallo." for "Hello" twice. Deepgram now owns the whole path (nova-3 → nova-2
  retry → give up). `STT_ALLOW_OPENAI_FALLBACK=1` restores the old chain.
- **`keywords` is a hard 400 on Nova-3 — FIXED 2026-07-30.** `deepgram_asr` sent
  the legacy `keywords` param; Nova-3 wants `keyterm` and rejects the old name
  outright. Because the adapter maps any non-200 to `""` and `transcribe()`
  reads `""` as "this provider had nothing", **every** Deepgram request 400'd
  silently and Whisper did all the work while the logs showed Deepgram healthy.
  Term boosting is now chosen by model family. Watch for this shape of bug: a
  provider that looks enabled, costs a round-trip, and never serves a request.
- **Silero VAD shared model — FIXED 2026-07-29 (commit `7c7490b`).** `vad_silero._model`
  was a process-wide singleton feeding a stateful Silero v5 RNN from every
  session's `StreamingSilero`. Concurrent sessions interleaved audio into one
  hidden state and one session's `reset()` wiped another's; confidence collapsed
  to ~0 and turns were rejected (`reject_reason=no_voice` / `silero_no_speech`)
  on clearly-audible PCM, worsening as WS reconnects piled up. Fix: each
  `StreamingSilero` now builds its own model via `_new_stream_model()` (~1-2 MB;
  batch `has_voice`/`trim_silence` still share the singleton — stateless use).
  Regression test: `server/tests/test_silero_per_session.py`. If `no_voice`
  rejections ever return, first check that fix is still present, then restart.
- **The crisis gate only consults the LLM when a keyword matches.** No match in
  `_SOFT_TRIGGERS` → returns `clean` without any model call. It matches
  contracted forms only, so *"I do not want to be here anymore"* misses (the
  list has `"don't want to be here"`), as do *"I want to disappear"* and
  *"I feel like a burden"*. Upgrading `CRISIS_MODEL` does not help — the model
  never sees those messages.
- **The therapy lane does not persist.** Routing is recomputed per turn from
  keywords, so a follow-up phrased without a trigger word silently drops from
  `therapist` back to `chat`, losing the CBT tools and coach handoffs
  mid-conversation. The router also costs a second sequential model call
  (measured 5.7s therapist alone vs 8.9s via router) even though
  `pick_initial_agent` already matched the emotional keyword in Python.

## Identity / face recognition

Two stores, joined only by a **name string**:

- **Robot:** `/home/nao/.local/share/vision/facerecognition/default` — OMRON
  OKAO (`OKAOFR70`) feature templates, ~1.4 KB for several faces. Never images;
  you cannot reconstruct a face from it. Inspect with
  `ALFaceDetection.getLearnedFacesList()`.
- **Server:** `users` table in `server/nao.db` (`face_id` → `display_name`).
  `face_id` holds the **learned name**, not a number.

`ALFaceDetection` extra_info: `[0]` is a transient internal id renumbered on
every detection (same person was 12, then 18); `[2]` is the recognised name.
`wake_state.identity_key_for_face()` prefers `[2]` and falls back to `[0]`.

Say **"remember me as X"** to trigger the `learn_face` fast-path — it teaches
the robot's face DB *and* upserts the `users` row (`_emit_motion` →
`memory.ensure_user`). Both must happen or recognition silently never works.

## Debugging the robot

- **Never diagnose reachability with `ping`.** The gateway and the robot
  ignore/drop ICMP; "100% packet loss" told us the robot was dead three times
  while SSH was wide open. Test the port you actually need:
  `nc -z -G 4 <ip> 22`.
- **Read the *failure mode*, not just "it didn't connect."** The three outcomes
  mean completely different things and point at different fixes:

  | Result | Meaning | Where the fault is |
  |---|---|---|
  | `Connection refused` | Packet arrived, host sent RST | **Host is UP**; no service listening |
  | timeout / no route | Nothing came back | Host off, wrong IP, or network blocking |
  | ARP `(incomplete)` | No L2 reply | Not on this subnet at all |

  A refusal is a *reply* — it proves the host is alive and the network path is
  clean in both directions, which rules out firewalls, client isolation, VLANs
  and wrong IPs in one shot. `arp -a | grep -v incomplete` lists what is
  genuinely on the subnet.
- **Identify an unknown device by its MAC vendor before trusting an IP.**
  `curl -s https://api.macvendors.com/<mac>` — the Pi returns *Raspberry Pi
  Trading Ltd*, and NAO returns Aldebaran/SoftBank. On 2026-09-03 `.env`'s
  `NAO_IP=172.20.95.101` resolved to a **WNC Corporation** device (a phone) that
  had taken the robot's old lease. Randomised MACs (locally-administered, second
  hex digit `2`/`6`/`a`/`e`) are phones and laptops, never the robot or the Pi.
- **Zero open ports on a live host = booted, but userspace never started.**
  On 2026-09-03 the Pi answered ARP and refused every one of ports 1-1024 plus
  the usual alternates — kernel and networking up, **not one service running**,
  including `sshd` and `nao-server`. A healthy box always has something
  listening, so this is not an SSH problem and no remote fix exists: there is no
  door open. It needs console access (monitor + keyboard) or the SD card in a
  Mac. Diagnose at the console with `systemctl --failed`,
  `systemctl status ssh.socket nao-server`, and `mount | grep ' / '` — a root
  filesystem mounted `ro` means SD-card damage (usually an unclean power-off),
  and the fix is card recovery, **not** restarting services. Remember Ubuntu
  uses socket activation: start `ssh.socket`, not `ssh.service`, and
  `enable --now` it or it will not survive the next reboot.
- **`main.py`'s structured logs are not in either obvious log file.** `logger.py`
  writes to a dated JSONL under `~/nao_assist/logs/` — so `boot_start`,
  `wake_engaged`, `boot_greeting_spoken` and friends appear in **none** of
  `nao_assist.log` (the `tee` target, mostly `print()` output) or
  `nao_assist/nao.log`. Grepping those two for a boot event finds zero hits and
  looks like the event never fired. Read this instead:
  `ssh nao 'tail -20 ~/nao_assist/logs/nao_$(date +%F).jsonl'`
- Robot logs: `/home/nao/nao_assist/nao.log` (recreated by `run.sh` on launch).
- `NAO_SHARED_SECRET` must not be inline in commands (the safety classifier
  blocks plaintext secrets, and `ps` exposes it on the robot). Source an env file.
- `pkill -f main.py` from your laptop can match your own SSH command and kill
  the connection; use `pgrep -af "[m]ain\.py"`.
- Wake gates are **face and touch only — there is no voice wake word.** Eyes:
  gray=idle, soft blue=aware, solid blue=engaged, cyan=listening. Touch while
  ENGAGED means *barge-in*; rear-head touch means *stop*.
- The face loop can wedge on `ALMemory read error` spam and stop streaming audio
  entirely (zero `audio_chunk`) while still answering touch. `./run.sh` clears it.
- The robot's battery drains fast; standing drains it faster. Keep it charged.
- Its deployed tree can diverge from local `nao/`. Diff before `--delete`.

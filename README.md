<div align="center">

# NAO‑SAGECBT

**A real‑time, embodied, multi‑agent voice assistant for the NAO H25 humanoid.**
*Built at Morgan State University · Department of Computer Science.*

<p>
  <img alt="License"   src="https://img.shields.io/badge/license-MIT-1f6feb?style=for-the-badge">
  <img alt="Python"    src="https://img.shields.io/badge/python-2.7%20%7C%203.11+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="FastAPI"   src="https://img.shields.io/badge/FastAPI-WS-009688?style=for-the-badge&logo=fastapi&logoColor=white">
  <img alt="OpenAI"    src="https://img.shields.io/badge/OpenAI-Agents%20SDK-412991?style=for-the-badge&logo=openai&logoColor=white">
  <img alt="ElevenLabs" src="https://img.shields.io/badge/ElevenLabs-Flash%20v2.5-000000?style=for-the-badge">
  <img alt="Deepgram"  src="https://img.shields.io/badge/Deepgram-Nova--2-13EF93?style=for-the-badge">
  <img alt="GPT‑4o"    src="https://img.shields.io/badge/Vision-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white">
  <img alt="NAO"       src="https://img.shields.io/badge/NAO-H25-FF6F00?style=for-the-badge">
  <img alt="Research"  src="https://img.shields.io/badge/Research-SAGE--CBT-7B1FA2?style=for-the-badge">
</p>

<sub>Aayush Shrestha · Advised by Dr. Shuangbao "Paul" Wang · Department of Computer Science, Morgan State University</sub>

<br/>

<img src="docs/img/nao.jpg" width="380" alt="NAO H25 robot">

</div>

---

## TL;DR

A NAO humanoid that **listens, sees, thinks, talks, and moves like a person.** A student walks up, NAO recognizes their face, greets them by name, answers questions about courses, weather, or feelings, gestures while talking, turns its head toward whoever speaks, and runs a hardcoded crisis gate that bypasses the LLM entirely on suicidal ideation.

End‑to‑end target: **&lt; 800 ms p50** mouth‑close to first audio.

> **For people new to this repo:** read this README, then [`docs/DECISIONS`](docs/DECISIONS.md) (how problems were navigated), then [`docs/PRD_v2`](docs/PRD_v2.md) (full spec). Index of all docs in [`docs/INDEX`](docs/INDEX.md).

---

## How a turn works

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant N as NAO
    participant S as Server
    participant A as Agents SDK
    participant T as ElevenLabs

    U->>N: speech
    N->>S: WS audio_chunk frames
    S->>S: VAD plus Silero gate
    S->>S: STT (Deepgram nova-3, nova-2 retry)
    S-->>S: Crisis gate (988 hard)
    S-->>S: Motion trigger short-circuit
    S->>A: Router to specialist
    A-->>S: streaming token deltas
    S->>S: Sentence chunker
    S->>T: per-sentence synth (parallel)
    T-->>S: MP3 chunks
    S->>N: WS audio_chunk plus action frames
    N->>N: ALAudioPlayer plus gestures
    N-->>U: speech plus motion
    N->>N: post-playback drain
    N->>S: mic_resumed (clean restart)
```

---

## System architecture

```mermaid
graph LR
    classDef u fill:#0d47a1,stroke:#0d47a1,color:#fff
    classDef nao fill:#1565c0,stroke:#1565c0,color:#fff
    classDef srv fill:#2e7d32,stroke:#2e7d32,color:#fff
    classDef ai fill:#412991,stroke:#412991,color:#fff
    classDef ext fill:#ef6c00,stroke:#ef6c00,color:#fff
    classDef gate fill:#b71c1c,stroke:#b71c1c,color:#fff
    classDef db fill:#6a1b9a,stroke:#6a1b9a,color:#fff

    USER([User]):::u

    subgraph NAO["NAO H25 naoqi 2.7"]
        WAKE["Wake state machine"]:::nao
        WSC["WS client"]:::nao
        AUD["Audio module<br/>fragment recorder"]:::nao
        TTS["Stream TTS<br/>MP3 to WAV to ALAudioPlayer"]:::nao
        EXEC["Action worker<br/>gestures, dances, motors"]:::nao
        SND["Sound localizer<br/>plus ALTracker face mode"]:::nao
        BRAIN[("Brain cache<br/>64 KB")]:::db
    end

    subgraph SRV["Server FastAPI plus uvicorn"]
        WSS["WebSocket ws endpoint"]:::srv
        SAFE["Crisis gate"]:::gate
        MTR["Motion trigger"]:::srv
        VAD["Silero VAD"]:::srv
        STT["STT lane<br/>Deepgram, Whisper, Scribe"]:::srv
        ROUTER["Router"]:::ai
        CHAT["Chat lanes<br/>pure and embodied"]:::ai
        CHATBOT["Chatbot"]:::ai
        SKILLS["Skills"]:::ai
        THER["Therapist plus CBT, Grounding, MI"]:::ai
        VIS["Claude vision<br/>lazy, fresh per question"]:::ai
        SES[("SQLite session<br/>plus user_prefs")]:::db
        OBS["metrics plus JSONL"]:::srv
    end

    EL[("ElevenLabs Flash v2.5")]:::ext
    OAI[("OpenAI Agents SDK")]:::ext
    CSN[("CS Navigator API<br/>Cloud Run")]:::ext

    USER -->|voice| WAKE --> WSC
    USER -->|face/proximity| WAKE
    AUD --> WSC -->|audio_chunk| WSS
    WSS --> VAD --> STT
    STT --> SAFE
    SAFE -.crisis.-> WSS
    SAFE --> MTR
    MTR -.short-circuit.-> WSS
    MTR --> ROUTER
    ROUTER --> CHAT & CHATBOT & SKILLS & THER
    CHATBOT --> CSN
    THER -.vision.-> VIS
    CHAT & CHATBOT & SKILLS & THER --> OAI
    OAI -->|stream tokens| WSS
    WSS --> EL
    EL -->|MP3| WSS
    WSS -->|audio_chunk + action| WSC
    WSC --> TTS --> EXEC
    SND --> EXEC
    BRAIN <-->|sync| WSS
    SES -.persistence.-> ROUTER
    OBS -.observability.-> WSS
```

---

## Multi‑agent graph

```mermaid
graph TD
    classDef gate fill:#b71c1c,stroke:#b71c1c,color:#fff
    classDef route fill:#0d47a1,stroke:#0d47a1,color:#fff
    classDef spec fill:#1565c0,stroke:#1565c0,color:#fff
    classDef sub fill:#7b1fa2,stroke:#7b1fa2,color:#fff
    classDef tool fill:#2e7d32,stroke:#2e7d32,color:#fff

    T([turn]) --> CG{crisis check}:::gate
    CG -- positive --> H["988 hotline<br/>fixed reply, no LLM"]:::gate
    CG -- clean --> MT{motion trigger}:::route
    MT -- match --> ACT["short-circuit action"]:::route
    MT -- no match --> CSD{obvious CS question?}:::route
    CSD -- yes --> CSDIR["straight to CS Navigator<br/>no model call"]:::route
    CSD -- no --> R["router<br/>Claude Haiku 4.5"]:::route

    R --> CH["chat<br/>pure and embodied"]:::spec
    R --> CB["chatbot"]:::spec
    R --> SK["skills"]:::spec
    R --> TH["therapist<br/>Claude Sonnet 5"]:::spec

    TH --> CBT["cbt_coach<br/>thought records"]:::sub
    TH --> GR["grounding_coach<br/>5-4-3-2-1, box, scan"]:::sub
    TH -. opt .-> MI["mi_coach<br/>OARS"]:::sub

    CH & TH -.-> NA[("nao_actions")]:::tool
    CB & CH & SK & TH -.-> CSN[("cs_navigator")]:::tool
    SK -.-> ST[("skills_tools")]:::tool
    TH & CBT & GR -.-> EM[("emotion plus memory")]:::tool
    CH -.-> LF[("learn_face")]:::tool
```

Models as deployed (September 2026). Any `*_MODEL` setting can name a Claude
or an OpenAI model; the code defaults are still OpenAI ids, the live `.env`
files point at Claude.

| Agent | Role | Live model |
|---|---|---|
| **router** | triage + handoff (sensory grounding rule, never denies senses) | Claude Haiku 4.5 |
| **chat (pure)** | fast everyday lane; only tool is CS Navigator | Claude Haiku 4.5 |
| **chat (embodied)** | gestures + actions + face learn | Claude Haiku 4.5 |
| **chatbot** | Morgan CS answers via CS Navigator | Claude Sonnet 5 |
| **skills** | time, weather, timers, todos | Claude Haiku 4.5 |
| **therapist** | empathy + handoffs | Claude Sonnet 5 |
| **cbt_coach** | Beck thought record (one step per turn) | Claude Sonnet 5 |
| **grounding_coach** | 5‑4‑3‑2‑1, box breathing, body scan | Claude Sonnet 5 |
| **mi_coach** | Motivational Interviewing (OARS) — experimental | Claude Sonnet 5 |
| **crisis** | safety classifier, CBT distortion check, memory rollups | Claude Opus 5 |
| **vision** | camera read, lazy, per question | Claude Sonnet 5 |

Every agent that answers users carries one shared rule
(`server/agents/_cs_rule.py`): Morgan State CS questions are answered from CS
Navigator, never from the model's memory. Obvious CS questions (a course code
like "COSC 220", "who teaches", "prerequisite") skip the models entirely and
go straight to CS Navigator.

---

## Wake state machine

```mermaid
stateDiagram-v2
    direction LR

    [*] --> IDLE
    IDLE --> AWARE: face detected (conf >= 0.35)
    AWARE --> IDLE: face lost or 8 s timeout
    AWARE --> ENGAGED: mutual gaze 1.5s, proximity, speech, or hey NAO
    ENGAGED --> LISTENING: chime + WS open
    LISTENING --> SPEAKING: TTS playing
    SPEAKING --> LISTENING: playback drained + grace
    LISTENING --> AWARE: face lost > 5 s
    SPEAKING --> LISTENING: barge-in (touch / speech onset)

    note right of AWARE
        Robot is aware but silent.
        No greeting, no chime —
        prevents passerby false-wakes.
    end note

    note right of LISTENING
        Eyes cyan - gaze drift
        Mic open - streaming PCM
    end note
```

---

## Voice + mic lifecycle (the hard part)

```mermaid
flowchart TD
    classDef a fill:#0d47a1,stroke:#0d47a1,color:#fff
    classDef ok fill:#2e7d32,stroke:#2e7d32,color:#fff
    classDef bad fill:#b71c1c,stroke:#b71c1c,color:#fff

    START([reply ready]) --> ENQ["stream_tts.enqueue<br/>per-sentence MP3"]:::a
    ENQ --> PLAY["ALAudioPlayer<br/>blocking_play_start"]:::a
    PLAY --> DONE{"queue empty?"}:::a
    DONE -- no --> NEXT["next chunk"]:::a --> PLAY
    DONE -- yes --> DRAIN["local_tts_queue_empty"]:::ok
    DRAIN --> SETTLE["wait 800 ms<br/>speaker cone settle"]:::a
    SETTLE --> OPEN["gate False<br/>fresh stream.wav"]:::ok
    OPEN --> RESUME["push mic_resumed"]:::ok
    RESUME --> LISTEN([mic live])

    ECHO{"server self_echo<br/>reject?"}:::bad
    LISTEN --> ECHO
    ECHO -- yes --> CLOSE["gate True<br/>stop recorder"]:::bad
    CLOSE --> RESETTLE["250 ms settle"]:::bad
    RESETTLE --> RESTART["gate False<br/>fresh stream.wav"]:::ok
    RESTART --> LISTEN
```

The old design opened the mic on the server's `tts_ended` frame — but that only means the server stopped *sending* audio. The robot's local queue could still play for 5–8 more seconds, and the mic would record NAO's own speaker output. See [`DECISIONS § D8`](docs/DECISIONS.md#d8-mic-lifecycle-tts-ended-vs-playback-drained).

---

## Action dispatch (don't block the recv thread)

```mermaid
sequenceDiagram
    participant Server
    participant Recv as WS recv loop
    participant Q as Action queue
    participant Worker as Action worker
    participant NAO as ALMotion plus ALBehavior

    Server->>Recv: action frame
    Recv->>Q: put_nowait
    Recv-->>Server: returns immediately
    Worker->>Q: get
    Worker->>NAO: dispatch, blocking up to 15 s
    NAO-->>Worker: done
    Worker->>Q: get next

    Note over Server,NAO: barge-in or crisis or shutdown
    Server->>Recv: control barge_in
    Recv->>Worker: drain queue
    Recv->>NAO: stopAllBehaviors
    NAO-->>Worker: cancelled
```

See [`DECISIONS § D9`](docs/DECISIONS.md#d9-action-dispatch-on-recv-thread-vs-worker), [`§ D10`](docs/DECISIONS.md#d10-blocking-vs-non-blocking-behavior-calls).

---

## Onboarding flow (face learn)

```mermaid
sequenceDiagram
    participant U as User
    participant W as Wake SM
    participant N as NAO
    participant S as Server
    participant A as Chat agent

    U->>W: walks into camera view
    W->>W: AWARE to ENGAGED, gate fires
    N->>S: control session_open
    N->>S: image (reason=session_open)
    N-->>U: "Heads up - my camera is on..."
    par face scan (3 s daemon)
        N->>N: ALFaceDetection.recognize
        alt known face
            N->>S: user_identified, name Aayush, recognized true
        else unknown
            N->>S: user_identified, name null, face_visible true
        end
    end
    U->>S: "hey can you see me"
    S->>A: USER name=Aayush returning=true + NAO_VISION + transcript
    alt known
        A-->>U: "Welcome back, Aayush!"
    else unknown
        A-->>U: "I see you. What's your name?"
        U->>S: "I'm Aayush, remember me"
        S->>A: route to chat
        A->>N: tool: learn_face(name=Aayush)
        N->>N: ALFaceDetection.learnFace
        A-->>U: "Got it, Aayush. Pleasure to meet you."
    end
```

---

## Highlights at a glance

<table>
<tr><td valign="top" width="50%">

#### Voice loop
- Streaming TTS with sentence chunker
- 3 STT backends, hot‑swap A/B
- Flash v2.5 voice profiles
- 3‑layer self‑echo defense
- Post‑playback mic resume waiter
- Recorder restart on echo reject

#### Multi‑agent
- OpenAI Agents SDK + handoffs
- Pure & embodied chat lanes
- CBT + Grounding + MI sub‑coaches
- Memory injection (recaps, themes)
- Per‑user voice profile in SQLite

#### Vision
- Lazy GPT‑4o, trigger‑phrase gated
- Fresh per question (no stale cache)
- `[NAO_VISION]` prompt block
- Server‑side image stash, never on robot

</td><td valign="top" width="50%">

#### Embodiment
- 47 native gesture intents
- 35‑style dance map
- `follow-me` Choregraphe pack
- Sound‑source localization
- ALTracker face mode
- Action worker thread
- `stopAllBehaviors` cancellation

#### Wake & onboarding
- Hybrid face‑first + keyword fallback
- 5‑state wake SM with LED cues
- Touch‑sensor barge‑in
- Face learn flow + persistent DB
- Camera consent + privacy LED

#### Safety + observability
- Hardcoded 988 crisis gate
- Re‑toned hotline reply
- `structlog` JSON, Prometheus, Grafana
- Robot‑side rotating JSONL
- 22 latency labels, 10 dash panels

</td></tr></table>

---

## Quick start

```mermaid
flowchart LR
    classDef step fill:#0d47a1,stroke:#0d47a1,color:#fff

    A["clone repo"]:::step --> B["python -m venv .venv<br/>pip install -r server/requirements.txt"]:::step
    B --> C["cp .env.example .env<br/>fill OPENAI_API_KEY etc."]:::step
    C --> D["./run.sh"]:::step
    D --> E(["uvicorn on 5050<br/>plus NAO main.py"])
```

```bash
git clone https://github.com/theaayushstha1/nao-sagecbt.git
cd nao-sagecbt
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt
cp .env.example .env             # fill keys
./run.sh                          # deploy to robot + boot uvicorn + tail logs
./run.sh stop                     # tear down
```

`run.sh` rsyncs `nao/` to `/home/nao/nao_assist/`, boots `uvicorn server.app_ws:app` on `:5050`, launches `main.py` on the robot, and tails a signal‑filtered combined log (set `RAW_LOGS=1` to see everything).

---

## Always‑on deployment (Raspberry Pi server)

Dev mode runs the server on your laptop via `./run.sh`. For demos and 24/7 availability we now also host the server on a Raspberry Pi 4 (Ubuntu Server 24.04) so the robot is reachable without anyone's laptop being online.

```mermaid
flowchart LR
    classDef box fill:#0d47a1,stroke:#0d47a1,color:#fff
    NAO["NAO robot<br/>172.20.95.127<br/>Choregraphe autostart<br/>→ launch_nao_assist.sh"]:::box
    PI["Raspberry Pi 4<br/>172.20.95.106<br/>systemd nao-server.service<br/>uvicorn server.app_ws:app"]:::box
    NAO -- "WebSocket :5050<br/>audio_chunk / control / image" --> PI
    PI -- "audio_chunk MP3 / action / transcript" --> NAO
```

- **Robot side**: power on → NAOqi boots → Choregraphe default behavior `nao-therapy-autostart/behavior_1` fires → `/home/nao/launch_nao_assist.sh` mutes NAO's native ALTextToSpeech (ElevenLabs is the only voice), disables autonomous life, and launches `python /home/nao/nao_assist/main.py`. Nothing to do by hand.
- **Server side**: Pi runs `nao-server.service` (enabled at boot). Server pulls latest `main` on `git pull && sudo systemctl restart nao-server`.

```bash
# Deploy code updates to the Pi server
ssh -t nao@172.20.95.106 'cd ~/nao-sagecbt && git pull && sudo systemctl restart nao-server && sudo systemctl is-active nao-server'

# Deploy code updates to the robot
./run.sh deploy-only        # rsyncs full nao/ folder + clears .pyc

# Restart main.py on the robot
ssh -t nao@172.20.95.127 'pkill -9 -f main.py; sleep 2; bash /home/nao/launch_nao_assist.sh'

# Tail server logs
ssh -t nao@172.20.95.106 'sudo journalctl -u nao-server -f'
```

The robot's `~/.bash_profile` previously auto‑started a separate Feb‑2026 project that collided with the new app over the mic and speaker (the "two main.py at different paths" bug); that autostart is now neutralized — see `~/.bash_profile.old_nao_chat_backup.*` for the original.

---

## Recent changes (September 2026)

- **Looks at whoever is talking** (`nao/awareness.py`): the head turns toward a
  voice, holds there, and the face tracker locks on; sounds from behind NAO
  are ignored and it recentres when nobody is found. Sound direction is now
  parsed with the NAOqi 2.8 layout (it read as 0° before).
- **Autonomous Life stays off**: it re-enabled itself after boot and locked
  the head on one person; a guard now keeps it disabled.
- **A new person is a new person**: after each turn toward a voice NAO
  re-checks the face; weak matches (< 0.75) count as unknown, so a stranger is
  no longer called by the previous person's name.
- **Starts on its own**: a few seconds after power-up NAO says "Heads up, my
  camera is on for this conversation." and "Hi, I'm NAO. How can I help you
  today?" No tap needed. Stands while chatting, sits after 90 s of quiet.
- **Faster CS answers**: obvious CS questions go straight to CS Navigator
  with a "Let me check that." filler (answer in ~4.5–7 s, was 10–15 s).
- **Camera off is real**: "stop watching me" persists and stops photos; NAO no
  longer turns its own camera off by hearing its announcement.
- **Mute replies**: "Okay, my lips are closed now." / "I'm back."
- **One voice**: ElevenLabs only; the OpenAI backup voice is removed. OpenAI
  tracing is off.
- **Live dashboard**: `http://<pi-ip>:5050/dashboard` (no login) shows robot
  and Pi status, battery, a live check of every outside service, and the
  conversation (support conversations hidden). A hosted copy can run on
  Vercel from `vercel-dashboard/`.

## Recent changes (May 2026)

- **Pi migration**: server now runs on a Raspberry Pi 4 via systemd; robot autostarts via Choregraphe default behavior. The robot can be unboxed, powered on, and talked to without any laptop.
- **Mic‑stall watchdog** (`nao/audio_module.py`): detects when `ALAudioRecorder` wedges (WAV stops growing) and restarts the recorder after 2 s. Prevents mid‑conversation silence.
- **VAD adaptive threshold** (`server/vad_silero.py`): Silero VAD threshold lowered to 0.15 with an adaptive floor at 0.05 so NAO actually hears quiet voices instead of rejecting every turn as `no_voice`. Per‑second `[silero_trace]` log line for tuning.
- **Durable `had_speech`** (`server/app_ws.py`, `server/vad_silero.py`): per‑utterance flag captures speech bursts that end on a quiet syllable, so the end‑of‑utterance arbiter doesn't drop the turn.
- **TTS sentence dedupe** (`server/app_ws.py`): prevents the agent from speaking the same sentence twice when an LLM tool call interrupts streaming.
- **Native voice locked off** (`nao/main.py`, `nao/stream_tts.py`, `nao/ws_client.py`): `ALTextToSpeech.setVolume(0.0)` is now pinned at every collision point (boot, TTS playback, announcer fillers gated behind `ENABLE_NATIVE_FILLER=1`). NAO's kid voice never leaks.
- **`learn_face` fast‑path** (`server/motion_trigger.py`): "remember me as X" / "save my face as X" / "my name is X" short‑circuit before any agent routes, so face learning works from any mode (router, chat, therapy, morgan).
- **Persistent therapy memory** (`server/session.py`, `server/tools/emotion.py`, `server/memory.py`): two new SQLite tables — `mood_log` and `thought_records` — populated by the `log_emotion`, `identify_distortion`, and `suggest_reframe` tools. The system preamble now surfaces a `Therapy memory` section with the latest mood, a 5‑entry trajectory, and the last CBT thought record so the therapist can open the next session with continuity.
- **Proactive name use** (`server/agents/therapist.py`, `server/agents/chat.py`): both agents weave the user's name into ~1 in 3 replies (greetings, transitions, validations, peaks). Memory‑aware first turn for the therapist references recent mood / thought records.

---

## Talking to NAO

| Say | Triggers |
|---|---|
| (nothing, it greets you on power-up) | Boot engage: camera line + "Hi, I'm NAO. How can I help you today?" |
| Step into view / touch the head | Wake state machine (there is no voice wake word) |
| "Wave at my friend" | `wave_hand` action |
| "Do the kung fu" | `dance(style='kungfu')` -> `KungFu_1` Choregraphe pack |
| "Follow me" / "Track me" | `follow_movement` (`follow-me` pack) |
| "Stop following me" / "Freeze" | `stop_follow` |
| "What am I wearing?" / "Can you see me?" | Lazy vision call (Claude Sonnet 5) |
| "Switch to a man voice" / "Use my voice" | Voice profile flip (per‑user persisted) |
| "Remember me as Aayush" | `learn_face(name='Aayush')` |
| "Stop watching me" / "You can watch me again" | Camera off / on, remembered per person |
| "Be quiet" / "Mute" | "Okay, my lips are closed now." then silent |
| "You can talk" / "Unmute" | "I'm back." |
| "Set a 10‑minute timer" | Skills agent |
| "I'm anxious about finals" | Therapist agent |
| "What classes does Morgan offer in spring?" | CS Navigator (chatbot agent) |
| "Who teaches COSC 220?" | CS Navigator directly, no model call |

Tap NAO's head sensors at any time to **barge in** — TTS stops within ~200 ms, current behavior cancels via `stopAllBehaviors()`.

---

## Performance

| Metric | Target | Where measured |
|---|---|---|
| `e2e_user_to_first_audio` p50 | &lt; 800 ms | `phase_ms` in `turn_complete` log |
| `e2e_user_to_first_audio` p95 | &lt; 1.2 s | Prometheus histogram |
| `tts_synth_first_chunk` | &lt; 500 ms | Flash v2.5 streaming |
| `vision_call` | ~1.5 s | only on visual triggers |
| Barge‑in stop time | &lt; 200 ms | `tts_player.stop()` + `stopAllBehaviors` |

`/metrics` exposes histograms per phase. Grafana JSON in [`server/dashboards/grafana_voice.json`](server/dashboards/grafana_voice.json).

---

## Repo layout

```
nao-sagecbt/
├── nao/                      Python 2.7 — runs on the robot
│   ├── main.py               entry: wake SM + session controller
│   ├── ws_client.py          long-lived WebSocket session
│   ├── audio_module.py       ALAudioRecorder fragment streamer
│   ├── stream_tts.py         MP3 -> WAV -> ALAudioPlayer + ffmpeg loudness
│   ├── wake_state.py         IDLE -> AWARE -> ENGAGED -> LISTENING -> SPEAKING
│   ├── awareness.py          Look at the speaker, keep Autonomous Life off
│   ├── sound_localize.py     ALSoundLocalization (legacy head path)
│   ├── leds.py               eye LED helpers
│   ├── idle_motion.py        background breathing + gaze drift
│   └── utils/
│       ├── nao_execute.py    dispatcher: gestures, dances, motors
│       ├── face_naoqi.py     face recognition + learning
│       ├── brain.py          64 KB local identity cache
│       └── camera_capture.py per-turn JPEG snap
│
├── server/                   Python 3.11+ — runs on dev / cloud
│   ├── app_ws.py             FastAPI + WebSocket — main entry
│   ├── safety.py             pre-dispatch crisis gate (988)
│   ├── motion_trigger.py     regex short-circuit for body commands
│   ├── streaming.py          sentence chunker for streaming TTS
│   ├── elevenlabs_tts.py     Flash v2.5 streaming with voice profiles
│   ├── deepgram_asr.py       Nova-2 streaming STT
│   ├── elevenlabs_stt.py     Scribe v2 Realtime STT
│   ├── vad_silero.py         server-side authoritative voice gate
│   ├── memory.py             recaps, weekly themes, monthly personas
│   ├── session.py            SQLiteSession + camera consent
│   ├── logging_setup.py      structlog JSON
│   ├── metrics.py            Prometheus exporter
│   ├── dashboards/           Grafana JSON
│   ├── agents/               router, chat, chatbot, skills, therapist, ...
│   └── tools/                nao_actions, cs_navigator, emotion, skills
│
├── sim/                      Python 3.11+ — load test + benchmarks
├── docs/                     PRD, DECISIONS, INDEX, phase task maps
└── run.sh                    one-shot deploy + boot + log tail
```

---

## Documentation

| Tier | Doc | What it gives you |
|---|---|---|
| 1 | [`docs/INDEX`](docs/INDEX.md) | Doc graph + tag glossary |
| 1 | [`docs/DECISIONS`](docs/DECISIONS.md) | **How problems were navigated** — 12 key decisions with the why |
| 1 | [`docs/PRD_v2`](docs/PRD_v2.md) | 600‑line product/architecture spec, 9 phases |
| 2 | [`docs/spike_results`](docs/spike_results.md) | Phase 0.5 — FastAPI WS vs Realtime API benchmark |
| 3 | `docs/PHASE_*_TASK_MAP` | Per‑phase delivery plans with shared contracts |

Every doc carries an HTML‑comment frontmatter block with `tags:` and `related:` so they're greppable and graph‑traversable. Search example:

```bash
grep -lZ "tags:.*embodiment" docs/*.md      # every doc tagged with embodiment
grep -lZ "related:.*PHASE_4"  docs/*.md     # every doc that links to Phase 4
```

---

## Configuration

`.env` essentials (full list in `.env.example`, **never commit `.env`**):

```ini
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_GIRL=21m00Tcm4TlvDq8ikWAM
ELEVENLABS_VOICE_MAN=...
ELEVENLABS_VOICE_NEUTRAL=...
ELEVENLABS_VOICE_MY=...                    # cloned "my voice" (optional)
DEEPGRAM_API_KEY=...                        # optional, fastest STT
NAO_IP=172.20.95.127                        # robot LAN IP
NAO_PASSWORD=...                            # SSH for run.sh deploy (gitignored)
NAO_SHARED_SECRET=...                       # WS auth between robot and server
SERVER_IP=192.168.x.x                       # this machine's LAN IP
USE_WS=1                                    # FastAPI + WebSocket transport
```

Full env reference: see [`server/config.py`](server/config.py).

---

## Testing

```bash
pytest -q                       # 30 unit + integration tests
python -m sim.live_proof         # full pipeline soak test
python -m sim.stt_ab             # A/B Deepgram vs Whisper vs Scribe
python -m sim.chat_model_bench   # per-model latency
```

---

## NAO connection cheatsheet

```bash
ssh nao@$NAO_IP                                      # CS network
rsync -avz --delete nao/ nao@$NAO_IP:/home/nao/nao_assist/
qicli call ALBehaviorManager.getInstalledBehaviors   # 915 behaviors
qicli call ALBehaviorManager.runBehavior "follow-me"
qicli call ALBehaviorManager.stopAllBehaviors
```

VS Code Remote‑SSH:

```
Host nao
  HostName 172.20.95.127
  User nao
```

---

## Acknowledgments

- **Dr. Shuangbao "Paul" Wang** for advising the SAGE‑CBT research direction
- **Morgan State Department of Computer Science** for NAO + lab access
- **OpenAI**, **Anthropic Claude Code**, **ElevenLabs**, **Deepgram**, **Aldebaran/SoftBank Robotics**

## License

MIT — see [`LICENSE`](LICENSE).

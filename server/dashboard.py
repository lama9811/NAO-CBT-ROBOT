"""Live status dashboard: is NAO up, is the Pi up, what is being asked.

Served by the Pi's own server at ``/dashboard`` (the Pi sits behind Morgan's
NAT, so it is reachable only on campus Wi-Fi). If the page will not load at
all, the Pi itself is down.

Data sources, none of which touch the conversation code paths:

* Every structlog event passes through :func:`capture` (wired in
  ``logging_setup``). Turns, rejections, mute/camera changes and warnings
  are folded into small in-memory ring buffers.
* The robot sends a ``robot_status`` control frame every 30 s (battery,
  charging, Autonomous Life state, posture).
* A background task checks each outside service every few minutes with a
  free call (Claude, OpenAI, Deepgram, ElevenLabs, CS Navigator), so a dead
  API key shows up here instead of as a silent robot.

Access: no login. Anyone with the link can view it (the user's choice,
2026-09-30).

Privacy: NAO is a support robot. Turns answered by the support agents, and
anything crisis-related, are shown as "Support conversation" with no words.
Nothing is written to disk; a restart clears the history.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

_STATIC = Path(__file__).parent / "dashboard_static"

# Agents whose words are private. Anything crisis-related is too.
SUPPORT_AGENTS = {"therapist", "cbt_coach", "grounding_coach", "mi_coach"}

_TURN_LIMIT = 60
_PROBLEM_LIMIT = 30
_ROBOT_STALE_S = 90.0

# Events that are noise on a status page even at warning level.
_IGNORED_PROBLEMS = {"mute_listener_no_match"}

_lock = threading.Lock()


class _QuietPolling(logging.Filter):
    """Keep the page's 3-second polling out of the access log (~1200 lines/h)."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "/dashboard/api/state" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_QuietPolling())


def _now() -> float:
    return time.time()


class _State:
    def __init__(self) -> None:
        self.started_at = _now()
        self.version = _git_version()
        self.turns: collections.deque = collections.deque(maxlen=_TURN_LIMIT)
        self.problems: collections.deque = collections.deque(
            maxlen=_PROBLEM_LIMIT)
        self.pending_transcript: dict[str, str] = {}
        self.connections: dict[str, dict[str, Any]] = {}
        self.robot: dict[str, Any] = {}
        self.robot_ip: str | None = None
        self.robot_ssh_ok: bool | None = None
        self.robot_ssh_checked_at: float | None = None
        self.services: dict[str, dict[str, Any]] = {}
        self.muted = False
        self.day = time.strftime("%Y-%m-%d")
        self.today = collections.Counter()
        self.answer_ms: list[float] = []

    def roll_day(self) -> None:
        d = time.strftime("%Y-%m-%d")
        if d != self.day:
            self.day = d
            self.today = collections.Counter()
            self.answer_ms = []



def _git_version() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=3,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


STATE = _State()


# ─────────────────────────── event capture ────────────────────────────────
def _is_private(ev: dict[str, Any]) -> bool:
    agent = str(ev.get("active_agent") or "")
    outcome = str(ev.get("outcome") or "")
    name = str(ev.get("event") or "")
    # ``private`` / ``redacted`` come from server/privacy.py (log
    # redaction): a turn whose words were blanked stays wordless here too.
    if ev.get("private") or ev.get("redacted"):
        return True
    return (agent in SUPPORT_AGENTS or "crisis" in agent
            or "crisis" in outcome or "crisis" in name)


def _turn_from(ev: dict[str, Any], transcript: str) -> dict[str, Any]:
    phase = ev.get("phase_ms") or {}
    first_ms = phase.get("e2e_user_to_first_audio")
    answer_ms = phase.get("e2e_user_to_answer") or phase.get(
        "e2e_user_to_complete")
    private = _is_private(ev)
    outcome = str(ev.get("outcome") or "")
    reason = str(ev.get("reject_reason") or ev.get("reason") or "")
    item: dict[str, Any] = {
        "ts": _now(),
        "agent": str(ev.get("active_agent") or ""),
        "outcome": outcome,
        "reason": reason,
        "first_audio_ms": first_ms,
        "answer_ms": answer_ms,
        "cs_navigator_ms": phase.get("cs_navigator"),
        "private": private,
        "question": "",
        "reply": "",
    }
    if not private:
        if outcome != "rejected" or reason == "self_echo":
            item["question"] = (transcript or str(ev.get("transcript") or ""))[:240]
        item["reply"] = str(ev.get("reply_preview") or "")[:240]
    return item


def capture(_logger: Any, _name: str, ev: dict[str, Any]) -> dict[str, Any]:
    """structlog processor. Never raises and never alters the event."""
    try:
        _ingest_event(ev)
    except Exception:  # noqa: BLE001
        pass
    return ev


def _ingest_event(ev: dict[str, Any]) -> None:
    name = str(ev.get("event") or "")
    sid = str(ev.get("session_id") or "")
    level = str(ev.get("level") or "")
    with _lock:
        STATE.roll_day()
        if name == "stt_legacy":
            STATE.pending_transcript[sid] = str(ev.get("transcript") or "")
            return
        if name in ("turn_complete", "turn_rejected"):
            outcome = str(ev.get("outcome") or "")
            if outcome == "client_dropped":
                c = STATE.connections.get(sid)
                if c is not None:
                    c["open"] = False
                    c["closed_at"] = _now()
                return
            if outcome == "rejected" and str(ev.get("reject_reason")) in (
                    "wait_more_audio",):
                return
            tx = STATE.pending_transcript.pop(sid, "")
            turn = _turn_from(ev, tx)
            if name == "turn_rejected":
                turn["outcome"] = "rejected"
            STATE.turns.appendleft(turn)
            if turn["outcome"] == "ok":
                STATE.today["answered"] += 1
                if turn["agent"] == "cs_direct" or turn["agent"] == "chatbot":
                    STATE.today["cs"] += 1
                if turn["private"]:
                    STATE.today["support"] += 1
                ms = turn["first_audio_ms"]
                if isinstance(ms, (int, float)):
                    STATE.answer_ms.append(float(ms))
                    STATE.answer_ms = STATE.answer_ms[-200:]
            else:
                STATE.today["dropped"] += 1
            return
        if name == "motion_match":
            STATE.turns.appendleft({
                "ts": _now(), "agent": "action", "outcome": "ok",
                "reason": "", "first_audio_ms": None, "answer_ms": None,
                "cs_navigator_ms": None, "private": False,
                "question": str(ev.get("transcript") or "")[:240],
                "reply": str(ev.get("reply_preview") or ""),
            })
            STATE.today["answered"] += 1
            return
        if name == "mute_command":
            STATE.muted = bool(ev.get("now_muted"))
            return
        if name == "ws_connected":
            STATE.connections[sid] = {"open": True, "opened_at": _now()}
            return
        if level in ("warning", "error", "critical") and \
                name not in _IGNORED_PROBLEMS:
            detail = str(ev.get("error") or ev.get("reason") or "")[:200]
            STATE.problems.appendleft({
                "ts": _now(), "level": level, "event": name,
                "detail": detail,
            })


def install_log_hook() -> None:
    """Insert :func:`capture` into the *active* structlog chain.

    ``logging_setup.configure_logging`` is never called by the server, so
    the Pi runs structlog's default chain; the hook must be added to
    whatever chain is live, just before its renderer. Idempotent, and it
    leaves the log format exactly as it was.
    """
    try:
        import structlog
        procs = list(structlog.get_config().get("processors") or [])
        if any(getattr(p, "__name__", "") in ("capture", "_dashboard_capture")
               for p in procs):
            return
        insert_at = max(len(procs) - 1, 0)  # before the renderer
        procs.insert(insert_at, capture)
        structlog.configure(processors=procs)
    except Exception:  # noqa: BLE001 -- never break logging for a dashboard
        pass


# ───────────────────────── robot + connections ────────────────────────────
def robot_connected(ip: str | None) -> None:
    with _lock:
        if ip:
            STATE.robot_ip = ip


def robot_status(data: dict[str, Any]) -> None:
    with _lock:
        STATE.robot = dict(data or {})
        STATE.robot["received_at"] = _now()


def _probe_tcp(ip: str, port: int = 22, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


# ───────────────────────────── service checks ─────────────────────────────
async def _check(name: str, fn) -> None:
    t0 = time.perf_counter()
    try:
        ok, detail = await fn()
    except Exception as e:  # noqa: BLE001
        ok, detail = False, type(e).__name__
    ms = round((time.perf_counter() - t0) * 1000)
    with _lock:
        STATE.services[name] = {"ok": ok, "detail": detail, "ms": ms,
                                "checked_at": _now()}


def _key(name: str) -> str:
    return (os.environ.get(name) or "").strip()


async def _get(url: str, headers: dict[str, str]) -> httpx.Response:
    async with httpx.AsyncClient(timeout=10) as c:
        return await c.get(url, headers=headers)


async def _anthropic():
    k = _key("ANTHROPIC_API_KEY")
    if not k:
        return False, "No key set"
    r = await _get("https://api.anthropic.com/v1/models",
                   {"x-api-key": k, "anthropic-version": "2023-06-01"})
    return r.status_code == 200, _http_detail(r)


async def _openai():
    k = _key("OPENAI_API_KEY")
    if not k:
        return False, "No key set"
    r = await _get("https://api.openai.com/v1/models",
                   {"Authorization": f"Bearer {k}"})
    return r.status_code == 200, _http_detail(r)


async def _deepgram():
    k = _key("DEEPGRAM_API_KEY")
    if not k:
        return False, "No key set"
    r = await _get("https://api.deepgram.com/v1/projects",
                   {"Authorization": f"Token {k}"})
    return r.status_code == 200, _http_detail(r)


async def _elevenlabs():
    k = _key("ELEVENLABS_API_KEY")
    if not k:
        return False, "No key set"
    r = await _get("https://api.elevenlabs.io/v1/models", {"xi-api-key": k})
    if r.status_code == 200:
        return True, "OK"
    # The key is scoped for speech only; "missing_permissions" means the key
    # is valid but may not list models, which is fine for NAO.
    if "missing_permissions" in r.text:
        return True, "OK"
    return False, _http_detail(r)


async def _cs_navigator():
    from server import config
    base = (config.CS_NAVIGATOR_URL or "").rstrip("/")
    if not base:
        return False, "No link set"
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(base + "/")
    return r.status_code < 500, "Reachable" if r.status_code < 500 else \
        f"HTTP {r.status_code}"


def _http_detail(r: httpx.Response) -> str:
    if r.status_code == 200:
        return "OK"
    if r.status_code in (401, 403):
        return "Key rejected"
    if r.status_code == 429:
        return "Rate limited"
    return f"HTTP {r.status_code}"


SERVICE_CHECKS = {
    "claude": _anthropic,
    "openai": _openai,
    "deepgram": _deepgram,
    "elevenlabs": _elevenlabs,
    "cs_navigator": _cs_navigator,
}
# OpenAI is checked only if its backup voice is switched back on.
if os.environ.get("USE_OPENAI_TTS", "0") != "1":
    SERVICE_CHECKS.pop("openai", None)


async def run_checks_forever(interval_s: float = 300.0) -> None:
    while True:
        await asyncio.gather(*(_check(n, f) for n, f in SERVICE_CHECKS.items()))
        await asyncio.sleep(interval_s)


async def probe_robot_forever(interval_s: float = 20.0) -> None:
    while True:
        ip = STATE.robot_ip
        if ip:
            ok = await asyncio.to_thread(_probe_tcp, ip)
            with _lock:
                STATE.robot_ssh_ok = ok
                STATE.robot_ssh_checked_at = _now()
        await asyncio.sleep(interval_s)


# ─────────────────────────── remote (Vercel) push ─────────────────────────
def _remote_payload() -> dict[str, Any]:
    """What the Pi sends to the hosted dashboard.

    Support-agent and crisis turns never carry words (see _turn_from).
    Unless DASHBOARD_PUSH_CONVERSATION=1, every question and answer is
    stripped too, so only status, service health and counts leave campus.
    """
    snap = snapshot()
    # Off by default (2026-10-02): words leave campus only when someone
    # sets DASHBOARD_PUSH_CONVERSATION=1 on purpose.
    if os.environ.get("DASHBOARD_PUSH_CONVERSATION", "0") != "1":
        for t in snap["turns"]:
            t["question"] = ""
            t["reply"] = ""
        snap["conversation_hidden"] = True
    return snap


async def push_remote_forever(interval_s: float = 10.0) -> None:
    """POST the snapshot to the hosted dashboard, if one is configured.

    The Pi sits behind Morgan's NAT, so a cloud dashboard cannot reach it;
    the Pi reports out instead. Needs DASHBOARD_REMOTE_URL (the Vercel site)
    and DASHBOARD_INGEST_SECRET (shared with the site). Failures are logged
    at most once a minute and never affect the robot.
    """
    url = (os.environ.get("DASHBOARD_REMOTE_URL") or "").strip().rstrip("/")
    secret = (os.environ.get("DASHBOARD_INGEST_SECRET") or "").strip()
    if not url or not secret:
        return
    last_warn = 0.0
    log = logging.getLogger("sage.dashboard")
    async with httpx.AsyncClient(timeout=8) as client:
        while True:
            try:
                r = await client.post(
                    url + "/api/ingest", json=_remote_payload(),
                    headers={"Authorization": f"Bearer {secret}"})
                if r.status_code >= 300 and _now() - last_warn > 60:
                    last_warn = _now()
                    log.warning("dashboard push got HTTP %s", r.status_code)
            except Exception as e:  # noqa: BLE001
                if _now() - last_warn > 60:
                    last_warn = _now()
                    log.warning("dashboard push failed: %r", e)
            await asyncio.sleep(interval_s)


# ──────────────────────────────── snapshot ────────────────────────────────
def snapshot() -> dict[str, Any]:
    now = _now()
    with _lock:
        robot = dict(STATE.robot)
        age = now - robot["received_at"] if robot.get("received_at") else None
        open_conns = [c for c in STATE.connections.values() if c.get("open")]
        link_up = bool(open_conns)
        if age is not None and age <= _ROBOT_STALE_S and link_up:
            robot_state = "online"
        elif STATE.robot_ssh_ok:
            robot_state = "program_down"
        elif STATE.robot_ip is None and age is None:
            robot_state = "unknown"
        else:
            robot_state = "offline"
        ms = sorted(STATE.answer_ms)
        median = ms[len(ms) // 2] if ms else None
        return {
            "now": now,
            "server": {
                "online": True,
                "version": STATE.version,
                "started_at": STATE.started_at,
            },
            "robot": {
                "state": robot_state,
                "ip": STATE.robot_ip,
                "status_age_s": age,
                "battery": robot.get("battery"),
                "charging": robot.get("charging"),
                "life_state": robot.get("life_state"),
                "posture": robot.get("posture"),
                "awake": robot.get("awake"),
                "link_up": link_up,
                "muted": STATE.muted,
            },
            "services": dict(STATE.services),
            "turns": list(STATE.turns)[:40],
            "problems": list(STATE.problems)[:15],
            "today": {
                "answered": STATE.today["answered"],
                "cs": STATE.today["cs"],
                "support": STATE.today["support"],
                "dropped": STATE.today["dropped"],
                "median_first_audio_ms": median,
            },
        }


# ─────────────────────────────── routes ───────────────────────────────────
router = APIRouter()


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page() -> Response:
    return FileResponse(_STATIC / "index.html", media_type="text/html")


@router.get("/dashboard/nao.jpg")
async def dashboard_image() -> Response:
    return FileResponse(_STATIC / "nao.jpg", media_type="image/jpeg")


@router.get("/dashboard/api/state")
async def dashboard_state() -> Response:
    # Open to anyone with the link, by the user's choice (2026-09-30).
    # Support-agent and crisis turns never carry words regardless.
    return JSONResponse(snapshot(), headers={"Cache-Control": "no-store"})

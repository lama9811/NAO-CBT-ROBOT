#!/usr/bin/env python3
"""naostat -- one-screen diagnostics for the robot, the Pi and this laptop.

    ./scripts/naostat.py              probe everything once and print
    ./scripts/naostat.py --watch      refresh every 10s until Ctrl-C
    ./scripts/naostat.py --scan       force a /24 sweep even on a cache hit
                                      (a sweep already runs automatically
                                      whenever the known addresses all miss)
    ./scripts/naostat.py --no-color   plain text

Stdlib only, Python 3.9+, macOS / Linux / Windows. Every panel reports what it
actually measured -- a field it could not read prints "--", never a guess.

RUNNING IT ON A DIFFERENT MACHINE
  1. git clone https://github.com/lama9811/NAO-CBT-ROBOT.git
     cd NAO-CBT-ROBOT && ./scripts/naostat.py
     (no venv, no pip install; on Windows: python scripts\naostat.py)
  2. Install an SSH key that the robot and the Pi accept. THIS is the step
     people miss: every number below is read over ssh, so a machine without a
     key shows the entire fleet as offline. Either copy ~/.ssh/id_ed25519 and
     its .pub from a machine that already works (chmod 600 the private key),
     or enrol this one:
         ssh-keygen -t ed25519
         ssh-copy-id nao@<pi-ip>
         ssh-copy-id nao@<robot-ip>
     ssh-copy-id prompts for NAO_PASSWORD and must be typed in a real
     terminal -- it fails instantly from a script. The report says outright
     when a key was rejected, rather than calling the host offline.
  3. Be on the same LAN as the robot. Not a VPN, not a guest SSID.
  No .env is needed: it only ever supplied address hints, and discovery no
  longer depends on them.

Two house rules from CLAUDE.md are baked in:
  * never ping to test reachability -- this network drops ICMP
  * a host is only believed once it says what it is, because the DHCP lease
    moves and `ssh nao` has landed on the Pi before now

Discovery tries, in order: the last address that worked, mDNS, whatever the Pi
can see (live :5050 peer, then its ARP table), ~/.ssh/config, .env, and finally
a /24 sweep. All of those go stale at once often enough that no single one is
trusted -- that is what once reported the robot offline while it sat on .129
talking to the Pi.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENV = REPO / ".env"
SSH_CONFIG = Path.home() / ".ssh" / "config"
PI_REPO = "~/nao-sagecbt"

# Hosts that answered but rejected our key. Tracked so the report can say
# "this laptop has no key here" instead of "the robot is offline" -- on a new
# machine those look identical, and the wrong one sends you to the robot.
SSH_AUTH_FAILURES: set[str] = set()

SSH = [
    "ssh",
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=5",
    "-o", "LogLevel=ERROR",
]

# ---------------------------------------------------------------- formatting

class C:
    """ANSI colors, blanked out when the output is not a terminal."""
    on = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    reset = "\033[0m" if on else ""
    dim = "\033[2m" if on else ""
    bold = "\033[1m" if on else ""
    red = "\033[31m" if on else ""
    green = "\033[32m" if on else ""
    yellow = "\033[33m" if on else ""
    blue = "\033[34m" if on else ""
    cyan = "\033[36m" if on else ""
    grey = "\033[90m" if on else ""


def width() -> int:
    return min(shutil.get_terminal_size((88, 24)).columns, 100)


def plain(s: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", s)


def rule(title: str = "") -> str:
    w = width()
    if not title:
        return f"{C.grey}{'-' * w}{C.reset}"
    bar = "-" * max(0, w - len(title) - 3)
    return f"{C.grey}--{C.reset} {C.bold}{title}{C.reset} {C.grey}{bar}{C.reset}"


def row(label: str, value: str, tone: str = "") -> str:
    return f"  {C.grey}{label:<13}{C.reset}{tone}{value}{C.reset}"


def badge(text: str, tone: str) -> str:
    return f"{tone}{C.bold}[{text}]{C.reset}"


def ago(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    if d:
        return f"{d}d {h}h {m}m"
    if h:
        return f"{h}h {m}m"
    return f"{m}m"


def bar(pct: float, cells: int = 16, tone: str = "") -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round(pct / 100 * cells))
    return f"{tone}{'#' * filled}{C.grey}{'.' * (cells - filled)}{C.reset}"


# ---------------------------------------------------------------- primitives

def run(cmd: list[str], timeout: int = 12) -> tuple[int, str, str]:
    """Return (rc, stdout, stderr), kept apart.

    They must never be merged: the robot's sshd prints a multi-line warning
    banner on stderr, and folding that into stdout makes the banner the first
    line of every command's output -- which turned `hostname` into a warning
    string and let any host pass as the robot. stderr is still wanted, though,
    because "Permission denied (publickey)" is how a missing SSH key announces
    itself, and that must not be reported as a robot that is switched off.
    """
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout or "", p.stderr or ""
    except (subprocess.TimeoutExpired, OSError):
        return 124, "", ""


def ssh(host: str, script: str, timeout: int = 15, attempts: int = 1) -> tuple[int, str, str]:
    """Run a script over ssh, optionally retrying.

    The robot's WiFi link drops connects perfectly often -- measured at 3 of 5
    attempts timing out while the host was up and serving. A single failed
    connect therefore means nothing, so anything that concludes "not there"
    from a miss has to retry first.
    """
    rc, out, err = 124, "", ""
    for i in range(max(1, attempts)):
        rc, out, err = run(SSH + [host, script], timeout=timeout)
        if rc == 0 and out.strip():
            return rc, out, err
        low = err.lower()
        if "permission denied" in low or "host key verification failed" in low:
            SSH_AUTH_FAILURES.add(host.split("@")[-1])
            break          # a rejected key will be rejected again; do not retry
        if i + 1 < attempts:
            time.sleep(1.0)
    return rc, out, err


def port_open(ip: str, port: int = 22, timeout: float = 1.2) -> bool:
    """TCP probe. A refusal still means the host is up, but for :22 we want the
    service, so only a completed connection counts."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def mdns(name: str) -> str | None:
    try:
        return socket.gethostbyname(name)
    except OSError:
        return None


def env_value(key: str) -> str | None:
    if not ENV.exists():
        return None
    for line in ENV.read_text(errors="replace").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def ssh_config_host(alias: str) -> str | None:
    if not SSH_CONFIG.exists():
        return None
    current, text = None, SSH_CONFIG.read_text(errors="replace")
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("host "):
            current = s.split(None, 1)[1].strip()
        elif current == alias and s.lower().startswith("hostname"):
            return s.split(None, 1)[1].strip()
    return None


def local_subnet() -> str | None:
    """The /24 this machine is on, on any OS.

    A UDP socket "connected" to an off-machine address sends no packets -- it
    just makes the kernel pick the outbound interface, which is the one facing
    the robot. The previous version shelled out to `ipconfig getifaddr en0`,
    which exists only on macOS, so the sweep fallback silently did nothing on
    Linux and Windows -- exactly the machines most likely to need it.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return None
    return ip.rsplit(".", 1)[0] if IPV4.match(ip) else None


def sweep(prefix: str) -> list[str]:
    """TCP :22 across the /24, to narrow 254 hosts down to a handful worth
    an ssh. The last resort when every known address has gone stale."""
    hosts = [f"{prefix}.{i}" for i in range(1, 255)]
    found = []
    with futures.ThreadPoolExecutor(max_workers=64) as pool:
        for ip, ok in zip(hosts, pool.map(lambda h: port_open(h, 22, 0.6), hosts)):
            if ok:
                found.append(ip)
    return found


# ---------------------------------------------------------------- discovery

CACHE = Path.home() / ".cache" / "naostat" / "robot_ip"
IPV4 = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


def cache_read() -> str | None:
    try:
        ip = CACHE.read_text().strip()
        return ip if IPV4.match(ip) else None
    except OSError:
        return None


def cache_write(ip: str) -> None:
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(ip + "\n")
    except OSError:
        pass


def identify(ip: str) -> dict:
    """Ask a host what it is, rather than trusting the address it answered on.

    Returns {} when nothing usable came back. The robot runs NAOqiOS, so
    /etc/os-release is the reliable tell -- its hostname is plain "nao", which
    is too generic to bet on alone.
    """
    rc, out, _ = ssh(f"nao@{ip}", "hostname; cat /etc/os-release 2>/dev/null",
                     timeout=10, attempts=3)
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    if rc != 0 or not lines:
        return {}
    host = lines[0]
    blob = out.lower()
    return {"ip": ip, "host": host,
            "is_robot": "naoqi" in blob,
            "is_pi": host == "naoserver"}


def _first_match(candidates: list[tuple[str, str]], want: str) -> dict | None:
    """Identify every candidate in parallel and return the first that matches,
    in the order given -- so a cheap trusted source beats a sweep hit.

    Deliberately NOT gated behind a port probe. The robot's sshd is slow to
    accept: a 1.2s TCP probe reports :22 closed while `ssh` to the same address
    connects fine a second later, which made the robot vanish from this report
    at random. ssh carries its own ConnectTimeout, so let it be the judge.
    The sweep still pre-filters -- there, one probe each beats 254 ssh attempts.
    """
    if not candidates:
        return None
    ips = [ip for _, ip in candidates]
    with futures.ThreadPoolExecutor(max_workers=16) as pool:
        facts = dict(zip(ips, pool.map(identify, ips)))
    for src, ip in candidates:
        f = facts.get(ip) or {}
        if f.get(want):
            return {**f, "via": src}
    return None


def hints_from_pi(pi_ip: str | None) -> list[str]:
    """The Pi can see the robot even when this laptop cannot: as a live
    WebSocket peer on :5050, or simply as an ARP neighbour."""
    if not pi_ip:
        return []
    script = (
        "ss -tn 2>/dev/null | awk '/:5050/{print $5}' | cut -d: -f1; "
        "ip neigh 2>/dev/null | grep -vE 'FAILED|INCOMPLETE' | awk '{print $1}'"
    )
    _, out, _err = ssh(f"nao@{pi_ip}", script, timeout=12)
    seen, ips = set(), []
    for line in out.splitlines():
        ip = line.strip()
        if IPV4.match(ip) and ip != pi_ip and not ip.endswith(".255") and ip not in seen:
            seen.add(ip)
            ips.append(ip)
    return ips


def find_robot(scan: bool, pi_ip: str | None = None) -> dict:
    """The robot's DHCP lease moves constantly, so no single source is trusted.

    Sources are tried cheapest-first; the sweep is a real fallback rather than
    an opt-in, because the addresses in mDNS, ~/.ssh/config and .env are all
    routinely stale at the same time -- which is what made this report the
    robot offline while it was sitting on .129 talking to the Pi.
    """
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []

    def add(src: str, val: str | None) -> None:
        if val and IPV4.match(val) and val not in seen:
            seen.add(val)
            candidates.append((src, val))

    add("last known", cache_read())
    add("mdns nao.local", mdns("nao.local"))
    for ip in hints_from_pi(pi_ip):
        add("seen by the Pi", ip)
    add("ssh config", ssh_config_host("nao"))
    add(".env NAO_IP", env_value("NAO_IP"))

    hit = _first_match(candidates, "is_robot")
    if not hit or scan:
        prefix = local_subnet()
        if prefix:
            swept = [("subnet sweep", ip) for ip in sweep(prefix) if ip not in seen]
            hit = _first_match(swept, "is_robot")

    if hit:
        cache_write(hit["ip"])
        return {**hit, "tried": [f"{ip} ({src})" for src, ip in candidates]}
    return {"ip": None, "via": None, "host": None,
            "tried": [f"{ip} ({src})" for src, ip in candidates]}


def find_pi() -> dict:
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for src, val in (
        ("mdns naoserver.local", mdns("naoserver.local")),
        ("ssh config", ssh_config_host("naoserver")),
        (".env PI_IP", env_value("PI_IP")),
    ):
        if val and IPV4.match(val) and val not in seen:
            seen.add(val)
            candidates.append((src, val))
    return _first_match(candidates, "is_pi") or {"ip": None, "via": None}


# ---------------------------------------------------------------- collectors

ROBOT_PROBE = r"""
echo "uptime:$(cut -d' ' -f1 /proc/uptime 2>/dev/null)"
echo "main:$(pgrep -f 'python.*[m]ain\.py' | head -1)"
for c in "qicli call ALBattery.getBatteryCharge" \
         "qicli call ALMemory.getData Device/SubDeviceList/Battery/Charge/Sensor/Value"; do
  v=$($c 2>/dev/null | tr -d '\r' | tail -1)
  if [ -n "$v" ]; then echo "battery:$v"; break; fi
done
echo "charging:$(qicli call ALMemory.getData Device/SubDeviceList/Battery/Charge/Sensor/Status 2>/dev/null | tr -d '\r' | tail -1)"
echo "micraw:$(amixer -c 0 sget 'Numeric Left mics' 2>/dev/null | sed -n 's/.*Front Left: Capture \([0-9][0-9]*\).*/\1/p' | head -1)"
echo "server:$(ps ax 2>/dev/null | grep '[m]ain\.py' | grep -oE 'SERVER_IP=[0-9.]+' | head -1 | cut -d= -f2)"
echo "log:$(ls -t /home/nao/nao_assist/logs/*.jsonl 2>/dev/null | head -1)"
"""

PI_PROBE = r"""
echo "uptime:$(cut -d' ' -f1 /proc/uptime)"
echo "active:$(systemctl is-active nao-server 2>/dev/null)"
echo "started:$(systemctl show nao-server -p ExecMainStartTimestampMonotonic --value 2>/dev/null)"
echo "restarts:$(systemctl show nao-server -p NRestarts --value 2>/dev/null)"
echo "timer:$(systemctl is-active nao-autodeploy.timer 2>/dev/null)"
echo "temp:$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)"
echo "mem:$(free -m | awk '/^Mem:/{print $3"/"$2}')"
echo "disk:$(df -h / | awk 'NR==2{print $3"/"$2" "$5}')"
echo "load:$(cut -d' ' -f1 /proc/loadavg)"
echo "git:$(git -C REPO log -1 --format='%h %cd %s' --date=short 2>/dev/null | cut -c1-58)"
echo "ws:$(curl -s --max-time 4 http://localhost:5050/metrics 2>/dev/null | awk '/^nao_ws_connections_active/{print $2}')"
# The journal is hundreds of MB. Read it newest-first and stop at the first
# hit, or this single line takes longer than the whole probe's timeout.
echo "lastturn:$(journalctl -u nao-server -o short-iso --no-pager -r 2>/dev/null | grep -m1 turn_complete | cut -d' ' -f1)"
echo "today:$(journalctl -u nao-server -o short-iso --since today --no-pager 2>/dev/null | grep -c turn_complete)"
echo "todayok:$(journalctl -u nao-server -o short-iso --since today --no-pager 2>/dev/null | grep turn_complete | grep -c 'outcome=ok')"
""".replace("REPO", PI_REPO)


def parse_kv(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def collect_robot(scan: bool, pi_ip: str | None = None) -> dict:
    info = find_robot(scan, pi_ip)
    if not info["ip"]:
        return {"up": False, **info}
    rc, out, _ = ssh(f"nao@{info['ip']}", ROBOT_PROBE, timeout=20, attempts=3)
    raw = parse_kv(out)
    return {"up": bool(raw), "raw": raw, **info}


def collect_pi(info: dict | None = None) -> dict:
    info = info if info is not None else find_pi()
    if not info.get("ip"):
        return {"up": False, **info}
    rc, out, _ = ssh(f"nao@{info['ip']}", PI_PROBE, timeout=25)
    raw = parse_kv(out)
    return {"up": bool(raw.get("uptime")), "raw": raw, **info}


def collect_local() -> dict:
    d = {}
    rc, out, _ = run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"], timeout=6)
    d["branch"] = out.strip() if rc == 0 else None
    rc, out, _ = run(["git", "-C", str(REPO), "log", "-1", "--format=%h %cd %s", "--date=short"], timeout=6)
    d["head"] = out.strip()[:58] if rc == 0 else None
    rc, out, _ = run(["git", "-C", str(REPO), "status", "--porcelain"], timeout=8)
    d["dirty"] = len([l for l in out.splitlines() if l.strip()]) if rc == 0 else None

    pid_file = REPO / "logs" / "server.pid"
    d["server"] = False
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            d["server"] = True
        except (ValueError, OSError):
            d["server"] = False

    venv = REPO / ".venv" / "bin" / "python"
    d["venv"] = None
    if venv.exists():
        rc, out, _ = run([str(venv), "-V"], timeout=8)
        d["venv"] = out.strip() if rc == 0 else None
    return d


# ---------------------------------------------------------------- rendering

def render_robot(r: dict) -> list[str]:
    lines = []
    if not r["up"]:
        lines.append(rule(f"NAO robot  {badge('OFFLINE', C.red)}"))
        lines.append(row("address", "not found on any known address", C.grey))
        for t in r.get("tried", [])[:4]:
            lines.append(row("", f"tried {t}", C.grey))
        if not r.get("tried"):
            lines.append(row("", "no candidate address in mDNS, ssh config or .env", C.grey))
        lines.append(row("", "press the chest button -- NAO speaks its own IP", C.grey))
        return lines

    raw = r.get("raw", {})
    lines.append(rule(f"NAO robot  {badge('ONLINE', C.green)}"))
    lines.append(row("address", f"{r['ip']}  {C.grey}via {r['via']} -- hostname {r['host']}"))

    batt = raw.get("battery", "")
    try:
        pct = float(batt)
        if pct <= 1.0:            # the ALMemory key reports 0..1
            pct *= 100
        tone = C.green if pct >= 50 else C.yellow if pct >= 25 else C.red
        charging = raw.get("charging", "")
        tag = " charging" if charging and charging not in ("0", "") else ""
        lines.append(row("battery", f"{bar(pct, 16, tone)} {tone}{pct:5.1f}%{C.reset}{C.grey}{tag}"))
    except ValueError:
        lines.append(row("battery", "-- (ALBattery did not answer)", C.grey))

    try:
        up = float(raw.get("uptime", ""))
        lines.append(row("powered on", f"{ago(up)} ago"))
    except ValueError:
        lines.append(row("powered on", "--", C.grey))

    main = raw.get("main", "")
    if main:
        lines.append(row("main.py", f"running  {C.grey}pid {main}", C.green))
    else:
        lines.append(row("main.py", "not running -- robot will not answer", C.red))

    gain = raw.get("micraw", "")
    if gain.isdigit():
        g = int(gain)
        tone = C.green if g >= 60 else C.red
        note = "" if g >= 60 else "  <- too low, NAO will hear nothing"
        lines.append(row("mic gain", f"{g}/88{C.reset}{tone}{note}", tone))
    else:
        lines.append(row("mic gain", "--", C.grey))

    if raw.get("server"):
        lines.append(row("points at", raw["server"]))
    return lines


def render_pi(p: dict) -> list[str]:
    lines = []
    if not p["up"]:
        lines.append(rule(f"naoserver (Pi)  {badge('UNREACHABLE', C.red)}"))
        lines.append(row("address", "naoserver.local did not resolve to a live host", C.grey))
        return lines

    raw = p.get("raw", {})
    active = raw.get("active", "")
    tone = C.green if active == "active" else C.red
    lines.append(rule(f"naoserver (Pi)  {badge('UP', C.green)}"))
    lines.append(row("address", f"{p['ip']}  {C.grey}via {p['via']}"))
    lines.append(row("nao-server", f"{active or '--'}{C.reset}{C.grey}"
                                   f"  restarts {raw.get('restarts', '--')}"
                                   f"  autodeploy {raw.get('timer', '--')}", tone))

    try:
        lines.append(row("uptime", ago(float(raw.get("uptime", "")))))
    except ValueError:
        lines.append(row("uptime", "--", C.grey))

    temp = raw.get("temp", "")
    bits = []
    if temp.isdigit():
        c = int(temp) / 1000
        bits.append(f"{C.green if c < 70 else C.yellow}{c:.1f}C{C.reset}")
    if raw.get("load"):
        bits.append(f"{C.grey}load {raw['load']}{C.reset}")
    if raw.get("mem"):
        bits.append(f"{C.grey}mem {raw['mem']}MB{C.reset}")
    if raw.get("disk"):
        bits.append(f"{C.grey}disk {raw['disk']}{C.reset}")
    if bits:
        lines.append(row("health", "  ".join(bits)))

    ws = raw.get("ws", "")
    try:
        n = int(float(ws))
        lines.append(row("live sockets", f"{n} open{'' if n else '  (nothing connected)'}",
                         C.green if n else C.grey))
    except ValueError:
        pass

    today, ok = raw.get("today", "0"), raw.get("todayok", "0")
    if today.isdigit() and int(today):
        lines.append(row("turns today", f"{today}  {C.grey}{ok} answered"))
    else:
        lines.append(row("turns today", "none", C.grey))

    if raw.get("lastturn"):
        lines.append(row("last turn", raw["lastturn"]))
    if raw.get("git"):
        lines.append(row("deployed", raw["git"], C.grey))
    return lines


def render_local(l: dict, pi: dict) -> list[str]:
    lines = [rule("this laptop")]
    srv = l.get("server")
    lines.append(row("dev server", "running on this Mac" if srv else "not running",
                     C.green if srv else C.grey))
    if l.get("branch"):
        dirty = l.get("dirty") or 0
        tail = f"  {C.yellow}{dirty} uncommitted{C.reset}" if dirty else ""
        lines.append(row("branch", f"{l['branch']}{C.reset}{tail}"))
    if l.get("head"):
        lines.append(row("head", l["head"], C.grey))
    if l.get("venv"):
        lines.append(row("venv", l["venv"], C.grey))

    # Deployed-vs-local drift is the failure that looks like a broken robot.
    pi_git = (pi.get("raw") or {}).get("git", "")
    if pi_git and l.get("head"):
        if pi_git.split()[0] != l["head"].split()[0]:
            lines.append(row("drift", f"Pi serves {pi_git.split()[0]}, local is "
                                      f"{l['head'].split()[0]}", C.yellow))
    return lines


def render_auth_warning() -> list[str]:
    """A host that rejected our key is not a host that is switched off.

    Without this, a laptop with no key installed shows the whole fleet as
    offline -- which reads as "the robot is broken" and sends you to go and
    poke the robot, when the fix is three commands on this machine.
    """
    if not SSH_AUTH_FAILURES:
        return []
    hosts = ", ".join(sorted(SSH_AUTH_FAILURES))
    return [
        f"  {C.yellow}{C.bold}! SSH key not accepted by: {hosts}{C.reset}",
        f"  {C.grey}  Those hosts are UP -- this machine just cannot log in, so"
        f" every field below reads empty.{C.reset}",
        f"  {C.grey}  Fix on THIS machine:  ssh-keygen -t ed25519   then"
        f"  ssh-copy-id nao@<host>{C.reset}",
        f"  {C.grey}  ssh-copy-id asks for NAO_PASSWORD and must be run in a"
        f" real terminal, not a script.{C.reset}",
        "",
    ]


def render(robot: dict, pi: dict, local: dict) -> str:
    out = []
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    out.append(f"{C.bold}{C.cyan}NAO fleet{C.reset}  {C.grey}{stamp}{C.reset}")
    out.append("")
    out += render_auth_warning()
    out += render_robot(robot)
    out.append("")
    out += render_pi(pi)
    out.append("")
    out += render_local(local, pi)
    return "\n".join(out)


# ---------------------------------------------------------------- entrypoint

def probe(scan: bool) -> str:
    pi_info = find_pi()
    with futures.ThreadPoolExecutor(max_workers=3) as pool:
        f_robot = pool.submit(collect_robot, scan, pi_info.get("ip"))
        f_pi = pool.submit(collect_pi, pi_info)
        f_local = pool.submit(collect_local)
        return render(f_robot.result(), f_pi.result(), f_local.result())


def main() -> int:
    ap = argparse.ArgumentParser(description="One-screen NAO fleet diagnostics.")
    ap.add_argument("--watch", nargs="?", const=10, type=int, metavar="SECS",
                    help="refresh every SECS seconds (default 10) until Ctrl-C")
    ap.add_argument("--scan", action="store_true",
                    help="sweep the local /24 for the robot when known addresses miss")
    ap.add_argument("--no-color", action="store_true", help="plain text output")
    args = ap.parse_args()

    if args.no_color:
        C.on = False
        for k in ("reset", "dim", "bold", "red", "green", "yellow", "blue", "cyan", "grey"):
            setattr(C, k, "")

    if not args.watch:
        print(probe(args.scan))
        return 0

    try:
        while True:
            text = probe(args.scan)
            print("\033[H\033[J" if C.on else "\n" + "=" * width())
            print(text)
            print(f"\n{C.grey}refreshing every {args.watch}s -- Ctrl-C to stop{C.reset}")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())

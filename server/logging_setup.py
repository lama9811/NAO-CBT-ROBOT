"""Structured logging setup for Phase 1 of NAO Morgan Assist v2.

This module wires up `structlog` so every per-turn event in the FastAPI
WebSocket transport emits a JSON log line with consistent shape. It is
imported once at app startup (`configure_logging()`), then any caller
can `from server.logging_setup import logger` and bind contextual fields
(user, session_id, turn_idx) for free.

Configuration (env vars):
    LOG_FORMAT  "json" (default) or "console" — JSON for prod / aggregation,
                console (pretty, colorized) for local dev.
    LOG_LEVEL   "INFO" by default. Standard logging levels.

Auto-injected fields on every event:
    ts          UTC ISO 8601 with millisecond precision (e.g. 2026-05-06T20:00:00.123Z)
    level       lower-case log level
    event       the event name passed by the caller — see allowed set below

Documented event names (callers MUST use one of these — not enforced in
code, but anything else is a bug):
    turn_complete    a /turn or /ws turn finished cleanly
    turn_error       the turn raised; payload includes exception details
    wake_event       face / proximity / keyword wake fired
    crisis_block     safety.crisis_check rejected before agent ran
    motion_match     motion_trigger.detect short-circuited the LLM
    session_open     a new WS session was accepted
    session_close    the WS session ended
    stt_complete     STT returned a transcript

Public API:
    logger                              — module-level structlog BoundLogger;
                                          can be re-bound via `.bind(...)`.
    configure_logging() -> None         — idempotent setup; safe to call many
                                          times. Read once on first call;
                                          subsequent calls are no-ops.
    per_turn_logger(user, session_id, turn_idx) -> structlog.BoundLogger
                                        — convenience: returns a logger
                                          pre-bound with the three fields
                                          every per-turn event needs.

Phase 1 contract: see docs/PHASE_1_TASK_MAP.md, "Logging shape" section,
for the full per-turn JSON schema this module produces.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import structlog


# Re-exported allowed event names so callers can sanity-check via membership
# (the README in the docstring is the source of truth; this is a convenience).
ALLOWED_EVENTS: frozenset[str] = frozenset({
    "turn_complete",
    "turn_error",
    "wake_event",
    "crisis_block",
    "motion_match",
    "session_open",
    "session_close",
    "stt_complete",
})


_CONFIGURED = False


def _utc_iso_ms(_logger, _name, event_dict):
    """Custom timestamper — UTC ISO 8601 with milliseconds and trailing Z.

    structlog's bundled `TimeStamper(fmt="iso", utc=True)` emits microsecond
    precision and a `+00:00` suffix; we want millisecond + Z to match the
    Phase 1 task-map spec exactly.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    # millisecond precision, Z suffix
    event_dict["ts"] = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
    return event_dict


def _dashboard_capture(logger, name, event_dict):
    """Feed the live dashboard (server/dashboard.py). Never raises."""
    try:
        from server.dashboard import capture
        return capture(logger, name, event_dict)
    except Exception:  # noqa: BLE001
        return event_dict


def configure_logging() -> None:
    """Configure structlog + stdlib logging from env. Idempotent.

    Reads LOG_FORMAT and LOG_LEVEL once; subsequent calls return early.
    Tests that need to reconfigure can flip the module-level `_CONFIGURED`
    sentinel directly.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_format = os.environ.get("LOG_FORMAT", "json").strip().lower()
    log_level_name = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    # Stdlib logging is what structlog's wrap_for_formatter eventually writes
    # through. We point it at stderr with a minimal formatter — structlog has
    # already shaped the message by then.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    # Replace handlers to keep idempotency: re-running configure_logging in
    # tests doesn't pile up duplicate handlers.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(log_level)

    # The processor chain. Order matters:
    #   1. add_log_level         — populates "level"
    #   2. _utc_iso_ms           — populates "ts" (our spec format)
    #   3. format_exc_info       — turns exc_info=True into a string field
    #   4. final renderer        — JSON or console based on env
    if log_format == "console":
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    else:
        # JSON is the default — production, aggregators, Phase 1 spec.
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            _utc_iso_ms,
            structlog.processors.format_exc_info,
            _dashboard_capture,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


# Module-level lazy logger. Most callers will do:
#     from server.logging_setup import logger
#     logger.info("session_open", user=user, session_id=sid)
# It can be re-bound:
#     log = logger.bind(user="aayush", session_id=sid, turn_idx=7)
#     log.info("turn_complete", phase_ms={...})
logger: structlog.stdlib.BoundLogger = structlog.get_logger()


def per_turn_logger(user: str, session_id: str, turn_idx: int) -> structlog.stdlib.BoundLogger:
    """Return a logger pre-bound with the three fields every per-turn event
    needs. Callers should still pass `event=` as the first argument.

    Example:
        log = per_turn_logger(user, sid, idx)
        log.info("turn_complete", phase_ms={...}, transcript=t, outcome="ok")
    """
    return logger.bind(user=user, session_id=session_id, turn_idx=turn_idx)


__all__ = [
    "ALLOWED_EVENTS",
    "configure_logging",
    "logger",
    "per_turn_logger",
]

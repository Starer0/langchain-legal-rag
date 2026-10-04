"""Request correlation and bounded JSON Lines logs, including streamed responses."""

import asyncio
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4

from starlette.responses import JSONResponse

_logger_lock = Lock()
_loggers = {}


def create_request_logger(path, *, max_bytes=5 * 1024 * 1024, backup_count=5):
    """Create an independent console/file logger; caller owns its handlers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.Logger("legal_rag.requests", level=logging.INFO)
    formatter = logging.Formatter("%(message)s")
    file_handler = RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8",
    )
    console_handler = logging.StreamHandler(sys.stderr)
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def default_request_logger():
    path = Path(os.getenv("WEB_REQUEST_LOG_PATH", "logs/requests.jsonl")).resolve()
    with _logger_lock:
        if path not in _loggers:
            configured = create_request_logger(path)
            logger = logging.getLogger("legal_rag.requests")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            # A single default destination per process avoids duplicate writes.
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            for handler in configured.handlers:
                logger.addHandler(handler)
            _loggers.clear()
            _loggers[path] = logger
        return _loggers[path]


def write_event(logger, event, **fields):
    logger.info(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event, **fields,
    }, ensure_ascii=False))


class RequestLoggingMiddleware:
    """Observe ASGI body completion, not just creation of a StreamingResponse."""

    def __init__(self, app, logger=None):
        self.app = app
        self.logger = logger if logger is not None else default_request_logger()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        state = scope.setdefault("state", {})
        state['request_logger'] = self.logger
        state["request_id"] = uuid4().hex
        fields = {"request_id": state["request_id"], "method": scope["method"], "path": scope["path"]}
        started = perf_counter()
        status = None
        body_complete = False
        disconnected = False
        write_event(self.logger, "request_started", **fields)

        async def observed_receive():
            nonlocal disconnected
            message = await receive()
            if message["type"] == "http.disconnect" and not body_complete:
                disconnected = True
            return message

        async def observed_send(message):
            nonlocal status, body_complete, disconnected
            if message["type"] == "http.response.start":
                status = message["status"]
                message = {**message, "headers": [
                    (key, value) for key, value in message.get("headers", [])
                    if key.lower() != b"x-request-id"
                ] + [(b"x-request-id", state["request_id"].encode("ascii"))]}
            try:
                await send(message)
            except OSError:
                disconnected = True
                raise
            if message["type"] == "http.response.pathsend" or (
                message["type"] == "http.response.body" and not message.get("more_body", False)
            ):
                body_complete = True

        try:
            await self.app(scope, observed_receive, observed_send)
        except asyncio.CancelledError:
            disconnected = True
            raise
        except Exception:
            if not disconnected:
                state["request_outcome"] = "failed"
                state["request_error"] = traceback.format_exc()
                if status is None:
                    response = JSONResponse({
                        "detail": "暂时无法完成请求，请稍后重试。", "request_id": state["request_id"],
                    }, status_code=500)
                    await response(scope, observed_receive, observed_send)
                else:
                    raise
        finally:
            if disconnected or (not body_complete and not state.get("request_error")):
                outcome = "disconnected"
            elif state.get("request_outcome"):
                outcome = state["request_outcome"]
            elif status is not None and status >= 400:
                outcome = "failed" if status >= 500 else "rejected"
            else:
                outcome = "completed"
            result = {**fields, "status_code": status, "outcome": outcome,
                      "duration_ms": round((perf_counter() - started) * 1000, 3)}
            if state.get("request_error"):
                result["error"] = state["request_error"]
            write_event(self.logger, "request_finished", **result)

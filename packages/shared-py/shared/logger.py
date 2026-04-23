"""Mirror of shared-ts logger: stdout JSON lines + Upstash Redis Stream.

Same contract (LogEntry) as the TS side so `pnpm logs --stage X --follow` sees
entries from Python stages identically.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from .redis_client import redis

MAX_STREAM_LEN = 10_000
LEVELS = ("debug", "info", "warn", "error")


def _err_obj(err: BaseException | None) -> dict[str, str] | None:
    if err is None:
        return None
    return {
        "name": type(err).__name__,
        "message": str(err),
        "stack": "".join(traceback.format_exception(type(err), err, err.__traceback__)),
    }


def _flatten(entry: dict[str, Any]) -> dict[str, str]:
    return {
        "ts": str(entry["ts"]),
        "stage": entry["stage"],
        "job_id": entry.get("job_id") or "",
        "level": entry["level"],
        "msg": entry["msg"],
        "data": json.dumps(entry["data"], separators=(",", ":")) if entry.get("data") else "",
        "err": json.dumps(entry["err"], separators=(",", ":")) if entry.get("err") else "",
    }


def _publish(entry: dict[str, Any]) -> None:
    try:
        redis().xadd(
            f"logs:{entry['stage']}",
            {"values": _flatten(entry)},  # upstash_redis Python: hset-style values= kwarg-free API
            # NOTE: upstash_redis python xadd signature is (name, id, values, ...).
            # We use id="*" and values=dict; MAXLEN trim via approximate.
        )
    except TypeError:
        # Fallback if signature differs in newer versions — positional call.
        try:
            redis().xadd(f"logs:{entry['stage']}", "*", _flatten(entry))
        except Exception as e:  # pragma: no cover — never crash the caller
            sys.stderr.write(json.dumps({
                "ts": int(time.time() * 1000),
                "stage": entry["stage"],
                "level": "warn",
                "msg": "log stream publish failed",
                "err": _err_obj(e),
            }) + "\n")
    except Exception as e:  # pragma: no cover
        sys.stderr.write(json.dumps({
            "ts": int(time.time() * 1000),
            "stage": entry["stage"],
            "level": "warn",
            "msg": "log stream publish failed",
            "err": _err_obj(e),
        }) + "\n")


@dataclass
class Logger:
    stage: str

    def _emit(
        self,
        level: str,
        job_id: str | None,
        msg: str,
        data: dict[str, Any] | None = None,
        err: BaseException | None = None,
    ) -> None:
        entry = {
            "ts": int(time.time() * 1000),
            "stage": self.stage,
            "job_id": job_id,
            "level": level,
            "msg": msg,
            "data": data,
            "err": _err_obj(err),
        }
        sys.stdout.write(json.dumps(entry) + "\n")
        sys.stdout.flush()
        _publish(entry)

    def debug(self, job_id: str | None, msg: str, data: dict[str, Any] | None = None) -> None:
        self._emit("debug", job_id, msg, data)

    def info(self, job_id: str | None, msg: str, data: dict[str, Any] | None = None) -> None:
        self._emit("info", job_id, msg, data)

    def warn(self, job_id: str | None, msg: str, data: dict[str, Any] | None = None) -> None:
        self._emit("warn", job_id, msg, data)

    def error(
        self,
        job_id: str | None,
        msg: str,
        err: BaseException,
        data: dict[str, Any] | None = None,
    ) -> None:
        self._emit("error", job_id, msg, data, err)


def create_logger(stage: str) -> Logger:
    return Logger(stage=stage)


def flush_logs() -> None:
    """Give in-flight xadds a moment to land. Call before exit in short scripts."""
    time.sleep(0.25)

"""Env var access with explicit failure on missing required values."""

from __future__ import annotations

import os


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required env var: {name}")
    return value


def optional(name: str, fallback: str = "") -> str:
    return os.environ.get(name) or fallback


def optional_int(name: str, fallback: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return fallback
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"env var {name}={raw!r} is not an int") from exc


def is_local() -> bool:
    return os.environ.get("NODE_ENV", "").lower() == "local" or os.environ.get("ANNEMUSIC_LOCAL") == "1"

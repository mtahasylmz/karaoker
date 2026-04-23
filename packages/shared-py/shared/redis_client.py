"""Lazy singleton Upstash Redis client."""

from __future__ import annotations

from upstash_redis import Redis

from .env import required

_client: Redis | None = None


def redis() -> Redis:
    global _client
    if _client is None:
        _client = Redis(
            url=required("UPSTASH_REDIS_REST_URL"),
            token=required("UPSTASH_REDIS_REST_TOKEN"),
        )
    return _client

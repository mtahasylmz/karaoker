"""Upstash Redis access layer for the worker service."""

import os
import time

from upstash_redis import Redis

_redis: Redis | None = None


def r() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis(
            url=os.environ["UPSTASH_REDIS_REST_URL"],
            token=os.environ["UPSTASH_REDIS_REST_TOKEN"],
        )
    return _redis


def _now() -> str:
    return str(int(time.time()))


def update_job_status(job_id: str, status: str, **extra: str) -> None:
    values = {"status": status, "updated_at": _now(), **extra}
    r().hset(f"job:{job_id}", values=values)


def set_job_failed(job_id: str, error: str) -> None:
    update_job_status(job_id, "failed", error=error[:500])


def set_video_status(youtube_id: str, status: str, **extra: str) -> None:
    values = {"status": status, "updated_at": _now(), **extra}
    r().hset(f"video:{youtube_id}", values=values)


def set_video_done(youtube_id: str, video_url: str) -> None:
    set_video_status(youtube_id, "done", video_url=video_url)


def set_video_failed(youtube_id: str, error: str) -> None:
    set_video_status(youtube_id, "failed", error=error[:500])


def get_video(youtube_id: str) -> dict | None:
    data = r().hgetall(f"video:{youtube_id}")
    return data or None

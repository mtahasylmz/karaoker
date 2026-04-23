"""Upstash Redis access layer for the API service."""

import os
import re
import time

from upstash_redis import Redis

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{1,23}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_JOBS_PER_USER = 50  # LTRIM keeps the list bounded

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


def valid_username(username: str) -> bool:
    return bool(_USERNAME_RE.match(username or ""))


def valid_sha256(s: str) -> bool:
    return bool(_SHA256_RE.match(s or ""))


def reserve_username(username: str) -> bool:
    """Atomic SETNX on user:{username}. True if the username was free."""
    return bool(r().set(f"user:{username}", "1", nx=True))


def user_exists(username: str) -> bool:
    return r().exists(f"user:{username}") == 1


# ---------- uploads ----------

def get_upload(sha256: str) -> dict | None:
    data = r().hgetall(f"upload:{sha256}")
    return data or None


def record_upload(sha256: str, size: int, content_type: str, object_path: str) -> None:
    r().hset(
        f"upload:{sha256}",
        values={
            "sha256": sha256,
            "size": str(size),
            "content_type": content_type,
            "object_path": object_path,
            "created_at": _now(),
        },
    )


# ---------- video dedup cache ----------

def get_video(sha256: str) -> dict | None:
    data = r().hgetall(f"video:{sha256}")
    return data or None


def claim_video(sha256: str, job_id: str) -> bool:
    """Become the first writer for this content hash.

    HSETNX on the status field — only one caller wins. Winner fills in the rest
    with a follow-up HSET. Good enough for a family-scale app.
    """
    key = f"video:{sha256}"
    won = bool(r().hsetnx(key, "status", "queued"))
    if won:
        r().hset(key, values={"job_id": job_id, "created_at": _now()})
    return won


# ---------- jobs ----------

def create_job(job_id: str, sha256: str, object_path: str, username: str) -> None:
    r().hset(
        f"job:{job_id}",
        values={
            "job_id": job_id,
            "sha256": sha256,
            "object_path": object_path,
            "username": username,
            "status": "queued",
            "created_at": _now(),
            "updated_at": _now(),
        },
    )


def get_job(job_id: str) -> dict | None:
    data = r().hgetall(f"job:{job_id}")
    return data or None


def append_user_job(username: str, job_id: str) -> None:
    key = f"user:{username}:jobs"
    r().lpush(key, job_id)
    r().ltrim(key, 0, _MAX_JOBS_PER_USER - 1)


def list_user_job_ids(username: str, limit: int = 20) -> list[str]:
    return r().lrange(f"user:{username}:jobs", 0, limit - 1) or []

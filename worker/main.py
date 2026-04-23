"""Worker HTTP entrypoint. One endpoint: POST /process (invoked by Cloud Tasks)."""

import logging
import os

from fastapi import FastAPI, Header, HTTPException, Request

import pipeline
import redis_client

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("worker")

app = FastAPI(title="annemusic worker")


@app.get("/ping")
def ping():
    return {"ok": True}


@app.post("/process")
async def process(request: Request, authorization: str | None = Header(default=None)):
    if os.environ.get("SKIP_AUTH") != "1":
        _verify_oidc(authorization)

    body = await request.json()
    job_id = body.get("job_id")
    sha256 = body.get("sha256")
    object_path = body.get("object_path")
    if not (job_id and sha256 and object_path):
        raise HTTPException(400, "missing job_id/sha256/object_path")

    try:
        pipeline.run(job_id, sha256, object_path)
    except Exception as e:
        log.exception("pipeline failed for job %s", job_id)
        msg = f"{type(e).__name__}: {e}"
        redis_client.set_job_failed(job_id, msg)
        redis_client.set_video_failed(sha256, msg)
        # Return 500 so Cloud Tasks retries (up to queue max-attempts).
        raise HTTPException(500, msg)

    return {"ok": True, "job_id": job_id}


def _verify_oidc(authorization: str | None) -> None:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")
    token = authorization.split(" ", 1)[1]
    audience = os.environ.get("OIDC_AUDIENCE")
    if not audience:
        raise HTTPException(500, "OIDC_AUDIENCE not configured")
    try:
        from google.auth.transport import requests as g_requests
        from google.oauth2 import id_token
        id_token.verify_oauth2_token(token, g_requests.Request(), audience=audience)
    except Exception as e:
        raise HTTPException(401, f"invalid oidc token: {e}")

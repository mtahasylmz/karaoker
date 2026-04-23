import logging
import os
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import gcs_signed
import redis_client
import tasks

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

app = FastAPI(title="annemusic API")

_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- constants ----------

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
EXT_BY_CONTENT_TYPE = {
    "video/mp4": "mp4",
    "video/quicktime": "mov",
    "video/webm": "webm",
    "video/x-matroska": "mkv",
}


# ---------- schemas ----------

class RegisterRequest(BaseModel):
    username: str = Field(min_length=2, max_length=24)


class UploadRequest(BaseModel):
    username: str
    sha256: str
    size: int = Field(ge=1, le=MAX_UPLOAD_BYTES)
    content_type: str


class JobRequest(BaseModel):
    username: str
    sha256: str


# ---------- exception handler ----------

@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    log.exception("unhandled exception on %s %s", request.method, request.url.path)
    # Preserve CORS so browsers surface the real error instead of masking as CORS.
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
        headers={"Access-Control-Allow-Origin": "*"},
    )


# ---------- endpoints ----------

@app.get("/ping")
def ping():
    return {"ok": True}


@app.post("/users", status_code=201)
def register_user(req: RegisterRequest):
    if not redis_client.valid_username(req.username):
        raise HTTPException(400, "username must be 2-24 chars, alnum/_/./- only")
    if not redis_client.reserve_username(req.username):
        raise HTTPException(409, "username taken")
    return {"username": req.username}


@app.get("/users/{username}")
def get_user(username: str):
    if not redis_client.user_exists(username):
        raise HTTPException(404, "unknown username")
    return {"username": username}


@app.post("/uploads")
def request_upload(req: UploadRequest):
    if not redis_client.user_exists(req.username):
        raise HTTPException(404, "unknown username")
    if not redis_client.valid_sha256(req.sha256):
        raise HTTPException(400, "sha256 must be 64 hex chars")
    ext = EXT_BY_CONTENT_TYPE.get(req.content_type)
    if not ext:
        raise HTTPException(400, f"unsupported content_type; accept one of: {list(EXT_BY_CONTENT_TYPE)}")

    # 1. Fully processed already → skip upload + job entirely.
    video = redis_client.get_video(req.sha256)
    if video and video.get("status") == "done" and video.get("video_url"):
        return {
            "cached": True,
            "status": "done",
            "sha256": req.sha256,
            "video_url": video["video_url"],
            "need_upload": False,
        }

    # 2. Upload already completed (Redis record AND object present in GCS) →
    # skip PUT, client goes straight to /jobs. Stale Redis records (PUT that
    # never succeeded) fall through to step 3 so we mint a fresh URL.
    existing = redis_client.get_upload(req.sha256)
    if existing and existing.get("object_path") and gcs_signed.object_exists(existing["object_path"]):
        return {
            "cached": False,
            "status": "uploaded",
            "sha256": req.sha256,
            "object_path": existing["object_path"],
            "need_upload": False,
        }

    # 3. Fresh (or stale-reset) upload — mint signed PUT URL.
    object_path = f"uploads/{req.sha256}.{ext}"
    try:
        put_url = gcs_signed.signed_put_url(object_path, req.content_type)
    except Exception as e:
        log.exception("signed URL generation failed")
        raise HTTPException(500, f"could not sign upload URL: {e}")
    redis_client.record_upload(req.sha256, req.size, req.content_type, object_path)
    return {
        "cached": False,
        "status": "pending_upload",
        "sha256": req.sha256,
        "object_path": object_path,
        "need_upload": True,
        "signed_put_url": put_url,
        "expires_in": 900,
    }


@app.post("/jobs")
def create_job(req: JobRequest):
    if not redis_client.user_exists(req.username):
        raise HTTPException(404, "unknown username")
    if not redis_client.valid_sha256(req.sha256):
        raise HTTPException(400, "sha256 must be 64 hex chars")

    upload = redis_client.get_upload(req.sha256)
    if not upload or not upload.get("object_path"):
        raise HTTPException(404, "no upload recorded for this sha256; call POST /uploads first")
    object_path = upload["object_path"]
    if not gcs_signed.object_exists(object_path):
        raise HTTPException(412, "upload file not found in storage; retry POST /uploads and re-upload")

    existing_video = redis_client.get_video(req.sha256)
    if existing_video:
        status = existing_video.get("status")
        if status == "done":
            return {
                "status": "done",
                "sha256": req.sha256,
                "video_url": existing_video.get("video_url"),
                "cached": True,
            }
        if status and status != "failed":
            return {
                "status": status,
                "sha256": req.sha256,
                "job_id": existing_video.get("job_id"),
                "cached": True,
            }

    job_id = uuid.uuid4().hex[:12]
    redis_client.create_job(job_id, req.sha256, object_path, req.username)
    redis_client.claim_video(req.sha256, job_id)
    redis_client.append_user_job(req.username, job_id)
    tasks.enqueue(job_id, req.sha256, object_path)
    return {"status": "queued", "sha256": req.sha256, "job_id": job_id, "cached": False}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = redis_client.get_job(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    if job.get("status") == "done":
        video = redis_client.get_video(job["sha256"])
        if video:
            job["video_url"] = video.get("video_url")
    return job


@app.get("/users/{username}/jobs")
def list_jobs(username: str, limit: int = 20):
    if not redis_client.user_exists(username):
        raise HTTPException(404, "unknown username")
    ids = redis_client.list_user_job_ids(username, limit)
    jobs = []
    for jid in ids:
        j = redis_client.get_job(jid)
        if j:
            if j.get("status") == "done":
                v = redis_client.get_video(j["sha256"])
                if v:
                    j["video_url"] = v.get("video_url")
            jobs.append(j)
    return {"jobs": jobs}

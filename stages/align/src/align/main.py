from __future__ import annotations

import os
import threading

if "SSL_CERT_FILE" not in os.environ:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

from fastapi import Body, Depends, FastAPI, HTTPException

from shared import create_logger, flush_logs, verify_stage_auth
from shared.schemas import validate, ValidationError

from . import pipeline

log = create_logger("align")
app = FastAPI(title="annemusic-align")


@app.get("/ping")
def ping() -> dict:
    return {"ok": True, "service": "align"}



# One job at a time: model singletons and (on GPU deploys) a single L4 leave
# no headroom for concurrent runs. The handler below is sync (def), so
# FastAPI executes it on the threadpool and the event loop stays free —
# /ping and health probes answer mid-job, unlike the old async-def handler
# that froze the loop for the whole run. The lock serializes the jobs that
# Cloud Run lets through (--concurrency 1 guards the front door in prod).
_job_lock = threading.Lock()

@app.post("/process", dependencies=[Depends(verify_stage_auth)])
def process(body: dict = Body(...)) -> dict:
    try:
        validate(body, "align_request")
    except ValidationError as e:
        log.error(None, "invalid request", e, {"path": list(e.absolute_path)})
        raise HTTPException(status_code=400, detail=f"contract violation: {e.message}")
    job_id = body["job_id"]
    try:
        with _job_lock:
            result = pipeline.run(
                job_id=job_id,
                vocals_uri=body["vocals_uri"],
                segments=body["segments"],
                language=body["language"],
                vocal_activity=body["vocal_activity"],
            )
    except Exception as e:
        log.error(job_id, "pipeline failed", e)
        flush_logs()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return _validated(result)


def _validated(result: dict) -> dict:
    """Producer-side contract check: a malformed response fails HERE with a
    named violation instead of one stage downstream (or never)."""
    try:
        validate(result, "align_response")
    except ValidationError as e:
        log.error(result.get("job_id"), "response contract violation", e,
                  {"path": list(e.absolute_path)})
        flush_logs()
        raise HTTPException(status_code=500, detail=f"response contract violation: {e.message}")
    return result


def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", "8103"))
    uvicorn.run("align.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

from __future__ import annotations

import os

if "SSL_CERT_FILE" not in os.environ:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

from fastapi import FastAPI, HTTPException, Request

from shared import create_logger, flush_logs
from shared.schemas import validate, ValidationError

from . import pipeline

log = create_logger("record-mix")
app = FastAPI(title="annemusic-record-mix")


@app.get("/ping")
def ping() -> dict:
    return {"ok": True, "service": "record-mix"}


@app.post("/process")
async def process(request: Request) -> dict:
    body = await request.json()
    try:
        validate(body, "record_mix_request")
    except ValidationError as e:
        log.error(None, "invalid request", e, {"path": list(e.absolute_path)})
        raise HTTPException(status_code=400, detail=f"contract violation: {e.message}")
    job_id = body["job_id"]
    try:
        return pipeline.run(
            job_id=job_id,
            recording_uri=body["recording_uri"],
            instrumental_uri=body["instrumental_uri"],
            autotune=body.get("autotune", "off"),
            gain_db=float(body.get("gain_db", 0.0)),
        )
    except Exception as e:
        log.error(job_id, "pipeline failed", e)
        flush_logs()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", "8105"))
    uvicorn.run("record_mix.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

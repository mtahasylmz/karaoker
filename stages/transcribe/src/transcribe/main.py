from __future__ import annotations

import os

# macOS CA bundle hand-off for urllib + requests used by HF model downloads.
if "SSL_CERT_FILE" not in os.environ:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

from fastapi import FastAPI, HTTPException, Request

from shared import create_logger, flush_logs
from shared.schemas import validate, ValidationError

from . import pipeline

log = create_logger("transcribe")
app = FastAPI(title="annemusic-transcribe")


@app.get("/ping")
def ping() -> dict:
    return {"ok": True, "service": "transcribe", "model": os.environ.get("WHISPER_MODEL", "small")}


@app.post("/process")
async def process(request: Request) -> dict:
    body = await request.json()
    try:
        validate(body, "transcribe_request")
    except ValidationError as e:
        log.error(None, "invalid request", e, {"path": list(e.absolute_path)})
        raise HTTPException(status_code=400, detail=f"contract violation: {e.message}")
    job_id = body["job_id"]
    try:
        return pipeline.run(
            job_id=job_id,
            vocals_uri=body["vocals_uri"],
            language=body.get("language"),
            known_lyrics=body.get("known_lyrics"),
            title=body.get("title"),
            artist=body.get("artist"),
        )
    except Exception as e:
        log.error(job_id, "pipeline failed", e)
        flush_logs()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", "8102"))
    uvicorn.run("transcribe.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

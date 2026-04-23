"""HTTP server for the separate stage. Single endpoint: POST /process.

Validates the request against the SeparateRequest JSON schema (emitted from
packages/contracts), runs the pipeline, returns a SeparateResponse.
"""

from __future__ import annotations

import os

# macOS Python can't find system CAs by default; torch.hub + demucs downloads
# fail with CERTIFICATE_VERIFY_FAILED. Point urllib at certifi's bundle so
# subprocess children inherit it too. Harmless on Linux.
if "SSL_CERT_FILE" not in os.environ:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

from fastapi import FastAPI, HTTPException, Request

from shared import create_logger, flush_logs
from shared.schemas import validate, ValidationError

from . import pipeline

log = create_logger("separate")

app = FastAPI(title="annemusic-separate")


@app.get("/ping")
def ping() -> dict:
    return {
        "ok": True,
        "service": "separate",
        "model": os.environ.get("SEPARATE_MODEL", pipeline.DEFAULT_MODEL),
    }


@app.post("/process")
async def process(request: Request) -> dict:
    body = await request.json()
    try:
        validate(body, "separate_request")
    except ValidationError as e:
        log.error(None, "invalid request", e, {"path": list(e.absolute_path)})
        raise HTTPException(status_code=400, detail=f"contract violation: {e.message}")

    job_id = body["job_id"]
    try:
        return pipeline.run(job_id, body["source_uri"], body.get("model"))
    except Exception as e:
        log.error(job_id, "pipeline failed", e)
        flush_logs()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


def main() -> None:  # python -m separate
    import uvicorn
    port = int(os.environ.get("PORT", "8101"))
    uvicorn.run(
        "separate.main:app",
        host="0.0.0.0",
        port=port,
        reload=bool(os.environ.get("RELOAD")),
    )


if __name__ == "__main__":
    main()

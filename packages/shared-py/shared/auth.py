"""Shared bearer-token gate for stage /process endpoints.

Stages deploy on Cloud Run with --allow-unauthenticated because the
orchestrator's context.call() executes from Upstash's infrastructure, which
cannot mint Google OIDC identity tokens for the Cloud Run audience. Callers
(orchestrator, apps/api stage proxy) instead send
`Authorization: Bearer $STAGE_AUTH_TOKEN`, verified here at the app layer.

Unset STAGE_AUTH_TOKEN = gate open (local dev). deploy-stage.sh refuses to
deploy without one.
"""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request


def verify_stage_auth(request: Request) -> None:
    """FastAPI dependency: raise 401 unless the shared stage token matches."""
    token = os.environ.get("STAGE_AUTH_TOKEN", "")
    if not token:
        return
    got = request.headers.get("authorization", "")
    if not hmac.compare_digest(got, f"Bearer {token}"):
        raise HTTPException(status_code=401, detail="missing or invalid stage token")

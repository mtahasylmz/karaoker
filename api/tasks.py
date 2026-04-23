"""Enqueue a background karaoke job.

Production: Cloud Tasks with OIDC auth targeting the worker's /process.
Local dev (TASKS_QUEUE unset): fire-and-forget HTTP POST to WORKER_URL.
"""

import json
import os

import httpx

_tasks_client = None


def _cloud_tasks():
    global _tasks_client
    if _tasks_client is None:
        from google.cloud import tasks_v2  # lazy import — heavy dep
        _tasks_client = tasks_v2.CloudTasksClient()
    return _tasks_client


def enqueue(job_id: str, sha256: str, object_path: str) -> None:
    body = {"job_id": job_id, "sha256": sha256, "object_path": object_path}
    worker_url = os.environ["WORKER_URL"].rstrip("/") + "/process"
    queue = os.environ.get("TASKS_QUEUE")

    if not queue:
        # Local dev: fire and forget, don't block the HTTP response.
        with httpx.Client(timeout=5.0) as client:
            try:
                client.post(worker_url, json=body, timeout=2.0)
            except httpx.ReadTimeout:
                pass  # expected — worker processes asynchronously
        return

    from google.cloud import tasks_v2
    project = os.environ["GCP_PROJECT"]
    region = os.environ["GCP_REGION"]
    invoker_sa = os.environ["TASKS_INVOKER_SA"]

    parent = _cloud_tasks().queue_path(project, region, queue)
    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": worker_url,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body).encode(),
            "oidc_token": {
                "service_account_email": invoker_sa,
                "audience": worker_url.rsplit("/", 1)[0],
            },
        },
        "dispatch_deadline": {"seconds": 1800},
    }
    _cloud_tasks().create_task(parent=parent, task=task)

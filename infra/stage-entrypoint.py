"""Cloud Run entrypoint for ML stages.

If CODE_GCS_PATH is set, syncs the stage's Python source from GCS over the
baked-in `/app/stages/<STAGE_NAME>/src/<PY_MODULE>/` directory before
execing uvicorn. This is the "code-from-GCS" iteration pattern: the heavy
deps stay baked in the image; only the editable Python source moves.

Production deploys leave CODE_GCS_PATH unset → entrypoint just exec's
uvicorn against the baked code. Dev deploys set it to `code/<stage>` so
`bash infra/dev-sync.sh <stage>` can rsync edits without a Cloud Build
round trip.

Required env at runtime:
  STAGE_NAME   — separate | transcribe | align | record-mix
  PY_MODULE    — separate | transcribe | align | record_mix
  GCS_BUCKET   — set anyway by the stage; reused here

Optional:
  CODE_GCS_PATH — gs://${GCS_BUCKET}/${CODE_GCS_PATH}/ rsync source
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _sync_code() -> None:
    code_path = os.environ.get("CODE_GCS_PATH")
    if not code_path:
        return
    bucket_name = os.environ["GCS_BUCKET"]
    stage = os.environ["STAGE_NAME"]
    py_module = os.environ["PY_MODULE"]
    dst = Path(f"/app/stages/{stage}/src/{py_module}")

    # Lazy import — only the dev path pays the SDK import cost.
    from google.cloud import storage  # type: ignore[import-not-found]

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    prefix = code_path.strip("/") + "/"
    started = time.monotonic()
    fetched = skipped = 0

    for blob in client.list_blobs(bucket, prefix=prefix):
        rel = blob.name[len(prefix):]
        if not rel or rel.endswith("/"):
            continue
        local = dst / rel
        # blob.size is set by list_blobs; skip if already-correct.
        if local.exists() and local.stat().st_size == (blob.size or -1):
            skipped += 1
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local))
        fetched += 1

    elapsed = time.monotonic() - started
    print(
        f"[entrypoint] CODE_GCS_PATH={code_path} synced: "
        f"{fetched} new/changed, {skipped} unchanged ({elapsed:.1f}s)",
        flush=True,
    )


def main() -> int:
    _sync_code()
    py_module = os.environ["PY_MODULE"]
    # The image's `uv sync` step landed the venv at /app/.venv. Use its
    # interpreter directly — avoids a second `uv run` invocation and
    # guarantees `google.cloud.storage` (used above for the sync) is on
    # path because shared-py pulls it.
    venv_python = os.environ.get("VENV_PYTHON", "/app/.venv/bin/python")
    cmd = [venv_python, "-m", f"{py_module}.main"]
    print(f"[entrypoint] exec {' '.join(cmd)}", flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())

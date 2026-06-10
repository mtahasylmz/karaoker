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
  CODE_GCS_PATH   — gs://${GCS_BUCKET}/${CODE_GCS_PATH}/ rsync source
  MODEL_CACHE_SRC — path to the GCS-FUSE-mounted HF cache mirror (filled by
                    infra/model-fetch); copied into local HF_HOME at boot
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _seed_model_cache() -> None:
    """Copy the HF cache mirror (read-only GCS FUSE mount, populated by
    infra/model-fetch) into local HF_HOME, so model loads hit a warm,
    writable cache. A same-region FUSE read beats re-downloading multi-GB
    weights from HuggingFace on every cold start, and a plain file copy
    sidesteps gcsfuse's flock/write semantics entirely (HF's filelock does
    not work on FUSE). Cloud Run's writable filesystem is per-instance
    tmpfs, so this runs once per instance, not once per revision.

    No-op when MODEL_CACHE_SRC is unset (local dev, CPU deploys) or the
    mount is missing.
    """
    src = os.environ.get("MODEL_CACHE_SRC")
    if not src:
        return
    src_path = Path(src)
    if not src_path.is_dir():
        print(f"[entrypoint] MODEL_CACHE_SRC={src} not mounted; skipping seed", flush=True)
        return
    import shutil

    dst = Path(os.environ.get("HF_HOME", "/app/.cache/huggingface"))
    started = time.monotonic()
    copied = skipped = 0
    copied_bytes = 0
    for p in src_path.rglob("*"):
        if not p.is_file():
            continue
        local = dst / p.relative_to(src_path)
        size = p.stat().st_size
        if local.exists() and local.stat().st_size == size:
            skipped += 1
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, local)
        copied += 1
        copied_bytes += size
    elapsed = time.monotonic() - started
    print(
        f"[entrypoint] model cache seeded from {src}: {copied} files "
        f"({copied_bytes / 1e9:.2f} GB) copied, {skipped} already present "
        f"({elapsed:.1f}s)",
        flush=True,
    )


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
    _seed_model_cache()
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

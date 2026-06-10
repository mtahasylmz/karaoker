"""HF → GCS model fetcher (run as a Cloud Run Job).

Downloads each requested HuggingFace repo into the HF cache layout under
$HF_HOME, then mirrors that directory to gs://${GCS_BUCKET}/${GCS_PREFIX}/.
Preserves the `models--<org>--<repo>/snapshots/<sha>/` structure so when
the stages mount the bucket back as HF_HOME, transformers / faster-whisper
find the cached entry without re-downloading.

HF→GCP runs on Google's backbone (~50-200 MiB/s) so this finishes in
minutes for multi-GB models. hf_transfer accelerates the HF side.

Usage (env-var driven, no flags):
  GCS_BUCKET, GCS_PREFIX (default "models/hf"),
  HF_REPOS = comma-separated HF repo IDs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
HF_HOME = Path(os.environ.get("HF_HOME", "/tmp/hf"))
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME)

from huggingface_hub import snapshot_download  # noqa: E402
from google.cloud import storage  # noqa: E402

BUCKET = os.environ["GCS_BUCKET"]
PREFIX = os.environ.get("GCS_PREFIX", "models/hf").strip("/")
REPOS = [r.strip() for r in os.environ.get("HF_REPOS", "").split(",") if r.strip()]
if not REPOS:
    print("HF_REPOS env var is empty; nothing to do.", flush=True)
    sys.exit(0)

client = storage.Client()
bucket = client.bucket(BUCKET)


def upload_dir(local: Path, prefix: str) -> int:
    """Upload everything under `local/` to gs://{BUCKET}/{prefix}/...
    Skips objects whose size already matches on GCS — makes reruns cheap."""
    n = 0
    for p in local.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(local).as_posix()
        blob_name = f"{prefix}/{rel}"
        blob = bucket.blob(blob_name)
        if blob.exists():
            blob.reload()
            if blob.size == p.stat().st_size:
                continue
        blob.upload_from_filename(str(p))
        n += 1
        if n % 25 == 0:
            print(f"  uploaded {n} files…", flush=True)
    return n


for repo in REPOS:
    print(f"==> fetching {repo}", flush=True)
    snapshot_download(repo_id=repo, cache_dir=str(HF_HOME / "hub"))
    print(f"==> uploading {repo}", flush=True)
    n = upload_dir(HF_HOME, PREFIX)
    print(f"   {n} new/changed files for {repo}", flush=True)

print("done", flush=True)

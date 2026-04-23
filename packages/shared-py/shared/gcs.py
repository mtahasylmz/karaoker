"""GCS helpers used by Python stages (download source, upload outputs)."""

from __future__ import annotations

import os
from pathlib import Path

from google.cloud import storage as gcs_storage

from .env import required

_client: gcs_storage.Client | None = None


def storage() -> gcs_storage.Client:
    global _client
    if _client is None:
        _client = gcs_storage.Client()
    return _client


def bucket() -> gcs_storage.Bucket:
    return storage().bucket(required("GCS_BUCKET"))


def public_url(object_path: str) -> str:
    return f"https://storage.googleapis.com/{required('GCS_BUCKET')}/{object_path}"


def object_exists(object_path: str) -> bool:
    return bucket().blob(object_path).exists()


def download_file(object_path: str, local_path: str | Path) -> Path:
    local = Path(local_path)
    bucket().blob(object_path).download_to_filename(str(local))
    return local


def upload_file(object_path: str, local_path: str | Path, content_type: str = "application/octet-stream") -> str:
    blob = bucket().blob(object_path)
    blob.upload_from_filename(str(local_path), content_type=content_type)
    return public_url(object_path)

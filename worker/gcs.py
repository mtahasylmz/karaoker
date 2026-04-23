"""Upload final mp4 to GCS and return a URL the browser can play."""

import datetime as dt
import os
from pathlib import Path

from google.cloud import storage

_client: storage.Client | None = None


def _bucket():
    global _client
    if _client is None:
        _client = storage.Client()
    return _client.bucket(os.environ["GCS_BUCKET"])


def exists(object_path: str) -> bool:
    return _bucket().blob(object_path).exists()


def download(object_path: str, local_path: str | Path) -> Path:
    blob = _bucket().blob(object_path)
    local_path = Path(local_path)
    blob.download_to_filename(str(local_path))
    return local_path


def upload(object_path: str, local_path: str | Path, content_type: str = "video/mp4") -> str:
    blob = _bucket().blob(object_path)
    blob.upload_from_filename(str(local_path), content_type=content_type)
    return url_for(object_path)


def url_for(object_path: str) -> str:
    mode = os.environ.get("GCS_URL_MODE", "public")
    bucket_name = os.environ["GCS_BUCKET"]
    if mode == "signed":
        blob = _bucket().blob(object_path)
        return blob.generate_signed_url(
            version="v4",
            expiration=dt.timedelta(hours=24),
            method="GET",
        )
    return f"https://storage.googleapis.com/{bucket_name}/{object_path}"

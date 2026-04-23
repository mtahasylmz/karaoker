"""GCS helpers used by Python stages (download source, upload outputs).

Dev mode: if `DEV_FS_ROOT` is set, read/write go to the local filesystem
under that root instead of GCS. Lets stages run end-to-end without GCS
credentials. The object_path is kept identical — stages don't need to know
which mode they're in.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import storage as gcs_storage

from .env import required

_client: gcs_storage.Client | None = None


def _dev_root() -> Path | None:
    root = os.environ.get("DEV_FS_ROOT")
    return Path(root) if root else None


def storage() -> gcs_storage.Client:
    global _client
    if _client is None:
        _client = gcs_storage.Client()
    return _client


def bucket() -> gcs_storage.Bucket:
    return storage().bucket(required("GCS_BUCKET"))


def public_url(object_path: str) -> str:
    root = _dev_root()
    if root is not None:
        return f"file://{(root / object_path).resolve()}"
    return f"https://storage.googleapis.com/{required('GCS_BUCKET')}/{object_path}"


def object_exists(object_path: str) -> bool:
    root = _dev_root()
    if root is not None:
        return (root / object_path).exists()
    return bucket().blob(object_path).exists()


def download_file(object_path: str, local_path: str | Path) -> Path:
    local = Path(local_path)
    local.parent.mkdir(parents=True, exist_ok=True)
    root = _dev_root()
    if root is not None:
        src = root / object_path
        if not src.exists():
            raise FileNotFoundError(f"dev fs: {src} not found under DEV_FS_ROOT")
        shutil.copy(src, local)
        return local
    bucket().blob(object_path).download_to_filename(str(local))
    return local


def upload_file(
    object_path: str,
    local_path: str | Path,
    content_type: str = "application/octet-stream",
) -> str:
    root = _dev_root()
    if root is not None:
        dst = root / object_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_path, dst)
        return f"file://{dst.resolve()}"
    blob = bucket().blob(object_path)
    blob.upload_from_filename(str(local_path), content_type=content_type)
    return public_url(object_path)


def object_path_from_gs_uri(uri: str) -> str:
    """Convert gs://bucket/path/to/file → path/to/file (strips bucket)."""
    if uri.startswith("file://"):
        # Dev: "file:///abs/path" — strip the scheme; stages will treat as path.
        return uri[len("file://"):]
    parsed = urlparse(uri)
    if parsed.scheme != "gs":
        raise ValueError(f"expected gs:// uri, got {uri!r}")
    return parsed.path.lstrip("/")

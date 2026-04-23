"""V4 signed PUT URL generation for direct browser uploads to GCS.

On Cloud Run the runtime credentials are Compute Engine metadata tokens that
can't sign locally. We pass them through to generate_signed_url as
(service_account_email, access_token), which makes google-cloud-storage call
the IAM signBlob API. That requires roles/iam.serviceAccountTokenCreator on
the SA — granted in infra/setup.sh.
"""

import datetime as dt
import os

from google.auth import default as google_default
from google.auth.transport.requests import Request as AuthRequest
from google.cloud import storage

_client: storage.Client | None = None
_credentials = None


def _bucket():
    global _client
    if _client is None:
        _client = storage.Client()
    return _client.bucket(os.environ["GCS_BUCKET"])


def _creds():
    global _credentials
    if _credentials is None or not getattr(_credentials, "valid", False):
        creds, _ = google_default()
        creds.refresh(AuthRequest())
        _credentials = creds
    return _credentials


def signed_put_url(object_path: str, content_type: str, expires_in_seconds: int = 900) -> str:
    creds = _creds()
    blob = _bucket().blob(object_path)
    return blob.generate_signed_url(
        version="v4",
        expiration=dt.timedelta(seconds=expires_in_seconds),
        method="PUT",
        content_type=content_type,
        service_account_email=creds.service_account_email,
        access_token=creds.token,
    )


def public_url(object_path: str) -> str:
    return f"https://storage.googleapis.com/{os.environ['GCS_BUCKET']}/{object_path}"


def object_exists(object_path: str) -> bool:
    return _bucket().blob(object_path).exists()

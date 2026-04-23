"""Contract validation: minimal-valid, maximal-valid, rejection cases."""

from __future__ import annotations

import pytest

from shared.schemas import validate, ValidationError


JOB_ID = "a1b2c3d4e5f6"
BUCKET_URI_REC = "gs://bkt/uploads/rec.webm"
BUCKET_URI_INST = "gs://bkt/stages/separate/job/no_vocals.wav"
BUCKET_URI_VOC = "gs://bkt/stages/separate/job/vocals.wav"


def test_minimal_valid():
    body = {
        "job_id": JOB_ID,
        "recording_uri": BUCKET_URI_REC,
        "instrumental_uri": BUCKET_URI_INST,
    }
    validate(body, "record_mix_request")


def test_maximal_valid_all_edges():
    body = {
        "job_id": JOB_ID,
        "recording_uri": BUCKET_URI_REC,
        "instrumental_uri": BUCKET_URI_INST,
        "vocals_uri": BUCKET_URI_VOC,
        "autotune": "smooth",
        "clean_bleed": True,
        "gain_db": 24,
        "mix": {
            "vocal_gain_db": 12,
            "instrumental_gain_db": -12,
            "reverb_wet": 1,
            "duck_db": 12,
            "presence_db": 6,
        },
    }
    validate(body, "record_mix_request")


def test_rejects_missing_job_id():
    body = {
        "recording_uri": BUCKET_URI_REC,
        "instrumental_uri": BUCKET_URI_INST,
    }
    with pytest.raises(ValidationError):
        validate(body, "record_mix_request")


def test_rejects_reverb_wet_out_of_range():
    body = {
        "job_id": JOB_ID,
        "recording_uri": BUCKET_URI_REC,
        "instrumental_uri": BUCKET_URI_INST,
        "mix": {"reverb_wet": 1.5},
    }
    with pytest.raises(ValidationError):
        validate(body, "record_mix_request")


def test_rejects_unknown_autotune():
    body = {
        "job_id": JOB_ID,
        "recording_uri": BUCKET_URI_REC,
        "instrumental_uri": BUCKET_URI_INST,
        "autotune": "bogus",
    }
    with pytest.raises(ValidationError):
        validate(body, "record_mix_request")


def test_rejects_bad_gcs_uri():
    body = {
        "job_id": JOB_ID,
        "recording_uri": "https://example.com/rec.webm",  # not gs://
        "instrumental_uri": BUCKET_URI_INST,
    }
    with pytest.raises(ValidationError):
        validate(body, "record_mix_request")

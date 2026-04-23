"""End-to-end smoke: pipeline.run() on synthetic WAVs produces a playable mp3.

Uses DEV_FS_ROOT so GCS calls are short-circuited to the local filesystem.
No ffmpeg mocks — real binary, real filter graph, real Demucs-free path
(we skip clean_bleed in this smoke to keep pytest fast).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from record_mix import pipeline


def _duration_s(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def test_smoke_defaults(dev_fs_root: Path, job_id: str):
    result = pipeline.run(
        job_id=job_id,
        recording_uri="gs://test-bucket/uploads/recording.wav",
        instrumental_uri="gs://test-bucket/stages/instrumental.wav",
        vocals_uri="gs://test-bucket/stages/vocals.wav",
        autotune="off",
        clean_bleed=False,  # skip Demucs in smoke — covered by bench, not pytest
        gain_db=0.0,
        mix={"reverb_wet": 0.0, "duck_db": 4.0},
    )
    assert result["stage"] == "record-mix"
    assert result["mix_uri"].endswith("/mix.mp3")
    mix_path = dev_fs_root / result["mix_uri"].split("test-bucket/", 1)[1]
    assert mix_path.exists(), f"expected mix at {mix_path}"
    dur = _duration_s(mix_path)
    # Synthetic fixtures are 2 seconds; loudnorm + limiter preserve duration.
    assert abs(dur - 2.0) < 0.3, f"unexpected mp3 duration {dur:.3f} s"
    diag = result["diagnostics"]
    assert diag["skipped"] == []  # full happy path, nothing skipped
    assert "alignment_offset_ms" in diag["applied"] or "align_sync_no_signal" in diag["skipped"]
    assert "loudnorm" in diag["applied"]


def test_smoke_with_reverb(dev_fs_root: Path, job_id: str):
    result = pipeline.run(
        job_id=job_id,
        recording_uri="gs://test-bucket/uploads/recording.wav",
        instrumental_uri="gs://test-bucket/stages/instrumental.wav",
        autotune="off",
        clean_bleed=False,
        gain_db=0.0,
        mix={"reverb_wet": 0.3},
    )
    mix_path = dev_fs_root / result["mix_uri"].split("test-bucket/", 1)[1]
    assert mix_path.exists()
    assert "reverb_asset_missing" not in result["diagnostics"]["skipped"]

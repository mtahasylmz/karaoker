"""Shared test fixtures: synthesize tiny WAVs so pytest stays offline + fast."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


FS = 48_000
DURATION_S = 2.0


def _sine(freq_hz: float, duration_s: float = DURATION_S, fs: int = FS,
          noise_db: float = -40.0) -> np.ndarray:
    n = int(fs * duration_s)
    t = np.arange(n, dtype=np.float32) / fs
    y = 0.5 * np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    rng = np.random.default_rng(0xA11EA6)
    noise = rng.standard_normal(n).astype(np.float32) * (10 ** (noise_db / 20.0))
    return y + noise


def _write_wav(path: Path, mono: np.ndarray, fs: int = FS) -> None:
    stereo = np.stack([mono, mono], axis=1)
    sf.write(str(path), stereo.astype(np.float32), fs, subtype="PCM_16")


@pytest.fixture
def tmp_wav_dir(tmp_path: Path) -> Path:
    """Write synthetic recording/instrumental/vocals WAVs into `tmp_path`."""
    rec = _sine(220.0)
    voc = _sine(220.0)
    inst = _sine(440.0) * 0.6 + _sine(660.0) * 0.3
    _write_wav(tmp_path / "recording.wav", rec)
    _write_wav(tmp_path / "vocals.wav", voc)
    _write_wav(tmp_path / "instrumental.wav", inst)
    return tmp_path


@pytest.fixture
def dev_fs_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tmp_wav_dir: Path) -> Path:
    """Stage the WAVs under a DEV_FS_ROOT layout with gs://-style object paths."""
    root = tmp_path / "fs"
    root.mkdir()
    monkeypatch.setenv("DEV_FS_ROOT", str(root))
    monkeypatch.setenv("GCS_BUCKET", "test-bucket")
    # Shared helpers unwrap gs:// to "path/under/bucket" → the object_path is
    # whatever sits after the bucket segment.
    (root / "uploads").mkdir()
    (root / "stages").mkdir()
    import shutil
    shutil.copy(tmp_wav_dir / "recording.wav", root / "uploads" / "recording.wav")
    shutil.copy(tmp_wav_dir / "vocals.wav", root / "stages" / "vocals.wav")
    shutil.copy(tmp_wav_dir / "instrumental.wav", root / "stages" / "instrumental.wav")
    return root


@pytest.fixture
def job_id() -> str:
    return secrets.token_hex(8)  # 16 hex chars — within contract [12,32] range

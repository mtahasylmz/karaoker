"""Build recording/vocals/instrumental fixtures for the record-mix bench.

Default path: synthesize a 10-second "song" deterministically — a vocal sine
melody + chordal instrumental. No external downloads, no dep on `musdb` or a
copyrighted track. The vocal is then delayed 55 ms and noise-tagged to form
the "user recording" so GCC-PHAT alignment has something meaningful to find.

Upgrade path: install `musdb` (`uv pip install musdb` inside the workspace
venv) and set `ANNEMUSIC_BENCH_USE_MUSDB=1` to pull the MUSDB 7-track sample
instead. Will auto-download weights + run htdemucs on the first track.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FS = 48_000
DURATION_S = 10.0
INTRODUCED_OFFSET_MS = 55.0
ADDED_NOISE_DB = -32.0


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    if os.environ.get("ANNEMUSIC_BENCH_USE_MUSDB") == "1":
        _build_from_musdb()
    else:
        _build_synthetic()
    print("fixtures ready:")
    for name in ("recording.webm", "vocals.wav", "instrumental.wav"):
        p = FIXTURES_DIR / name
        print(f"  {p.relative_to(FIXTURES_DIR.parent)}  ({p.stat().st_size/1024:.1f} KB)")


def _build_synthetic() -> None:
    """Deterministic synthetic fixture — no network, no demucs."""
    n = int(FS * DURATION_S)
    t = np.arange(n, dtype=np.float32) / FS

    melody_notes = [220.0, 247.0, 262.0, 294.0, 330.0, 294.0, 262.0, 247.0]
    note_len = n // len(melody_notes)
    vocals = np.zeros(n, dtype=np.float32)
    for i, f in enumerate(melody_notes):
        start, end = i * note_len, min((i + 1) * note_len, n)
        seg = np.sin(2 * np.pi * f * t[start:end]).astype(np.float32)
        # Smooth note boundaries so there are no transient clicks.
        env = np.ones_like(seg)
        fade = min(2000, len(seg) // 4)
        if fade > 0:
            env[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            env[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        vocals[start:end] = seg * env * 0.45

    chord_freqs = [130.81, 164.81, 196.0]
    instrumental = 0.18 * sum(
        np.sin(2 * np.pi * f * t).astype(np.float32) for f in chord_freqs
    )
    # Mild amplitude modulation so the instrumental isn't a pure DC tone.
    instrumental *= 0.75 + 0.25 * np.sin(2 * np.pi * 1.1 * t).astype(np.float32)

    _write_stereo(FIXTURES_DIR / "vocals.wav", vocals)
    _write_stereo(FIXTURES_DIR / "instrumental.wav", instrumental)

    # Synthetic "recording" — vocals + delay + noise + a tiny gain bump.
    shift = int(FS * INTRODUCED_OFFSET_MS / 1000.0)
    shifted = np.concatenate([np.zeros(shift, dtype=np.float32), vocals])[:n]
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n).astype(np.float32) * (10 ** (ADDED_NOISE_DB / 20.0))
    recording = shifted * 0.9 + noise
    rec_wav = FIXTURES_DIR / "recording.wav"
    _write_mono(rec_wav, recording)

    rec_webm = FIXTURES_DIR / "recording.webm"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(rec_wav),
        "-c:a", "libopus", "-b:a", "96k", str(rec_webm),
    ], check=True, capture_output=True)
    rec_wav.unlink()


def _build_from_musdb() -> None:
    try:
        import musdb
    except ImportError as e:
        raise SystemExit(
            "ANNEMUSIC_BENCH_USE_MUSDB=1 but `musdb` is not installed. "
            "Run `uv pip install musdb` or unset the env var."
        ) from e

    db = musdb.DB(download=True)
    track = db.tracks[0]
    audio = track.audio.astype(np.float32)
    sr = track.rate
    trimmed = FIXTURES_DIR / "source_trimmed.wav"
    sf.write(str(trimmed), audio[: int(sr * DURATION_S)], sr, subtype="FLOAT")

    demucs_out = FIXTURES_DIR / "demucs"
    subprocess.run([
        "python", "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
        "-o", str(demucs_out), str(trimmed),
    ], check=True)
    stem_dir = next((demucs_out / "htdemucs").iterdir())
    (stem_dir / "vocals.wav").replace(FIXTURES_DIR / "vocals.wav")
    (stem_dir / "no_vocals.wav").replace(FIXTURES_DIR / "instrumental.wav")

    voc, _ = sf.read(str(FIXTURES_DIR / "vocals.wav"), dtype="float32", always_2d=True)
    shift = int(sr * INTRODUCED_OFFSET_MS / 1000.0)
    shifted = np.concatenate([np.zeros((shift, voc.shape[1]), dtype=np.float32), voc])
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(shifted.shape).astype(np.float32) * (10 ** (ADDED_NOISE_DB / 20.0))
    rec = (shifted + noise).mean(axis=1, keepdims=True).astype(np.float32)
    rec_wav = FIXTURES_DIR / "recording.wav"
    sf.write(str(rec_wav), rec, sr, subtype="PCM_16")
    rec_webm = FIXTURES_DIR / "recording.webm"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(rec_wav),
        "-c:a", "libopus", "-b:a", "96k", str(rec_webm),
    ], check=True, capture_output=True)
    rec_wav.unlink()


def _write_stereo(path: Path, mono: np.ndarray) -> None:
    stereo = np.stack([mono, mono], axis=1).astype(np.float32)
    sf.write(str(path), stereo, FS, subtype="PCM_16")


def _write_mono(path: Path, mono: np.ndarray) -> None:
    sf.write(str(path), mono.astype(np.float32), FS, subtype="PCM_16")


if __name__ == "__main__":
    main()

"""Materialize MUSDB18 sample tracks as WAV fixtures for the bench.

The `musdb` PyPI package auto-downloads a 7-track sample (4 train, 3 test,
~30 s each, 44.1 kHz stereo, stems in STEMS-mp4). We decode each to per-stem
WAVs under ../fixtures/{name}/{mixture,vocals,no_vocals}.wav so that the
separator under test (which takes file paths) and museval (which compares
numpy arrays) both have what they need.

Notes
- This is the low-res MUSDB sample, not MUSDB18-HQ. Absolute SDR won't match
  published numbers; relative rankings across models are still meaningful.
- For a real evaluation point upgrade to MUSDB18-HQ (separately licensed;
  ~30 GB of WAV) and re-run.
"""
from __future__ import annotations

from pathlib import Path

import musdb
import numpy as np
import soundfile as sf

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    db = musdb.DB(download=True)  # ships 7-track sample on first call
    print(f"musdb sample loaded: {len(db.tracks)} tracks at {db.root}")

    for track in db.tracks:
        name = _slugify(track.name)
        out = FIXTURES_DIR / name
        out.mkdir(exist_ok=True)
        sr = track.rate

        mixture = track.audio.astype(np.float32)
        vocals = track.targets["vocals"].audio.astype(np.float32)
        # MUSDB stems: drums + bass + other = "accompaniment" = no_vocals
        no_vocals = track.targets["accompaniment"].audio.astype(np.float32)

        sf.write(out / "mixture.wav", mixture, sr, subtype="FLOAT")
        sf.write(out / "vocals.wav", vocals, sr, subtype="FLOAT")
        sf.write(out / "no_vocals.wav", no_vocals, sr, subtype="FLOAT")
        dur = mixture.shape[0] / sr
        print(f"  {name}: {dur:.1f}s @ {sr} Hz, {mixture.shape[1]}ch")


def _slugify(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s).strip("_")


if __name__ == "__main__":
    main()

"""Synthetic-audio tests for the RMS-VAD. Real-fixture eval lives in bench/."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from transcribe import vad


def _synthesize(pattern: list[tuple[float, float, float]], sr: int = 16000) -> Path:
    """Render a mono wav from a list of (start, end, rms) triples.

    Everything outside listed regions is silence. RMS is the target loudness
    of a 440 Hz sine within the region.
    """
    total = max(end for _, end, _ in pattern)
    n = int(total * sr)
    t = np.arange(n) / sr
    out = np.zeros(n, dtype=np.float32)
    for start, end, rms in pattern:
        if rms <= 0:
            continue
        i0, i1 = int(start * sr), int(end * sr)
        amp = float(rms) * np.sqrt(2.0)
        out[i0:i1] += amp * np.sin(2 * np.pi * 440 * t[i0:i1]).astype(np.float32)
    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    sf.write(str(tmp), out, sr)
    return tmp


def _covers(regions: list[dict], duration: float) -> bool:
    if not regions:
        return duration == 0
    if abs(regions[0]["start"]) > 1e-6:
        return False
    for a, b in zip(regions, regions[1:]):
        if abs(a["end"] - b["start"]) > 1e-6:
            return False
    return abs(regions[-1]["end"] - duration) < 0.05


def _kind_at(regions: list[dict], t: float) -> str:
    for r in regions:
        if r["start"] <= t < r["end"]:
            return r["kind"]
    return regions[-1]["kind"]


def test_detect_pure_silence_is_all_instrumental():
    # 3 s of silence (below noise floor).
    path = _synthesize([(0.0, 3.0, 0.0)])
    regions = vad.detect(path)
    assert _covers(regions, 3.0)
    assert all(r["kind"] == "instrumental" for r in regions)


def test_detect_loud_singing_is_all_vocals():
    # 3 s of -12 dBFS tone — unambiguously vocals.
    path = _synthesize([(0.0, 3.0, 0.25)])
    regions = vad.detect(path)
    assert _covers(regions, 3.0)
    assert all(r["kind"] == "vocals" for r in regions)


def test_detect_vocal_instrumental_vocal():
    # Vocals 0-4s, instrumental 4-8s (>1.5s so not merged), vocals 8-12s.
    path = _synthesize([
        (0.0, 4.0, 0.20),
        (4.0, 8.0, 0.0),
        (8.0, 12.0, 0.20),
    ])
    regions = vad.detect(path)
    assert _covers(regions, 12.0)
    # Expect at least 3 regions with the pattern vocals/instrumental/vocals.
    kinds = [r["kind"] for r in regions]
    assert "instrumental" in kinds
    # Pick a midpoint of each era and check the classification.
    assert _kind_at(regions, 2.0) == "vocals"
    assert _kind_at(regions, 6.0) == "instrumental"
    assert _kind_at(regions, 10.0) == "vocals"


def test_short_instrumental_gap_merged():
    # 0.3s instrumental between vocals — shorter than _MIN_INSTRUMENTAL_SEC,
    # should be absorbed into surrounding vocals.
    path = _synthesize([
        (0.0, 3.0, 0.20),
        (3.0, 3.3, 0.0),
        (3.3, 6.0, 0.20),
    ])
    regions = vad.detect(path)
    assert _kind_at(regions, 3.15) == "vocals"

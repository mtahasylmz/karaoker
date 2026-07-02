"""Synthetic-audio tests for the RMS-VAD (harvested from stages/transcribe)."""

from __future__ import annotations

import numpy as np

from annemusic import vad

SR = 16000


def _synth(pattern: list[tuple[float, float, float]]) -> np.ndarray:
    """Render mono audio from (start, end, rms) triples; silence elsewhere."""
    total = max(end for _, end, _ in pattern)
    n = int(total * SR)
    t = np.arange(n) / SR
    out = np.zeros(n, dtype=np.float32)
    for start, end, rms in pattern:
        if rms <= 0:
            continue
        i0, i1 = int(start * SR), int(end * SR)
        amp = float(rms) * np.sqrt(2.0)
        out[i0:i1] += amp * np.sin(2 * np.pi * 440 * t[i0:i1]).astype(np.float32)
    return out


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


def test_pure_silence_is_all_instrumental():
    regions = vad.detect_regions(_synth([(0.0, 3.0, 0.0)]), SR)
    assert _covers(regions, 3.0)
    assert all(r["kind"] == "instrumental" for r in regions)


def test_loud_singing_is_all_vocals():
    regions = vad.detect_regions(_synth([(0.0, 3.0, 0.25)]), SR)
    assert _covers(regions, 3.0)
    assert all(r["kind"] == "vocals" for r in regions)


def test_vocal_instrumental_vocal():
    regions = vad.detect_regions(
        _synth([(0.0, 4.0, 0.20), (4.0, 8.0, 0.0), (8.0, 12.0, 0.20)]), SR
    )
    assert _covers(regions, 12.0)
    assert _kind_at(regions, 2.0) == "vocals"
    assert _kind_at(regions, 6.0) == "instrumental"
    assert _kind_at(regions, 10.0) == "vocals"


def test_short_instrumental_gap_merged():
    regions = vad.detect_regions(
        _synth([(0.0, 3.0, 0.20), (3.0, 3.3, 0.0), (3.3, 6.0, 0.20)]), SR
    )
    assert _kind_at(regions, 3.15) == "vocals"


def test_empty_audio_returns_no_regions():
    assert vad.detect_regions(np.zeros(0, dtype=np.float32), SR) == []


def test_stereo_input_is_downmixed(tmp_path):
    import soundfile as sf

    stereo = np.stack([_synth([(0.0, 3.0, 0.25)])] * 2, axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo, SR)
    regions = vad.detect(path)
    assert all(r["kind"] == "vocals" for r in regions)

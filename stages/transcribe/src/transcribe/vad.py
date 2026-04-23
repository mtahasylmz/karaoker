"""RMS-based vocal activity detection on the Demucs vocals stem.

Demucs leaves a very low noise floor in instrumental-only regions (typically
-50 to -40 dBFS), while active singing sits at -20 to -10 dBFS. A simple
short-time RMS envelope + hysteresis threshold separates the two cleanly —
far better than running Silero VAD (trained on speech) on singing.

The 2506.15514 ALT paper ("Exploiting Music Source Separation for ALT with
Whisper") found that using RMS on the vocal stem as a vocal-activity detector
beats Whisper's native 30s sliding window for long-form segmentation — this
module is the same trick.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

# Tunables. Defaults picked against Demucs output; can be overridden per-call.
_FRAME_SEC = 0.04           # 40 ms RMS frame
_HOP_SEC = 0.02             # 20 ms hop
_SMOOTH_SEC = 0.25          # 250 ms moving-average smoothing
_ON_THRESHOLD = 0.010       # RMS ≈ -40 dBFS — enter vocals above this
_OFF_THRESHOLD = 0.005      # RMS ≈ -46 dBFS — leave vocals below this (hysteresis)
_MIN_INSTRUMENTAL_SEC = 1.5  # merge shorter "instrumental" gaps back into vocals
_MIN_VOCALS_SEC = 0.3       # drop vocals blips shorter than this (artifacts)


def detect(audio_path: Path) -> list[dict]:
    """Return ordered, non-overlapping regions covering [0, duration].

    Each region: {"start": float, "end": float, "kind": "vocals"|"instrumental"}.
    Contract-compatible with `VocalActivityRegion`.
    """
    data, sr = sf.read(str(audio_path), always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32, copy=False)

    duration = len(data) / float(sr)
    if duration <= 0:
        return []

    envelope, times = _rms_envelope(data, sr)
    envelope = _smooth(envelope, sr)
    mask = _hysteresis_mask(envelope)
    regions = _mask_to_regions(mask, times, duration)
    regions = _dilate_merge(regions)
    return regions


def _rms_envelope(data: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    frame = max(1, int(_FRAME_SEC * sr))
    hop = max(1, int(_HOP_SEC * sr))
    # Pad so final frame fits.
    pad = (frame - (len(data) - frame) % hop) % hop
    if pad:
        data = np.pad(data, (0, pad), mode="constant")
    n_frames = 1 + (len(data) - frame) // hop
    # Strided view avoids allocating a big (n_frames, frame) matrix.
    shape = (n_frames, frame)
    strides = (data.strides[0] * hop, data.strides[0])
    frames = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    times = np.arange(n_frames) * (hop / sr) + (frame / (2 * sr))
    return rms, times


def _smooth(envelope: np.ndarray, sr: int) -> np.ndarray:
    win_frames = max(1, int(_SMOOTH_SEC / _HOP_SEC))
    if win_frames <= 1:
        return envelope
    kernel = np.ones(win_frames, dtype=np.float32) / win_frames
    return np.convolve(envelope, kernel, mode="same")


def _hysteresis_mask(envelope: np.ndarray) -> np.ndarray:
    """True = vocals active. Hysteresis avoids chatter around the threshold."""
    mask = np.zeros(envelope.shape, dtype=bool)
    active = False
    for i, v in enumerate(envelope):
        if active:
            if v < _OFF_THRESHOLD:
                active = False
        else:
            if v > _ON_THRESHOLD:
                active = True
        mask[i] = active
    return mask


def _mask_to_regions(
    mask: np.ndarray, times: np.ndarray, duration: float
) -> list[dict]:
    if len(mask) == 0:
        return [{"start": 0.0, "end": duration, "kind": "instrumental"}]
    regions: list[dict] = []
    # Treat each frame as covering [times[i] - hop/2, times[i] + hop/2]. Walk
    # the mask, emitting a region every time the value flips.
    half = _HOP_SEC / 2
    cur_kind = "vocals" if mask[0] else "instrumental"
    cur_start = 0.0
    for i in range(1, len(mask)):
        kind = "vocals" if mask[i] else "instrumental"
        if kind != cur_kind:
            boundary = max(0.0, times[i] - half)
            regions.append({"start": cur_start, "end": boundary, "kind": cur_kind})
            cur_start = boundary
            cur_kind = kind
    regions.append({"start": cur_start, "end": duration, "kind": cur_kind})
    return regions


def _dilate_merge(regions: list[dict]) -> list[dict]:
    """Drop sub-threshold-duration regions and coalesce touching same-kind ones."""
    if not regions:
        return regions

    # 1. Drop tiny vocals blips: merge into neighbor.
    cleaned: list[dict] = []
    for r in regions:
        dur = r["end"] - r["start"]
        if r["kind"] == "vocals" and dur < _MIN_VOCALS_SEC:
            # Flip kind; coalesced in pass 2.
            cleaned.append({**r, "kind": "instrumental"})
        else:
            cleaned.append(dict(r))

    # 2. Merge short instrumental gaps back into surrounding vocals. Only merge
    #    when the instrumental region is flanked by vocals on both sides — a
    #    short instrumental at the head/tail of the song stays instrumental.
    merged: list[dict] = []
    i = 0
    while i < len(cleaned):
        r = cleaned[i]
        if (
            r["kind"] == "instrumental"
            and (r["end"] - r["start"]) < _MIN_INSTRUMENTAL_SEC
            and merged
            and merged[-1]["kind"] == "vocals"
            and i + 1 < len(cleaned)
            and cleaned[i + 1]["kind"] == "vocals"
        ):
            merged[-1]["end"] = cleaned[i + 1]["end"]
            i += 2
            continue
        merged.append(r)
        i += 1

    # 3. Coalesce consecutive same-kind regions (can happen after step 1).
    coalesced: list[dict] = []
    for r in merged:
        if coalesced and coalesced[-1]["kind"] == r["kind"]:
            coalesced[-1]["end"] = r["end"]
        else:
            coalesced.append(r)
    return coalesced

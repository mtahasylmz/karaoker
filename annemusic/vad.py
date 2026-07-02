"""RMS-based vocal activity detection on the separated vocals stem.

Separation leaves a very low noise floor in instrumental-only regions
(typically -50 to -40 dBFS) while active singing sits at -20 to -10 dBFS, so
a short-time RMS envelope + hysteresis threshold separates the two cleanly —
far better than a speech-trained VAD on singing (same trick as the
2506.15514 ALT paper). Thresholds are tuned for stem output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_FRAME_SEC = 0.04            # 40 ms RMS frame
_HOP_SEC = 0.02              # 20 ms hop
_SMOOTH_SEC = 0.25           # 250 ms moving-average smoothing
_ON_THRESHOLD = 0.010        # RMS ~ -40 dBFS — enter vocals above this
_OFF_THRESHOLD = 0.005       # RMS ~ -46 dBFS — leave vocals below (hysteresis)
_MIN_INSTRUMENTAL_SEC = 1.5  # merge shorter instrumental gaps back into vocals
_MIN_VOCALS_SEC = 0.3        # drop vocals blips shorter than this (artifacts)


def detect(audio_path: Path) -> list[dict]:
    """Read a stem file and return its vocal-activity regions."""
    import soundfile as sf  # lazy: keeps `import annemusic.vad` numpy-only

    data, sr = sf.read(str(audio_path), always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return detect_regions(data.astype(np.float32, copy=False), sr)


def detect_regions(data: np.ndarray, sr: int) -> list[dict]:
    """Pure DSP core: ordered, non-overlapping regions covering [0, duration].

    Each region: {"start": float, "end": float, "kind": "vocals"|"instrumental"}.
    """
    duration = len(data) / float(sr)
    if duration <= 0:
        return []
    envelope, times = _rms_envelope(data, sr)
    envelope = _smooth(envelope)
    mask = _hysteresis_mask(envelope)
    regions = _mask_to_regions(mask, times, duration)
    return _dilate_merge(regions)


def _rms_envelope(data: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    frame = max(1, int(_FRAME_SEC * sr))
    hop = max(1, int(_HOP_SEC * sr))
    pad = (frame - (len(data) - frame) % hop) % hop
    if pad:
        data = np.pad(data, (0, pad), mode="constant")
    n_frames = 1 + (len(data) - frame) // hop
    shape = (n_frames, frame)
    strides = (data.strides[0] * hop, data.strides[0])
    frames = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    times = np.arange(n_frames) * (hop / sr) + (frame / (2 * sr))
    return rms, times


def _smooth(envelope: np.ndarray) -> np.ndarray:
    win_frames = max(1, int(_SMOOTH_SEC / _HOP_SEC))
    if win_frames <= 1 or envelope.size == 0:
        # np.convolve rejects empty input (audio shorter than one RMS frame);
        # an empty envelope flows through to the all-instrumental fallback.
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
    """Drop sub-threshold-duration regions and coalesce same-kind neighbors."""
    if not regions:
        return regions
    return _coalesce(_bridge_short_gaps(_flip_short_vocals(regions)))


def _flip_short_vocals(regions: list[dict]) -> list[dict]:
    """Tiny vocals blips flip to instrumental; coalesced by _coalesce."""
    return [
        {**r, "kind": "instrumental"}
        if r["kind"] == "vocals" and r["end"] - r["start"] < _MIN_VOCALS_SEC
        else dict(r)
        for r in regions
    ]


def _bridgeable(regions: list[dict], merged: list[dict], i: int) -> bool:
    """Short instrumental gap flanked by vocals on both sides — a short
    instrumental at the head/tail of the song stays instrumental."""
    r = regions[i]
    return (
        r["kind"] == "instrumental"
        and (r["end"] - r["start"]) < _MIN_INSTRUMENTAL_SEC
        and bool(merged)
        and merged[-1]["kind"] == "vocals"
        and i + 1 < len(regions)
        and regions[i + 1]["kind"] == "vocals"
    )


def _bridge_short_gaps(regions: list[dict]) -> list[dict]:
    """Merge bridgeable instrumental gaps into the surrounding vocals."""
    merged: list[dict] = []
    i = 0
    while i < len(regions):
        if _bridgeable(regions, merged, i):
            merged[-1]["end"] = regions[i + 1]["end"]
            i += 2
            continue
        merged.append(regions[i])
        i += 1
    return merged


def _coalesce(regions: list[dict]) -> list[dict]:
    """Coalesce consecutive same-kind regions."""
    out: list[dict] = []
    for r in regions:
        if out and out[-1]["kind"] == r["kind"]:
            out[-1]["end"] = r["end"]
        else:
            out.append(r)
    return out

"""Generate `plate_ir.wav` — synthetic plate-style impulse response.

Exponentially-decaying stereo noise band-limited into a plate-ish resonance
region (500 Hz – 8 kHz). Fully deterministic from a fixed RNG seed so the
committed WAV is reproducible.

Licence: we authored the IR programmatically → CC0 (public domain dedication).
No external-attribution strings to carry.

Run:
    uv run --package annemusic-stage-record-mix python assets/make_plate_ir.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


OUT = Path(__file__).resolve().parent / "plate_ir.wav"
FS = 48_000
DUR_S = 1.8            # plate tail length — long enough to sound lush, short enough to stay small
RT60_S = 1.2           # target decay time
LOW_HZ, HIGH_HZ = 500, 8000


def main() -> None:
    rng = np.random.default_rng(20260424)
    n = int(FS * DUR_S)
    noise = rng.standard_normal((n, 2)).astype(np.float32)

    # Band-pass into the "plate" range so it doesn't sound like broadband hiss.
    sos = butter(4, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=FS, output="sos")
    shaped = sosfilt(sos, noise, axis=0).astype(np.float32)

    # Exponential decay reaching -60 dB (1/1000) at RT60.
    decay = np.exp(-np.linspace(0, n / FS, n) * (np.log(1000.0) / RT60_S)).astype(np.float32)
    ir = shaped * decay[:, None]

    # Short pre-delay + early-reflection sparkle so the transient doesn't feel
    # like plain noise. Five taps spaced around 7–40 ms.
    taps_ms = [7, 13, 19, 29, 41]
    taps = np.zeros_like(ir)
    for i, ms in enumerate(taps_ms):
        k = int(FS * ms / 1000)
        gain = 0.5 * (0.7 ** i)
        taps[k, 0] += gain
        taps[k, 1] += gain * (-1.0 if i % 2 else 1.0)
    ir = ir * 0.85 + taps

    # Peak-normalise to -1 dBFS so the IR has predictable level.
    peak = float(np.max(np.abs(ir)))
    if peak > 0:
        ir = ir / peak * 10 ** (-1.0 / 20.0)

    sf.write(str(OUT), ir.astype(np.float32), FS, subtype="PCM_16")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB, {n} samples @ {FS} Hz, 2ch)")


if __name__ == "__main__":
    main()

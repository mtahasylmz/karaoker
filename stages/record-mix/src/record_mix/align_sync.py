"""GCC-PHAT time alignment.

Whitens the cross-spectrum so the correlation peak is a delta regardless of
spectral coloration — canonical for reverberant rooms and the default sync
algorithm in karaoke pipelines.

Pure numpy/scipy — no ffmpeg here, so unit tests stay hermetic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_FS = 48_000
DEFAULT_WINDOW_S = 10.0
DEFAULT_MAX_LAG_MS = 800.0
SNR_ACCEPT_DB = 3.0


def load_mono_48k(path: str | Path, *, max_seconds: float | None = DEFAULT_WINDOW_S) -> np.ndarray:
    """Read a WAV, downmix to mono, resample to 48 kHz, clip to max_seconds."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != TARGET_FS:
        # resample_poly needs integer up/down ratios — compute from GCD.
        g = np.gcd(sr, TARGET_FS)
        mono = resample_poly(mono, TARGET_FS // g, sr // g).astype(np.float32)
    if max_seconds is not None:
        n = int(TARGET_FS * max_seconds)
        mono = mono[:n]
    return mono


def gcc_phat(
    sig: np.ndarray,
    ref: np.ndarray,
    fs: int = TARGET_FS,
    *,
    max_lag_ms: float = DEFAULT_MAX_LAG_MS,
) -> tuple[float, float]:
    """Return (lag_seconds, snr_db).

    Positive lag_seconds → `sig` arrived late relative to `ref` (trim sig's start).
    Negative lag_seconds → `sig` is early (delay sig with adelay).
    snr_db = 20 * log10(peak / median(|cc|)); callers gate on >= SNR_ACCEPT_DB.
    """
    sig = np.asarray(sig, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    n = len(sig) + len(ref)
    nfft = 1 << (int(n - 1).bit_length())
    X = np.fft.rfft(sig, nfft)
    Y = np.fft.rfft(ref, nfft)
    # Whiten: divide by magnitude so only the phase carries through. The eps
    # prevents division-by-zero on silent bins.
    cross = X * np.conj(Y)
    cross /= np.abs(cross) + 1e-15
    cc = np.fft.irfft(cross, nfft)

    max_lag = int(round(fs * max_lag_ms / 1000.0))
    max_lag = min(max_lag, nfft // 2 - 1)
    # Arrange so index 0 == lag 0, positive indices == positive lags.
    window = np.concatenate([cc[: max_lag + 1], cc[-max_lag:]])
    lags = np.concatenate([np.arange(max_lag + 1), np.arange(-max_lag, 0)])
    i = int(np.argmax(np.abs(window)))
    peak = float(abs(window[i]))
    floor = float(np.median(np.abs(window))) + 1e-15
    snr_db = 20.0 * np.log10(peak / floor)
    lag_s = float(lags[i]) / fs
    return lag_s, snr_db


__all__ = [
    "TARGET_FS",
    "SNR_ACCEPT_DB",
    "DEFAULT_MAX_LAG_MS",
    "DEFAULT_WINDOW_S",
    "load_mono_48k",
    "gcc_phat",
]

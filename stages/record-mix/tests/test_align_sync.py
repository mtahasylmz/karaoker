"""GCC-PHAT: recover a synthetic delay within ±5 ms, SNR comfortably high."""

from __future__ import annotations

import numpy as np
import pytest

from record_mix import align_sync


FS = align_sync.TARGET_FS


def _mixed_signal(duration_s: float = 4.0, seed: int = 1) -> np.ndarray:
    """Broadband-ish signal with the spectral variety GCC-PHAT needs to whiten."""
    rng = np.random.default_rng(seed)
    n = int(FS * duration_s)
    t = np.arange(n, dtype=np.float32) / FS
    voice = 0.4 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    voice += 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    noise = rng.standard_normal(n).astype(np.float32) * 0.1
    return voice + noise


@pytest.mark.parametrize("delay_ms", [0, 35, 120, -80, 300])
def test_recovers_offset_within_5ms(delay_ms):
    ref = _mixed_signal()
    delay_samples = int(round(abs(delay_ms) * FS / 1000.0))
    if delay_ms >= 0:
        sig = np.concatenate([np.zeros(delay_samples, dtype=np.float32), ref])[: len(ref)]
    else:
        sig = np.concatenate([ref[delay_samples:], np.zeros(delay_samples, dtype=np.float32)])
    lag_s, snr_db = align_sync.gcc_phat(sig, ref)
    recovered_ms = lag_s * 1000.0
    assert abs(recovered_ms - delay_ms) < 5.0, (
        f"recovered {recovered_ms:.2f} ms for true {delay_ms} ms"
    )
    assert snr_db >= 10.0, f"expected high SNR, got {snr_db:.1f} dB"


def test_correlated_snr_dominates_uncorrelated():
    """PHAT whitens magnitude, so even random inputs yield some peak — but the
    correlated SNR must sit a long way above the uncorrelated SNR. Production
    gating on SNR_ACCEPT_DB is fine because real recordings are well into the
    correlated regime (typically 20–30 dB over floor)."""
    rng = np.random.default_rng(7)
    shared = _mixed_signal(seed=7)
    _, snr_correlated = align_sync.gcc_phat(shared, shared)
    a = rng.standard_normal(FS * 4).astype(np.float32)
    b = rng.standard_normal(FS * 4).astype(np.float32)
    _, snr_uncorrelated = align_sync.gcc_phat(a, b)
    assert snr_correlated > snr_uncorrelated + 10.0, (
        f"correlated SNR {snr_correlated:.1f} dB should dominate "
        f"uncorrelated {snr_uncorrelated:.1f} dB"
    )

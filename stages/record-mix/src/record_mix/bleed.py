"""Demucs bleed cleanup on the user recording.

Mirrors `stages/separate/_separate_demucs` — subprocess `python -m demucs`
with htdemucs (MIT weights from Meta, guaranteed-clean license). Returns
the isolated vocal stem for the downstream mixer. The non-vocal stem is
discarded.

Model weights auto-cache to `~/.cache/torch/hub` on first run (~320 MB).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


DEFAULT_MODEL = "htdemucs"


def clean_bleed(
    recording_wav: Path,
    out_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    runner=None,
) -> Path:
    """Run Demucs on `recording_wav`, return the vocals stem path.

    `runner` is a callable(list[str]) used for subprocess dispatch; defaults to
    this module's local runner so tests can inject a stub if needed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "demucs",
        "--two-stems=vocals",
        "-n",
        model,
        "-o",
        str(out_dir),
        str(recording_wav),
    ]
    (runner or _default_runner)(cmd)

    model_subdir = out_dir / model
    if not model_subdir.exists():
        raise RuntimeError(f"demucs output missing: expected {model_subdir}")
    stem_dir = next(model_subdir.iterdir())
    vocals = stem_dir / "vocals.wav"
    if not vocals.exists():
        raise RuntimeError(
            f"demucs vocals stem missing: {sorted(p.name for p in stem_dir.iterdir())}"
        )
    return vocals


def _default_runner(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"command failed ({r.returncode}): {' '.join(cmd[:3])}\n"
            f"stderr: {r.stderr[-2000:]}"
        )


__all__ = ["DEFAULT_MODEL", "clean_bleed"]

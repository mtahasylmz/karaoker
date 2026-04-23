"""Separate vocals from instrumental via Demucs.

Ported from MVP worker/pipeline.py:_separate with env-driven model selection.
Future SOTA models (BS-RoFormer, Mel-RoFormer) plug in by adding branches
here; the HTTP contract and env var `SEPARATE_MODEL` stay.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

from shared import create_logger, download_file, upload_file, object_path_from_gs_uri

log = create_logger("separate")


def run(job_id: str, source_uri: str, model: str | None = None) -> dict:
    """Download source, separate, upload vocals+instrumental. Returns dict
    matching SeparateResponse contract (less stage/job_id/timing wrappers)."""
    started = int(time.time() * 1000)
    active_model = (model or os.environ.get("SEPARATE_MODEL") or "htdemucs").strip()
    log.info(job_id, "starting", {"model": active_model, "source": source_uri})

    with tempfile.TemporaryDirectory(prefix=f"separate-{job_id}-") as tmp_s:
        tmp = Path(tmp_s)

        # 1) pull source file (mp4/mov/webm/mkv) to tmp
        source_obj = object_path_from_gs_uri(source_uri)
        ext = Path(source_obj).suffix or ".mp4"
        local_source = tmp / f"source{ext}"
        log.debug(job_id, "downloading source", {"object": source_obj})
        download_file(source_obj, local_source)

        # 2) ffmpeg → stereo 44.1k wav for demucs
        audio = tmp / "audio.wav"
        log.debug(job_id, "extracting audio", {})
        _run([
            "ffmpeg", "-y", "-i", str(local_source),
            "-vn", "-ar", "44100", "-ac", "2",
            str(audio),
        ])

        # 3) demucs two-stems separation
        out_dir = tmp / "demucs"
        log.info(job_id, "demucs running", {"model": active_model})
        _run([
            "python", "-m", "demucs",
            "--two-stems=vocals",
            "-n", active_model,
            "-o", str(out_dir),
            str(audio),
        ])

        model_subdir = out_dir / active_model
        stem_dir = next(model_subdir.iterdir())  # demucs uses input filename as subdir
        vocals_src = stem_dir / "vocals.wav"
        instrumental_src = stem_dir / "no_vocals.wav"
        if not vocals_src.exists() or not instrumental_src.exists():
            raise RuntimeError(f"demucs output missing: {list(stem_dir.iterdir())}")

        # 4) upload outputs under a stable stage-owned path
        vocals_obj = f"stages/separate/{job_id}/vocals.wav"
        instr_obj = f"stages/separate/{job_id}/no_vocals.wav"
        upload_file(vocals_obj, vocals_src, content_type="audio/wav")
        upload_file(instr_obj, instrumental_src, content_type="audio/wav")

    finished = int(time.time() * 1000)
    bucket = os.environ.get("GCS_BUCKET", "")
    result = {
        "job_id": job_id,
        "stage": "separate",
        "started_at": started,
        "finished_at": finished,
        "duration_ms": finished - started,
        "vocals_uri": f"gs://{bucket}/{vocals_obj}",
        "instrumental_uri": f"gs://{bucket}/{instr_obj}",
        "sample_rate": 44100,
        "model_used": active_model,
    }
    log.info(job_id, "done", {"duration_ms": result["duration_ms"], "model": active_model})
    return result


def _run(cmd: list[str]) -> None:
    log.debug(None, "exec", {"cmd": " ".join(cmd[:3]) + "..."})
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd[:3])}\n"
            f"stderr: {result.stderr[-2000:]}"
        )

"""Transcribe vocals. Fast path: LRCLIB synced lyrics. Slow path: Whisper.

Deliberately skips whisperx.load_model (its VAD model URL is a dead S3
bucket — that's the MVP scar). Uses faster-whisper directly with its
built-in Silero VAD. Segment-level output; word-level timings are produced
by stages/align.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from shared import (
    create_logger,
    download_file,
    object_path_from_gs_uri,
)

from . import lrclib

log = create_logger("transcribe")

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    from faster_whisper import WhisperModel  # lazy; heavy import
    size = os.environ.get("WHISPER_MODEL", "small")
    compute = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
    log.info(None, "loading whisper", {"size": size, "compute": compute})
    _model = WhisperModel(size, device="cpu", compute_type=compute)
    return _model


def run(
    job_id: str,
    vocals_uri: str,
    language: str | None = None,
    known_lyrics: str | None = None,
    title: str | None = None,
    artist: str | None = None,
) -> dict:
    started = int(time.time() * 1000)
    log.info(job_id, "starting", {"has_lyrics": bool(known_lyrics), "title": title, "artist": artist})

    # 1) Try LRCLIB if we have metadata. If matched → skip Whisper.
    if title and artist:
        segs = lrclib.fetch(title, artist)
        if segs:
            finished = int(time.time() * 1000)
            log.info(job_id, "lrclib hit", {"segments": len(segs), "title": title})
            return _response(
                job_id, started, finished,
                language=language or "und",
                segments=segs,
                source="lrclib",
                model_used="lrclib",
            )
        log.info(job_id, "lrclib miss", {"title": title, "artist": artist})

    # 2) Download the vocals stem to a tmp path.
    with tempfile.TemporaryDirectory(prefix=f"transcribe-{job_id}-") as tmp_s:
        tmp = Path(tmp_s)
        vocals_obj = object_path_from_gs_uri(vocals_uri)
        local_vocals = tmp / "vocals.wav"
        log.debug(job_id, "downloading vocals", {"object": vocals_obj})
        download_file(vocals_obj, local_vocals)

        # 3) Whisper with built-in Silero VAD. Segment-level only; align stage
        #    produces word timings.
        model = _load_model()
        log.info(job_id, "whisper transcribing", {})
        segments_iter, info = model.transcribe(
            str(local_vocals),
            language=language,
            vad_filter=True,
            beam_size=5,
            word_timestamps=False,
            initial_prompt=(known_lyrics[:200] if known_lyrics else None),
        )
        segments: list[dict] = []
        for seg in segments_iter:
            text = (seg.text or "").strip()
            if not text or seg.end <= seg.start:
                continue
            segments.append({"text": text, "start": float(seg.start), "end": float(seg.end)})

    finished = int(time.time() * 1000)
    log.info(
        job_id,
        "whisper complete",
        {"language": info.language, "prob": round(float(info.language_probability), 3), "segments": len(segments)},
    )
    return _response(
        job_id, started, finished,
        language=info.language,
        segments=segments,
        source="whisper",
        model_used=os.environ.get("WHISPER_MODEL", "small"),
    )


def _response(
    job_id: str,
    started: int,
    finished: int,
    *,
    language: str,
    segments: list[dict],
    source: str,
    model_used: str,
) -> dict:
    return {
        "job_id": job_id,
        "stage": "transcribe",
        "started_at": started,
        "finished_at": finished,
        "duration_ms": finished - started,
        "language": language,
        "segments": segments,
        "source": source,
        "model_used": model_used,
    }

"""Transcribe the song. Backend is flow-routed: Qwen3-ASR on the full mix
(for languages in its supported set) or faster-whisper on the vocals stem.

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
    flow_for,
    input_for_backend,
    object_path_from_gs_uri,
)

from . import vad

log = create_logger("transcribe")

# Flip to True once the Qwen3-ASR backend is wired up. Until then the flow
# still resolves to "qwen3" for supported languages but we fall back to
# whisper at dispatch time — cleanly, with a log line — rather than silently
# ignoring the flow.
_QWEN3_AVAILABLE = False

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
    source_uri: str | None = None,
    language: str | None = None,
    known_lyrics: str | None = None,
) -> dict:
    started = int(time.time() * 1000)
    log.info(job_id, "starting", {"has_lyrics": bool(known_lyrics), "language": language})

    # Resolve the flow and the *concrete* backend we're about to run.
    # flow.transcribe is advisory (what the language prefers); backend is
    # what actually executes after accounting for availability. The audio
    # URI is picked off the concrete backend, not the flow — so if Qwen3
    # isn't wired yet we still feed whisper its native input (vocals).
    flow = flow_for(language)
    backend = flow.transcribe
    if backend == "qwen3" and not _QWEN3_AVAILABLE:
        log.warn(
            job_id,
            "qwen3 backend unavailable; falling back to whisper",
            {"language": language, "flow_input": flow.transcribe_input},
        )
        backend = "whisper"

    audio_input = input_for_backend(backend)  # qwen3 → "mix", whisper → "vocals"
    audio_uri = source_uri if audio_input == "mix" else vocals_uri
    if audio_input == "mix" and not source_uri:
        # Orchestrator didn't pass the original upload (e.g. a stale caller).
        # Transcribing the vocals stem is still correct, just not ideal.
        log.warn(job_id, "flow wanted mix but source_uri missing; using vocals")
        audio_uri = vocals_uri
        audio_input = "vocals"

    log.info(
        job_id,
        "flow resolved",
        {
            "language_hint": language,
            "backend": backend,
            "audio_input": audio_input,
            "flow_transcribe": flow.transcribe,
            "flow_align": flow.align,
        },
    )

    # Download the chosen audio and dispatch.
    with tempfile.TemporaryDirectory(prefix=f"transcribe-{job_id}-") as tmp_s:
        tmp = Path(tmp_s)
        audio_obj = object_path_from_gs_uri(audio_uri)
        local_audio = tmp / ("mix.bin" if audio_input == "mix" else "vocals.wav")
        log.debug(job_id, "downloading audio", {"object": audio_obj, "input": audio_input})
        download_file(audio_obj, local_audio)
        # RMS-VAD always runs on the vocals stem (absence of energy on the
        # isolated stem is ground truth for instrumental breaks). Download
        # separately if the ASR input was the mix.
        if audio_input == "mix":
            vocals_obj = object_path_from_gs_uri(vocals_uri)
            local_vocals = tmp / "vocals.wav"
            download_file(vocals_obj, local_vocals)
        else:
            local_vocals = local_audio

        if backend == "qwen3":
            # Unreachable today (_QWEN3_AVAILABLE is False); kept here so the
            # dispatch point is visible when the backend lands.
            raise NotImplementedError("qwen3 backend not wired yet")

        # whisper path
        model = _load_model()
        log.info(job_id, "whisper transcribing", {"input": audio_input})
        segments_iter, info = model.transcribe(
            str(local_audio),
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

        # RMS-VAD on the vocals stem regardless of what the ASR model ate.
        vocal_activity = vad.detect(local_vocals)

    finished = int(time.time() * 1000)
    log.info(
        job_id,
        "whisper complete",
        {
            "language": info.language,
            "prob": round(float(info.language_probability), 3),
            "segments": len(segments),
            "vocal_regions": len(vocal_activity),
            "audio_input": audio_input,
        },
    )
    return _response(
        job_id, started, finished,
        language=info.language,
        segments=segments,
        vocal_activity=vocal_activity,
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
    vocal_activity: list[dict],
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
        "vocal_activity": vocal_activity,
        "source": source,
        "model_used": model_used,
    }

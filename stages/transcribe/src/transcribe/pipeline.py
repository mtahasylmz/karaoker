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

import soundfile as sf

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
_QWEN3_AVAILABLE = True

_model = None

# Qwen3-ASR's `language` kwarg wants English names, not ISO codes. This map
# covers _QWEN_TRANSCRIBE_LANGS from shared/flows.py (the 30 languages where
# we route to qwen3). If `language` is None we pass None (auto-detect).
_ISO_TO_QWEN: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}
_QWEN_TO_ISO: dict[str, str] = {v: k for k, v in _ISO_TO_QWEN.items()}

_qwen3_model = None
_qwen3_device: str | None = None  # tracks actual loaded device for retry logic


def _now_ms() -> int:
    return int(time.time() * 1000)


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


def _pick_device() -> str:
    """cuda > mps > cpu. TRANSCRIBE_DEVICE env overrides verbatim.

    Wrapped in try/except so a missing/broken torch install doesn't crash
    the import — we just land on cpu.
    """
    override = os.environ.get("TRANSCRIBE_DEVICE")
    if override:
        return override
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load_qwen3(force_device: str | None = None):
    """Lazy global singleton. Heavy imports (torch, qwen_asr) stay here so
    they don't pay the cost when the flow resolves to whisper.
    """
    global _qwen3_model, _qwen3_device
    if _qwen3_model is not None and force_device is None:
        return _qwen3_model
    import torch  # lazy
    from qwen_asr import Qwen3ASRModel  # lazy

    repo = os.environ.get("QWEN3_MODEL", "Qwen/Qwen3-ASR-1.7B")
    device = force_device or _pick_device()
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    max_new_tokens = int(os.environ.get("QWEN3_MAX_NEW_TOKENS", "512"))
    log.info(
        None,
        "loading qwen3",
        {"repo": repo, "device": device, "dtype": str(dtype), "max_new_tokens": max_new_tokens},
    )
    _qwen3_model = Qwen3ASRModel.from_pretrained(
        repo,
        dtype=dtype,
        device_map=device,
        max_new_tokens=max_new_tokens,
    )
    _qwen3_device = device
    return _qwen3_model


def _audio_duration_seconds(path: Path) -> float:
    """Duration in seconds for any format librosa can read.

    Soundfile alone can't handle mp4/m4a; librosa falls back to audioread for
    container formats. Used in the qwen3 path where the input is the caller's
    original upload (video/audio, any container).
    """
    try:
        # Fast path for wav/flac/ogg etc — soundfile only reads the header.
        return float(sf.info(str(path)).duration)
    except Exception:
        import librosa  # lazy
        return float(librosa.get_duration(path=str(path)))


def _force_qwen3_cpu_reload() -> None:
    """Drop the current (mps-resident) model and reload on cpu. One-shot retry
    path for MPS op-gap RuntimeError. Called with _qwen3_model already loaded.
    """
    global _qwen3_model, _qwen3_device
    _qwen3_model = None
    _qwen3_device = None
    _load_qwen3(force_device="cpu")


def run(
    job_id: str,
    vocals_uri: str,
    source_uri: str | None = None,
    language: str | None = None,
    known_lyrics: str | None = None,
) -> dict:
    started = _now_ms()
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
            # known_lyrics biasing is NOT supported by the local qwen-asr
            # package; context biasing exists only in the DashScope cloud
            # API. We log once per request when a caller passed it.
            if known_lyrics:
                log.info(
                    job_id,
                    "qwen3 ignoring known_lyrics (not supported by local qwen-asr package)",
                    {},
                )
            model = _load_qwen3()
            qwen_lang = _ISO_TO_QWEN.get(language) if language else None
            log.info(
                job_id,
                "qwen3 transcribing",
                {"device": _qwen3_device, "language": qwen_lang, "input": audio_input},
            )
            try:
                results = model.transcribe(
                    audio=str(local_audio),
                    language=qwen_lang,
                    return_time_stamps=False,
                )
            except RuntimeError as e:
                # MPS commonly hits op gaps at runtime. One-shot CPU retry.
                if _qwen3_device == "mps":
                    log.warn(
                        job_id,
                        "qwen3 mps failed, reloading on cpu",
                        {"err": str(e)},
                    )
                    _force_qwen3_cpu_reload()
                    results = _qwen3_model.transcribe(
                        audio=str(local_audio),
                        language=qwen_lang,
                        return_time_stamps=False,
                    )
                else:
                    raise

            r = results[0]
            detected_iso = _QWEN_TO_ISO.get(r.language, language or "und")
            # soundfile can't read mp4/m4a; Qwen3-ASR itself uses librosa with
            # audioread fallback for arbitrary container formats, and the mix
            # URI here is the caller's original upload (mp4/mov/etc). librosa
            # is already installed as a qwen-asr dep.
            duration = float(_audio_duration_seconds(local_audio))
            text = (r.text or "").strip()
            # Single coarse segment spanning the audio. Word-level timing is
            # stages/align's job; Qwen3-ForcedAligner explicitly out of scope.
            segments = (
                [{"text": text, "start": 0.0, "end": duration}] if text else []
            )

            vocal_activity = vad.detect(local_vocals)

            finished = _now_ms()
            log.info(
                job_id,
                "qwen3 complete",
                {
                    "language": detected_iso,
                    "qwen_language": r.language,
                    "text_chars": len(text),
                    "vocal_regions": len(vocal_activity),
                    "duration_s": round(duration, 2),
                    "device": _qwen3_device,
                },
            )
            return _response(
                job_id,
                started,
                finished,
                language=detected_iso,
                segments=segments,
                vocal_activity=vocal_activity,
                source="qwen3",
                model_used=os.environ.get("QWEN3_MODEL", "Qwen/Qwen3-ASR-1.7B"),
            )

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

    finished = _now_ms()
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

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
    if device.startswith("cuda"):
        dtype = torch.bfloat16
    elif device == "mps":
        # MPS can't load bf16 weights (TypeError in _load_state_dict_into_meta_model).
        # fp16 is Apple's mixed-precision path; any qwen3 ops missing MPS kernels
        # still fall through to _force_qwen3_cpu_reload at inference time.
        dtype = torch.float16
    else:
        dtype = torch.float32
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


# Routing on a missing language hint used to fall straight to DEFAULT_FLOW
# (whisper/vocals), which meant the qwen3 backend never ran for real jobs —
# the product UI doesn't collect a language. Detect instead, and only route
# on the detection when whisper is reasonably sure.
_LID_MIN_PROB = float(os.environ.get("LID_MIN_PROB", "0.5"))

# Cap the lyrics context fed to Qwen3's system prompt. Full song lyrics fit
# comfortably; the cap only guards against pathological inputs.
_QWEN3_CONTEXT_MAX = int(os.environ.get("QWEN3_CONTEXT_MAX_CHARS", "4000"))


def _detect_language(local_vocals: Path, vocal_activity: list[dict], job_id: str) -> str | None:
    """LID on a 30 s window of the vocals stem, anchored at the first sung
    region so an instrumental intro doesn't poison the detection. Returns an
    ISO code, or None when confidence is too low to route on (the whisper
    path then auto-detects per its default behaviour).
    """
    from faster_whisper.audio import decode_audio  # lazy

    start = next((r["start"] for r in vocal_activity if r.get("kind") == "vocals"), 0.0)
    audio = decode_audio(str(local_vocals), sampling_rate=16000)
    lo = int(start * 16000)
    window = audio[lo : lo + 30 * 16000]
    if window.shape[0] < 16000:  # under ~1 s of usable audio — don't guess
        return None
    model = _load_model()
    _, info = model.transcribe(window, language=None, beam_size=1, vad_filter=False)
    prob = float(info.language_probability)
    accepted = prob >= _LID_MIN_PROB
    log.info(
        job_id,
        "language detected",
        {
            "language": info.language,
            "prob": round(prob, 3),
            "window_start_s": round(start, 2),
            "routing_on_it": accepted,
        },
    )
    return info.language if accepted else None


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
    language_hint = language
    log.info(job_id, "starting", {"has_lyrics": bool(known_lyrics), "language": language})

    with tempfile.TemporaryDirectory(prefix=f"transcribe-{job_id}-") as tmp_s:
        tmp = Path(tmp_s)

        # Vocals stem first, before routing: RMS-VAD always runs on it
        # (absence of energy on the isolated stem is ground truth for
        # instrumental breaks), and when no language hint arrived the LID
        # pass needs it too.
        local_vocals = tmp / "vocals.wav"
        log.debug(job_id, "downloading vocals", {"object": object_path_from_gs_uri(vocals_uri)})
        download_file(object_path_from_gs_uri(vocals_uri), local_vocals)
        vocal_activity = vad.detect(local_vocals)

        if not language:
            language = _detect_language(local_vocals, vocal_activity, job_id)

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
        # CPU deploys can't run Qwen3-ASR-1.7B in any useful time. FORCE_WHISPER=1
        # is the deploy-time kill-switch that pins the backend to faster-whisper
        # regardless of the language flow. No-op on GPU deploys (don't set it).
        if backend == "qwen3" and os.environ.get("FORCE_WHISPER") == "1":
            log.info(
                job_id,
                "FORCE_WHISPER=1 — pinning to whisper",
                {"language": language},
            )
            backend = "whisper"

        audio_input = input_for_backend(backend)  # qwen3 → "mix", whisper → "vocals"
        if audio_input == "mix" and not source_uri:
            # Orchestrator didn't pass the original upload (e.g. a stale caller).
            # Transcribing the vocals stem is still correct, just not ideal.
            log.warn(job_id, "flow wanted mix but source_uri missing; using vocals")
            audio_input = "vocals"

        log.info(
            job_id,
            "flow resolved",
            {
                "language_hint": language_hint,
                "language": language,
                "backend": backend,
                "audio_input": audio_input,
                "flow_transcribe": flow.transcribe,
                "flow_align": flow.align,
            },
        )

        if audio_input == "mix":
            local_audio = tmp / "mix.bin"
            log.debug(job_id, "downloading mix", {"object": object_path_from_gs_uri(source_uri)})
            download_file(object_path_from_gs_uri(source_uri), local_audio)
        else:
            local_audio = local_vocals

        if backend == "qwen3":
            # known_lyrics rides qwen-asr's `context` parameter, which lands
            # in the model's system prompt — the local-package equivalent of
            # DashScope's context biasing. Most valuable exactly here (rare
            # vocabulary, Turkish), so don't drop it.
            context = (known_lyrics or "").strip()[:_QWEN3_CONTEXT_MAX]
            if known_lyrics:
                log.info(
                    job_id,
                    "qwen3 biasing with known_lyrics",
                    {"chars": len(context)},
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
                    context=context,
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
                        context=context,
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

"""Per-word forced alignment.

Flow-routed dispatcher. ``shared.flow_for(language).align`` picks between
``"qwen3"`` (Qwen3-ForcedAligner-0.6B, Apache-2.0; 11 languages) and
``"whisperx"`` (wav2vec2 via whisperx.align — the language list whisperx
ships checkpoints for). Qwen3 needs CUDA; without CUDA or without
``qwen-asr`` installed we log a ``backend_downgrade`` and fall back to
whisperx for the whole job. Per-chunk failures fall back one step further
(qwen3 → whisperx for that chunk; whisperx → even-split for that chunk).

Never calls ``whisperx.load_model`` — its VAD URL is dead (MVP scar #4).
Only ``whisperx.load_align_model`` + ``whisperx.align``.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from shared import create_logger, download_file, object_path_from_gs_uri
from shared.flows import flow_for

log = create_logger("align")

# whisperx.load_audio decodes to 16 kHz mono float32 via ffmpeg. Both backends
# operate on the same rate; keeping it as a constant so chunk-slicing maths is
# one-line.
_AUDIO_SR = 16000

# Qwen3-ForcedAligner: per-call audio cap. Qwen's published limit is 5 min.
_QWEN_MAX_SECONDS = 300.0

# Cache loaded wav2vec2 models per language — loading takes seconds.
_align_cache: dict[str, tuple] = {}

# Lazy singleton for the Qwen3-ForcedAligner model. Keyed on nothing — there's
# only one checkpoint (Qwen/Qwen3-ForcedAligner-0.6B).
_qwen_model: Any = None

# Optional import. `qwen-asr[vllm]` pins torchaudio==2.9.1 which conflicts with
# whisperx==3.1.6 / torch==2.2.2, so it's installed only on the GPU Cloud Run
# image via a second pip step. On CPU hosts the import fails → _QWEN3_IMPORT_OK
# stays False → _resolve_backend downgrades cleanly.
_QWEN3_IMPORT_OK = False
try:
    from qwen_asr import Qwen3ForcedAligner  # type: ignore[import-not-found]

    _QWEN3_IMPORT_OK = True
except Exception:
    Qwen3ForcedAligner = None  # type: ignore[assignment,misc]

# ISO 639-1 → Qwen's full-name language identifier. The 11 Qwen-aligner
# languages that shared.flows.flow_for emits "qwen3" for.
_QWEN_LANG: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
}


class _Qwen3SanityError(RuntimeError):
    """Raised when Qwen3 output fails the post-hoc sanity check for a chunk.

    Caller catches this and falls back to whisperx for that chunk only. We
    prefer a deterministic check over trial-and-error exception-catching.
    """


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

def plan_chunks(
    segments: list[dict],
    vocal_activity: list[dict],
    max_seconds: float = _QWEN_MAX_SECONDS,
) -> list[list[dict]]:
    """Group segments into ≤max_seconds windows, preferring to split at
    instrumental regions in ``vocal_activity``. Never splits a segment.

    Qwen3-ForcedAligner has a hard 5-minute-per-call limit; whisperx is
    chunk-friendly. Concatenating the returned chunks in order yields the
    original ``segments`` list.
    """
    if not segments:
        return []

    instrumental = [
        (float(r["start"]), float(r["end"]))
        for r in vocal_activity
        if r.get("kind") == "instrumental"
    ]

    def is_break(prev_end: float, next_start: float) -> bool:
        # Closed-interval overlap so a break that begins exactly at the next
        # segment's start (or ends exactly at the previous segment's end) still
        # counts. transcribe routinely emits boundaries that coincide.
        return any(a <= next_start and b >= prev_end for a, b in instrumental)

    start0 = float(segments[0]["start"])
    fit_upto = 1
    while (
        fit_upto < len(segments)
        and float(segments[fit_upto]["end"]) - start0 <= max_seconds
    ):
        fit_upto += 1

    if fit_upto == len(segments):
        span = float(segments[-1]["end"]) - start0
        if span > max_seconds:
            log.warn(
                None,
                "chunk exceeds limit; single segment kept whole",
                {"span": span, "limit": max_seconds},
            )
        return [list(segments)]

    split_at = fit_upto
    for i in range(fit_upto, 0, -1):
        prev_end = float(segments[i - 1]["end"])
        next_start = float(segments[i]["start"])
        if is_break(prev_end, next_start):
            split_at = i
            break

    first = list(segments[:split_at])
    rest = list(segments[split_at:])
    span = float(first[-1]["end"]) - float(first[0]["start"])
    if span > max_seconds:
        log.warn(
            None,
            "chunk exceeds limit; single segment kept whole",
            {"span": span, "limit": max_seconds},
        )
    return [first] + plan_chunks(rest, vocal_activity, max_seconds)


# --------------------------------------------------------------------------- #
# Backend resolution
# --------------------------------------------------------------------------- #

def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_backend(job_id: str, flow_align: str) -> str:
    """Turn the advisory flow choice into a concrete backend name.

    Qwen3 needs CUDA + the qwen-asr package actually importable. If either
    is missing we log ``backend_downgrade`` so ops can see at a glance why
    a GPU-routed job ended up on CPU.
    """
    if flow_align != "qwen3":
        return flow_align
    cuda = _cuda_available()
    if _QWEN3_IMPORT_OK and cuda:
        return "qwen3"
    log.warn(
        job_id,
        "backend_downgrade",
        {
            "requested": "qwen3",
            "resolved": "whisperx",
            "qwen3_import_ok": _QWEN3_IMPORT_OK,
            "cuda_available": cuda,
        },
    )
    return "whisperx"


# --------------------------------------------------------------------------- #
# whisperx backend
# --------------------------------------------------------------------------- #

def _load_whisperx(language: str) -> tuple:
    if language in _align_cache:
        return _align_cache[language]
    import whisperx  # heavy; lazy

    log.info(None, "loading align model", {"language": language})
    model, meta = whisperx.load_align_model(language_code=language, device="cpu")
    _align_cache[language] = (model, meta)
    return model, meta


def _whisperx_checkpoint(language: str) -> str:
    # Report the real HF/torchaudio checkpoint whisperx resolved to, so the
    # response's model_used reflects what actually ran rather than a generic
    # "whisperx" label.
    try:
        from whisperx.alignment import (
            DEFAULT_ALIGN_MODELS_HF,
            DEFAULT_ALIGN_MODELS_TORCH,
        )
    except Exception:
        return f"whisperx:{language}"
    if language in DEFAULT_ALIGN_MODELS_HF:
        return DEFAULT_ALIGN_MODELS_HF[language]
    if language in DEFAULT_ALIGN_MODELS_TORCH:
        return f"torchaudio:{DEFAULT_ALIGN_MODELS_TORCH[language]}"
    return f"whisperx:{language}"


def _align_whisperx(audio: Any, chunk: list[dict], language: str) -> list[dict]:
    """Forced-align one chunk of segments using whisperx's wav2vec2.

    Takes the preloaded full-track audio array (16 kHz mono float32).
    whisperx.align internally slices by seg.start/end, so we can pass the
    whole array and only the chunk's segments.
    """
    import whisperx  # lazy

    model_a, meta = _load_whisperx(language)
    aligned = whisperx.align(
        chunk, model_a, meta, audio, "cpu", return_char_alignments=False,
    )

    out: list[dict] = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words") or []:
            text = (w.get("word") or "").strip()
            start = w.get("start")
            end = w.get("end")
            if not text or start is None or end is None or end <= start:
                continue
            entry = {"text": text, "start": float(start), "end": float(end)}
            if "score" in w and w["score"] is not None:
                entry["score"] = float(w["score"])
            out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# Qwen3 backend
# --------------------------------------------------------------------------- #

def _load_qwen() -> Any:
    """Lazy singleton load of Qwen3-ForcedAligner-0.6B.

    Only called when _resolve_backend has already confirmed CUDA + import.
    """
    global _qwen_model
    if _qwen_model is not None:
        return _qwen_model
    assert _QWEN3_IMPORT_OK, "guarded by _resolve_backend"
    import torch

    log.info(None, "loading qwen3 aligner", {"model": "Qwen/Qwen3-ForcedAligner-0.6B"})
    _qwen_model = Qwen3ForcedAligner.from_pretrained(  # type: ignore[union-attr]
        "Qwen/Qwen3-ForcedAligner-0.6B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    return _qwen_model


def _align_qwen3(
    tmp: Path,
    audio: Any,
    chunk: list[dict],
    language: str,
    idx: int,
) -> list[dict]:
    """Forced-align one chunk via Qwen3-ForcedAligner. Raises on sanity error.

    Slices the preloaded audio to the chunk's [start, end] window, writes a
    small WAV into ``tmp``, runs Qwen3, rebases timestamps by +chunk_start,
    and validates the output before returning.
    """
    import soundfile as sf

    lang_name = _QWEN_LANG.get(language.lower())
    if lang_name is None:
        # Routing layer should have prevented this; surface loudly.
        raise _Qwen3SanityError(f"qwen3 has no aligner for language={language!r}")

    if not chunk:
        return []

    chunk_start = float(chunk[0]["start"])
    chunk_end = float(chunk[-1]["end"])
    text = " ".join(
        (seg.get("text") or "").strip() for seg in chunk if (seg.get("text") or "").strip()
    )
    if not text:
        return []

    s_sample = max(0, int(chunk_start * _AUDIO_SR))
    e_sample = min(len(audio), int(chunk_end * _AUDIO_SR))
    if e_sample <= s_sample:
        raise _Qwen3SanityError(
            f"empty chunk window sample range [{s_sample}, {e_sample})"
        )

    slice_path = tmp / f"qwen_chunk_{idx:03d}.wav"
    sf.write(str(slice_path), audio[s_sample:e_sample], _AUDIO_SR)

    results = _load_qwen().align(
        audio=str(slice_path), text=text, language=lang_name,
    )
    if not results:
        raise _Qwen3SanityError("qwen3 returned no results")
    items = results[0]

    # Rebase + post-hoc sanity check. Anything that fails triggers a per-chunk
    # whisperx fallback in the caller — no trial-and-error exception handling.
    words: list[dict] = []
    prev_start = -1.0
    # 50 ms slack on the window edges to tolerate Qwen's own rounding.
    lo = chunk_start - 0.05
    hi = chunk_end + 0.05
    for item in items:
        w_text = getattr(item, "text", None)
        w_s = getattr(item, "start_time", None)
        w_e = getattr(item, "end_time", None)
        if not w_text or w_s is None or w_e is None:
            raise _Qwen3SanityError("qwen3 item missing fields")
        ws = float(w_s) + chunk_start
        we = float(w_e) + chunk_start
        if not (ws < we):
            raise _Qwen3SanityError(f"non-increasing word span {ws} >= {we}")
        if ws < prev_start:
            raise _Qwen3SanityError(
                f"word starts not monotonic: {ws} < prev {prev_start}"
            )
        if ws < lo or we > hi:
            raise _Qwen3SanityError(
                f"word [{ws}, {we}] outside chunk [{lo}, {hi}]"
            )
        if we - ws > 5.0:
            raise _Qwen3SanityError(
                f"word span exceeds 5s: {we - ws:.2f}s on {w_text!r}"
            )
        prev_start = ws
        words.append({"text": str(w_text), "start": ws, "end": we})
    return words


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _synthesize_words(segments: list[dict]) -> list[dict]:
    """Fallback: evenly distribute each segment's tokens across its duration."""
    out: list[dict] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        toks = text.split()
        if not toks or seg["end"] <= seg["start"]:
            continue
        step = (seg["end"] - seg["start"]) / len(toks)
        for i, tok in enumerate(toks):
            out.append({
                "text": tok,
                "start": float(seg["start"]) + i * step,
                "end": float(seg["start"]) + (i + 1) * step,
            })
    return out


def _model_name(backend: str, language: str) -> str:
    if backend == "qwen3":
        return "Qwen/Qwen3-ForcedAligner-0.6B"
    if backend == "whisperx":
        return _whisperx_checkpoint(language)
    return "even-split"


def _response(
    job_id: str,
    started: int,
    finished: int,
    words: list[dict],
    vocal_activity: list[dict],
    *,
    source: str,
    model_used: str,
    diagnostics: dict | None = None,
) -> dict:
    out: dict[str, Any] = {
        "job_id": job_id,
        "stage": "align",
        "started_at": started,
        "finished_at": finished,
        "duration_ms": finished - started,
        "words": words,
        "vocal_activity": vocal_activity,
        "source": source,
        "model_used": model_used,
    }
    if diagnostics is not None:
        out["diagnostics"] = diagnostics
    return out


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def run(
    job_id: str,
    vocals_uri: str,
    segments: list[dict],
    language: str,
    vocal_activity: list[dict],
) -> dict:
    started = int(time.time() * 1000)
    log.info(
        job_id, "starting",
        {"segments": len(segments), "language": language,
         "vocal_regions": len(vocal_activity)},
    )

    flow = flow_for(language)
    backend = _resolve_backend(job_id, flow.align)
    log.info(
        job_id, "flow resolved",
        {"language": language, "flow_align": flow.align, "backend": backend},
    )

    qwen3_fallback_chunks: list[int] = []
    whisperx_fallback_chunks: list[int] = []
    words: list[dict] = []
    chunks: list[list[dict]] = []

    with tempfile.TemporaryDirectory(prefix=f"align-{job_id}-") as tmp_s:
        tmp = Path(tmp_s)
        vocals_obj = object_path_from_gs_uri(vocals_uri)
        local_vocals = tmp / "vocals.wav"
        download_file(vocals_obj, local_vocals)

        # Decode once; both backends reuse the float32 @ 16 kHz array.
        import whisperx  # lazy

        audio = whisperx.load_audio(str(local_vocals))

        chunks = plan_chunks(segments, vocal_activity, max_seconds=_QWEN_MAX_SECONDS)
        log.info(
            job_id, "chunk plan",
            {"chunk_count": len(chunks),
             "backend": backend,
             "audio_seconds": round(len(audio) / _AUDIO_SR, 2)},
        )

        for i, chunk in enumerate(chunks):
            if backend == "qwen3":
                try:
                    words += _align_qwen3(tmp, audio, chunk, language, idx=i)
                    continue
                except Exception as e:
                    log.warn(
                        job_id, "qwen3_fallback",
                        {"chunk_index": i,
                         "err": f"{type(e).__name__}: {e}"},
                    )
                    qwen3_fallback_chunks.append(i)
                    # fall through to whisperx for this chunk

            try:
                words += _align_whisperx(audio, chunk, language)
            except Exception as e:
                log.warn(
                    job_id, "whisperx_fallback",
                    {"chunk_index": i,
                     "err": f"{type(e).__name__}: {e}"},
                )
                whisperx_fallback_chunks.append(i)
                words += _synthesize_words(chunk)

    finished = int(time.time() * 1000)

    # Effective source: if every chunk fell back off the requested backend,
    # report what actually produced (most of) the words. Per-chunk details
    # live in diagnostics.
    effective_source = backend
    if backend == "qwen3" and len(qwen3_fallback_chunks) == len(chunks):
        effective_source = "whisperx"
    if effective_source == "whisperx" and whisperx_fallback_chunks:
        # whisperx still ran on most; downgrade label only if ALL failed.
        if len(whisperx_fallback_chunks) == len(chunks):
            effective_source = "even-split"

    diagnostics: dict[str, Any] = {
        "backend": backend,
        "chunk_count": len(chunks),
        "cuda_available": _cuda_available(),
    }
    if qwen3_fallback_chunks:
        diagnostics["qwen3_fallback_chunks"] = qwen3_fallback_chunks
    if whisperx_fallback_chunks:
        diagnostics["whisperx_fallback_chunks"] = whisperx_fallback_chunks

    log.info(
        job_id, "done",
        {"words": len(words),
         "duration_ms": finished - started,
         "source": effective_source,
         "qwen3_fallbacks": len(qwen3_fallback_chunks),
         "whisperx_fallbacks": len(whisperx_fallback_chunks)},
    )
    return _response(
        job_id, started, finished, words, vocal_activity,
        source=effective_source,
        model_used=_model_name(effective_source, language),
        diagnostics=diagnostics,
    )

"""Per-word forced alignment via whisperx wav2vec2.

Takes transcribe's segment-level output and returns per-word start/end
timings. Calls *only* `whisperx.load_align_model` + `whisperx.align()` —
never `whisperx.load_model` (its VAD URL is dead, MVP scar #4).
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from shared import create_logger, download_file, object_path_from_gs_uri

log = create_logger("align")

# cache the loaded align model per language — loading wav2vec2 takes seconds
_align_cache: dict[str, tuple] = {}


def plan_chunks(
    segments: list[dict],
    vocal_activity: list[dict],
    max_seconds: float = 300.0,
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

    # How many leading segments fit inside max_seconds starting from segments[0]?
    # fit_upto=1 always holds: we never split a segment, even if it alone exceeds.
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

    # Must split somewhere in [1, fit_upto]. Prefer the latest instrumental
    # break; fall back to fit_upto (the last boundary that keeps the first
    # chunk under the limit).
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


def _load(language: str):
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


def run(
    job_id: str,
    vocals_uri: str,
    segments: list[dict],
    language: str,
    vocal_activity: list[dict],
) -> dict:
    started = int(time.time() * 1000)
    log.info(job_id, "starting", {"segments": len(segments), "language": language})

    import whisperx  # lazy
    with tempfile.TemporaryDirectory(prefix=f"align-{job_id}-") as tmp_s:
        tmp = Path(tmp_s)
        vocals_obj = object_path_from_gs_uri(vocals_uri)
        local_vocals = tmp / "vocals.wav"
        download_file(vocals_obj, local_vocals)
        audio = whisperx.load_audio(str(local_vocals))

        try:
            model_a, meta = _load(language)
        except Exception as e:
            # No alignment model for this language — fall back to segment-level
            # word splits so downstream stages still have something to render.
            log.warn(job_id, "no align model; synthesizing word timings", {"err": str(e)})
            words = _synthesize_words(segments)
            finished = int(time.time() * 1000)
            return _response(
                job_id, started, finished, words, vocal_activity,
                source="even-split", model_used="even-split",
            )

        aligned = whisperx.align(
            segments, model_a, meta, audio, "cpu", return_char_alignments=False,
        )

    # whisperx returns {"segments": [{..., "words": [{word, start, end, score?}, ...]}]}
    words: list[dict] = []
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
            words.append(entry)

    finished = int(time.time() * 1000)
    log.info(job_id, "done", {"words": len(words), "duration_ms": finished - started})
    return _response(
        job_id, started, finished, words, vocal_activity,
        source="whisperx", model_used=_whisperx_checkpoint(language),
    )


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


def _response(
    job_id: str,
    started: int,
    finished: int,
    words: list[dict],
    vocal_activity: list[dict],
    *,
    source: str,
    model_used: str,
) -> dict:
    return {
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

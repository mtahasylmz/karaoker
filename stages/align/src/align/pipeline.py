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


def _load(language: str):
    if language in _align_cache:
        return _align_cache[language]
    import whisperx  # heavy; lazy
    log.info(None, "loading align model", {"language": language})
    model, meta = whisperx.load_align_model(language_code=language, device="cpu")
    _align_cache[language] = (model, meta)
    return model, meta


def run(
    job_id: str,
    vocals_uri: str,
    segments: list[dict],
    language: str,
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
            return _response(job_id, started, finished, words)

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
    return _response(job_id, started, finished, words)


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


def _response(job_id: str, started: int, finished: int, words: list[dict]) -> dict:
    return {
        "job_id": job_id,
        "stage": "align",
        "started_at": started,
        "finished_at": finished,
        "duration_ms": finished - started,
        "words": words,
    }

"""Pure pipeline logic: flow routing, chunk planning, word repair, manifest.

Zero GPU/IO — everything here is unit-testable on plain dicts and paths.
Harvested from the pre-fresh-start stages (shared/flows.py, align/pipeline.py)
and simplified.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

# Qwen3-ForcedAligner: per-call audio hard cap (published limit is 5 min).
QWEN_MAX_SECONDS = 300.0

# Soft packing target for qwen3 chunks. 300 s is the model's HARD cap, but
# quality collapses well before it: on an A40, a 285 s chunk returned 57%
# degenerate word spans while the same audio in ~60-130 s chunks aligned
# near-clean. Pack to this target; only the hard cap forces accepting an
# unsplittable single segment.
QWEN_TARGET_SECONDS = float(os.environ.get("QWEN_CHUNK_TARGET_S", "120"))

# Per-word span cap for the sanity checker. Generous on purpose: this is
# singing, and held notes routinely run 5-10 s (speech-derived limits reject
# real lyrics). Longer spans are clamped, not rejected.
MAX_WORD_SPAN_S = 12.0


class SanityError(RuntimeError):
    """Aligner output for a chunk is structurally garbage; fall back."""


# --------------------------------------------------------------------------- #
# Flow routing
# --------------------------------------------------------------------------- #

class Flow(NamedTuple):
    transcribe: str        # "qwen3" | "whisper"
    transcribe_input: str  # "mix" | "vocals"
    align: str             # "qwen3" | "whisperx"


# Qwen3-ForcedAligner-0.6B language coverage (the 11 langs Qwen ships an
# aligner for). Anything outside routes to whisperx/wav2vec2.
QWEN_ALIGN_LANGS = frozenset(
    {"en", "zh", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es"}
)

# Languages where Qwen3-ASR is preferred as the transcriber (30 languages,
# outperforms Whisper on singing per its published benchmarks).
QWEN_TRANSCRIBE_LANGS = frozenset(
    {
        "en", "zh", "yue", "ar", "de", "fr", "es", "pt", "id", "it", "ko",
        "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv", "da", "fi",
        "pl", "cs", "fil", "fa", "el", "hu", "mk", "ro",
    }
)

DEFAULT_FLOW = Flow("whisper", "vocals", "whisperx")


def flow_for(language: str | None) -> Flow:
    if not language:
        return DEFAULT_FLOW
    lang = language.lower()
    transcribe = "qwen3" if lang in QWEN_TRANSCRIBE_LANGS else "whisper"
    return Flow(
        transcribe=transcribe,
        transcribe_input=input_for_backend(transcribe),
        align="qwen3" if lang in QWEN_ALIGN_LANGS else "whisperx",
    )


def input_for_backend(backend: str) -> str:
    """Audio the backend expects. Qwen3-ASR was trained on mixes with BGM;
    faster-whisper prefers the vocals stem."""
    return "mix" if backend == "qwen3" else "vocals"


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

def _spans(vocal_activity: list[dict], kind: str) -> list[tuple[float, float]]:
    return [
        (float(r["start"]), float(r["end"]))
        for r in vocal_activity
        if r.get("kind") == kind
    ]


def _overlaps(
    spans: list[tuple[float, float]], start: float, end: float
) -> list[tuple[float, float]]:
    """Clip spans to [start, end], keeping only intersecting ones."""
    return [(max(a, start), min(b, end)) for a, b in spans if a < end and b > start]


def _allocate_words(
    words: list[str], spans: list[tuple[float, float]], total: float
) -> list[dict]:
    """One sub-segment per span, words allocated proportionally by duration
    (the last span takes the remainder)."""
    out: list[dict] = []
    idx = 0
    for i, (a, b) in enumerate(spans):
        if i == len(spans) - 1:
            part = words[idx:]
        else:
            count = max(0, round(len(words) * ((b - a) / total)))
            part = words[idx:idx + count]
            idx += count
        if part:
            out.append({"text": " ".join(part), "start": a, "end": b})
    return out


def _split_segment(seg: dict, vocals: list[tuple[float, float]]) -> list[dict]:
    words = ((seg.get("text") or "").strip()).split()
    spans = _overlaps(vocals, float(seg["start"]), float(seg["end"]))
    total = sum(b - a for a, b in spans)
    if len(spans) <= 1 or not words or total <= 0:
        return [dict(seg)]
    return _allocate_words(words, spans, total)


def split_at_vad_breaks(
    segments: list[dict], vocal_activity: list[dict]
) -> list[dict]:
    """Split any segment that spans >=2 vocals regions into one sub-segment
    per region, allocating words proportionally by each region's duration.

    Qwen3-ASR legitimately emits one coarse segment for the whole song;
    plan_chunks never splits a segment, so without this a single-segment
    input yields one chunk spanning the full audio. Segments inside a single
    vocals region pass through unchanged.
    """
    vocals = _spans(vocal_activity, "vocals")
    if not vocals:
        return list(segments)
    out: list[dict] = []
    for seg in segments:
        out.extend(_split_segment(seg, vocals))
    return out


def _fit_count(segments: list[dict], target: float) -> int:
    """Longest prefix (always >= 1) whose time window fits in ``target``."""
    start0 = float(segments[0]["start"])
    n = 1
    while n < len(segments) and float(segments[n]["end"]) - start0 <= target:
        n += 1
    return n


def _break_index(
    segments: list[dict], fit_upto: int, instrumental: list[tuple[float, float]]
) -> int:
    """Latest boundary <= fit_upto that coincides with an instrumental
    region. Closed-interval overlap: a break coinciding exactly with a
    segment boundary still counts."""
    for i in range(fit_upto, 0, -1):
        prev_end = float(segments[i - 1]["end"])
        next_start = float(segments[i]["start"])
        if any(a <= next_start and b >= prev_end for a, b in instrumental):
            return i
    return fit_upto


def plan_chunks(
    segments: list[dict],
    vocal_activity: list[dict],
    max_seconds: float = QWEN_MAX_SECONDS,
    target_seconds: float | None = None,
) -> list[list[dict]]:
    """Group segments into windows, preferring to split at instrumental
    regions. Never splits a segment; concatenating the returned chunks in
    order yields the original ``segments`` list.

    ``target_seconds`` is the soft packing size (defaults to max_seconds);
    ``max_seconds`` only matters when a single unsplittable segment exceeds
    the target.
    """
    if not segments:
        return []
    target = min(target_seconds or max_seconds, max_seconds)
    fit_upto = _fit_count(segments, target)
    if fit_upto == len(segments):
        return [list(segments)]
    split_at = _break_index(segments, fit_upto, _spans(vocal_activity, "instrumental"))
    return [list(segments[:split_at])] + plan_chunks(
        list(segments[split_at:]), vocal_activity, max_seconds, target_seconds
    )


# --------------------------------------------------------------------------- #
# Aligner-output repair: repair isolated anomalies, reject systemic garbage
# --------------------------------------------------------------------------- #

def _rebase_item(item: dict, chunk_start: float) -> tuple[str, float, float]:
    """Validate one aligner item and rebase its span to absolute time."""
    text = item.get("text")
    start = item.get("start")
    end = item.get("end")
    if not text or start is None or end is None:
        raise SanityError("aligner item missing fields")
    ws = float(start) + chunk_start
    we = float(end) + chunk_start
    if we < ws:
        raise SanityError(f"decreasing word span {ws} > {we}")
    return str(text), ws, we


def _repair_span(ws: float, we: float, hi: float) -> tuple[float, int]:
    """Clamp explainable span anomalies; returns (new_end, repair_count)."""
    repairs = 0
    if we - ws < 0.02:  # frame-quantization tie -> nominal 20 ms
        we = min(ws + 0.02, hi)
        repairs += 1
    if we - ws > MAX_WORD_SPAN_S:
        # Clamp BEFORE the window check: an absurd 30 s "word" otherwise
        # reads as out-of-window and used to discard the chunk.
        we = ws + MAX_WORD_SPAN_S
        repairs += 1
    return we, repairs


def _starts_in_window(ws: float, prev_start: float, lo: float, hi: float) -> bool:
    """In-order and starting inside the audio window."""
    return ws >= prev_start and lo <= ws <= hi


def _require_isolated(repaired: int, dropped: int, total: int) -> None:
    anomalies = repaired + dropped
    if anomalies > max(2, total // 5):
        raise SanityError(
            f"{anomalies}/{total} anomalies (repaired {repaired}, "
            f"dropped {dropped}) — systemic"
        )


def repair_words(
    items: list[dict], chunk_start: float, chunk_end: float
) -> list[dict]:
    """Rebase chunk-relative word items to absolute time and sanity-repair.

    ``items``: [{"text", "start", "end"}] with times relative to the chunk
    slice. Policy (learned on real A40 runs, where single-word anomalies used
    to discard whole chunks of otherwise-good alignment):

    - REPAIR isolated, explainable anomalies: zero-length spans (frame
      quantization ties) and over-long spans (held sung notes blow past any
      speech-derived cap).
    - REJECT structural garbage via SanityError: decreasing spans, missing
      fields, or repairs+drops on >20% of words (systemic, not isolated).
    """
    words: list[dict] = []
    prev_start = -1.0
    repaired = 0
    dropped = 0
    # 50 ms slack on the window edges to tolerate the aligner's own rounding.
    lo = chunk_start - 0.05
    hi = chunk_end + 0.05
    for item in items:
        text, ws, we = _rebase_item(item, chunk_start)
        we, repairs = _repair_span(ws, we, hi)
        repaired += repairs
        if not _starts_in_window(ws, prev_start, lo, hi):
            # Out-of-order or starts outside the audio entirely. Drop the
            # offender; one missing word beats losing the chunk's alignment.
            dropped += 1
            continue
        if we > hi:  # ends past the window: pull the tail in
            we = hi
            # Clamp to prev_start: the pull-in must not retract a start below
            # the previous word's — manifest starts are non-decreasing (pipeline-2).
            ws = max(prev_start, min(ws, max(lo, we - 0.02)))
            repaired += 1
        prev_start = ws
        words.append({"text": text, "start": ws, "end": we})
    _require_isolated(repaired, dropped, len(items))
    if not words:
        raise SanityError("no words survived sanity checks")
    return words


def synthesize_words(segments: list[dict]) -> list[dict]:
    """Last-resort fallback: evenly split each segment's tokens over its span."""
    out: list[dict] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
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


# --------------------------------------------------------------------------- #
# Manifest + CLI preflight decisions
# --------------------------------------------------------------------------- #

def clean_words(words: list[dict]) -> list[dict]:
    return [w for w in words if (w.get("text") or "").strip() and w["end"] > w["start"]]


def build_manifest(
    language: str,
    duration: float,
    words: list[dict],
    vocal_activity: list[dict],
) -> dict:
    return {
        "language": language,
        "duration": duration,
        "words": clean_words(words),
        "vocal_activity": vocal_activity,
    }


def resolve_out_dir(video: str, out: str | None) -> Path:
    return Path(out) if out else Path(Path(video).stem)

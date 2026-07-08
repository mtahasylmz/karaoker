"""Chunk planning for the qwen3 aligner: VAD-aware splitting and packing.

Pure logic (split from core.py), consumed by cli._align. Groups ASR segments
into aligner-sized windows, preferring to split at instrumental breaks.
"""

from __future__ import annotations

import os

# Qwen3-ForcedAligner: per-call audio hard cap (published limit is 5 min).
QWEN_MAX_SECONDS = 300.0

# Soft packing target for qwen3 chunks. 300 s is the model's HARD cap, but
# quality collapses well before it: on an A40, a 285 s chunk returned 57%
# degenerate word spans while the same audio in ~60-130 s chunks aligned
# near-clean. Pack to this target; only the hard cap forces accepting an
# unsplittable single segment.
QWEN_TARGET_SECONDS = float(os.environ.get("QWEN_CHUNK_TARGET_S", "120"))


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

"""Word-list sanitation: repair qwen3 aligner artifacts, drop degenerate words.

Pure logic (split from core.py). ``repair_words`` (backends.align_qwen3)
repairs isolated aligner anomalies and rejects systemic garbage;
``clean_words`` is the shared final filter for degenerate entries.
"""

from __future__ import annotations

# Per-word span cap for the sanity checker. Generous on purpose: this is
# singing, and held notes routinely run 5-10 s (speech-derived limits reject
# real lyrics). Longer spans are clamped, not rejected.
MAX_WORD_SPAN_S = 12.0


class SanityError(RuntimeError):
    """Aligner output for a chunk is structurally garbage; fall back."""


def clean_words(words: list[dict]) -> list[dict]:
    return [w for w in words if (w.get("text") or "").strip() and w["end"] > w["start"]]


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

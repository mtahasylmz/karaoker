from __future__ import annotations

import pytest

from align.pipeline import plan_chunks


def _seg(start: float, end: float, text: str = "x") -> dict:
    return {"start": start, "end": end, "text": text}


def _instr(start: float, end: float) -> dict:
    return {"start": start, "end": end, "kind": "instrumental"}


def _vox(start: float, end: float) -> dict:
    return {"start": start, "end": end, "kind": "vocals"}


def _flatten(chunks: list[list[dict]]) -> list[dict]:
    return [s for c in chunks for s in c]


def test_empty_input() -> None:
    assert plan_chunks([], []) == []


def test_single_chunk_when_short() -> None:
    segments = [_seg(i * 2.0, i * 2.0 + 1.5) for i in range(10)]  # span = 19.5s
    chunks = plan_chunks(segments, [], max_seconds=300.0)
    assert len(chunks) == 1
    assert chunks[0] == segments


def test_splits_at_instrumental() -> None:
    # 6 segments of 60s each → 360s total, must split.
    # Instrumental gap falls between segments 3 and 4 (180..182).
    segments = [_seg(i * 60.0, i * 60.0 + 55.0) for i in range(6)]
    vocal_activity = [_instr(180.0, 182.0)]
    chunks = plan_chunks(segments, vocal_activity, max_seconds=300.0)
    assert len(chunks) == 2
    assert chunks[0] == segments[:3]
    assert chunks[1] == segments[3:]
    assert _flatten(chunks) == segments


def test_splits_at_last_boundary_without_vad() -> None:
    # 6 segments of 60s each, no instrumental breaks. Max 300s.
    # First chunk holds segments 0..4 (span = 295s), chunk 2 holds segment 5.
    segments = [_seg(i * 60.0, i * 60.0 + 55.0) for i in range(6)]
    chunks = plan_chunks(segments, [], max_seconds=300.0)
    assert len(chunks) == 2
    assert chunks[0] == segments[:5]
    assert chunks[1] == segments[5:]


def test_preserves_order_and_count_property() -> None:
    segments = [_seg(i * 30.0, i * 30.0 + 25.0) for i in range(40)]  # ~1200s total
    vocal_activity = [_instr(150.0, 155.0), _instr(600.0, 610.0), _vox(0.0, 150.0)]
    chunks = plan_chunks(segments, vocal_activity, max_seconds=300.0)
    assert _flatten(chunks) == segments
    for c in chunks:
        assert c[-1]["end"] - c[0]["start"] <= 300.0 or len(c) == 1


def test_single_oversize_segment_kept_whole() -> None:
    segments = [_seg(0.0, 400.0, text="long")]
    chunks = plan_chunks(segments, [], max_seconds=300.0)
    assert chunks == [segments]


def test_oversize_segment_then_normal() -> None:
    # One 400s segment followed by a short one — oversize kept in its own
    # chunk, trailing segment continues in the next.
    segments = [_seg(0.0, 400.0, "long"), _seg(400.0, 410.0, "short")]
    chunks = plan_chunks(segments, [], max_seconds=300.0)
    assert len(chunks) == 2
    assert chunks[0] == segments[:1]
    assert chunks[1] == segments[1:]


def test_instrumental_after_fit_upto_not_preferred() -> None:
    # Instrumental break exists, but it's past the max_seconds window —
    # must fall back to the last-fitting boundary.
    segments = [_seg(i * 60.0, i * 60.0 + 55.0) for i in range(6)]  # 360s total
    # Instrumental well past 300s (between segments 5 and 6, but there's no 6).
    # Actually: place the break AFTER fit_upto to confirm it's not chosen.
    vocal_activity = [_instr(310.0, 315.0)]
    chunks = plan_chunks(segments, vocal_activity, max_seconds=300.0)
    # Should split at the last fitting boundary, not at the out-of-window break.
    assert chunks[0] == segments[:5]
    assert chunks[1] == segments[5:]


@pytest.mark.parametrize(
    "max_s",
    [60.0, 120.0, 300.0, 900.0],
)
def test_property_random_spans(max_s: float) -> None:
    segments = [_seg(i * 7.3, i * 7.3 + 6.0) for i in range(50)]
    vocal_activity = [_instr(73.0, 75.0), _instr(200.0, 205.0)]
    chunks = plan_chunks(segments, vocal_activity, max_seconds=max_s)
    assert _flatten(chunks) == segments
    for c in chunks:
        span = c[-1]["end"] - c[0]["start"]
        assert span <= max_s or len(c) == 1

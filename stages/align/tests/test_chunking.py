from __future__ import annotations

import pytest

from align.pipeline import plan_chunks, split_at_vad_breaks


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


# --------------------------------------------------------------------------- #
# split_at_vad_breaks
# --------------------------------------------------------------------------- #


def test_split_passthrough_single_region() -> None:
    # Segment entirely inside one vocals region → unchanged.
    va = [_vox(10.0, 90.0)]
    segs = [_seg(20.0, 80.0, "alpha beta gamma")]
    out = split_at_vad_breaks(segs, va)
    assert len(out) == 1
    assert out[0]["text"] == "alpha beta gamma"
    assert out[0]["start"] == 20.0
    assert out[0]["end"] == 80.0


def test_split_passthrough_no_vocals_regions() -> None:
    # No vocals regions declared → original segments pass through.
    va = [_instr(0.0, 50.0)]
    segs = [_seg(0.0, 50.0, "hello world")]
    out = split_at_vad_breaks(segs, va)
    assert out == segs


def test_split_single_coarse_segment_at_two_regions() -> None:
    # The real qwen3-transcribe shape: one segment spanning the whole song,
    # two vocals regions separated by an instrumental break.
    va = [
        _instr(0.0, 20.0),
        _vox(20.0, 80.0),       # 60s
        _instr(80.0, 120.0),
        _vox(120.0, 220.0),     # 100s
        _instr(220.0, 222.0),
    ]
    words = ["w" + str(i) for i in range(160)]
    segs = [_seg(0.0, 222.0, " ".join(words))]
    out = split_at_vad_breaks(segs, va)
    assert len(out) == 2
    assert out[0]["start"] == 20.0 and out[0]["end"] == 80.0
    assert out[1]["start"] == 120.0 and out[1]["end"] == 220.0
    # Proportional allocation: 60/(60+100) = 37.5% → round(160*0.375)=60.
    out0_words = out[0]["text"].split()
    out1_words = out[1]["text"].split()
    assert len(out0_words) == 60
    assert len(out1_words) == 100
    # Order preserved.
    assert out0_words + out1_words == words


def test_split_handles_segment_clipping_to_region() -> None:
    # Segment extends beyond the vocals regions; overlap is clipped.
    va = [_vox(10.0, 40.0), _vox(60.0, 90.0)]
    segs = [_seg(0.0, 100.0, "a b c d e f")]
    out = split_at_vad_breaks(segs, va)
    assert len(out) == 2
    assert out[0]["start"] == 10.0 and out[0]["end"] == 40.0
    assert out[1]["start"] == 60.0 and out[1]["end"] == 90.0
    # All 6 words accounted for, in order.
    assert " ".join(s["text"] for s in out).split() == ["a", "b", "c", "d", "e", "f"]


def test_split_multi_segment_input_individual_pass_through() -> None:
    # If upstream already gave us multiple segments, each is split or passed
    # through on its own merits.
    va = [_vox(0.0, 50.0), _vox(60.0, 100.0)]
    segs = [
        _seg(5.0, 40.0, "inside one"),                    # passes through
        _seg(10.0, 95.0, "spans two regions four words"), # splits into 2
    ]
    out = split_at_vad_breaks(segs, va)
    # 1 pass-through + 2 from the split → 3 total
    assert len(out) == 3
    assert out[0]["text"] == "inside one"
    assert out[1]["start"] == 10.0 and out[1]["end"] == 50.0
    assert out[2]["start"] == 60.0 and out[2]["end"] == 95.0

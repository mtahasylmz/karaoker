"""Unit tests for chunks.py — VAD-aware splitting and aligner chunk packing.
Moved verbatim from core_test.py when chunks.py was split out of core.py."""

from __future__ import annotations

from annemusic.chunks import plan_chunks, split_at_vad_breaks


def _seg(start: float, end: float, text: str = "x") -> dict:
    return {"start": start, "end": end, "text": text}


def _instr(start: float, end: float) -> dict:
    return {"start": start, "end": end, "kind": "instrumental"}


def _vox(start: float, end: float) -> dict:
    return {"start": start, "end": end, "kind": "vocals"}


def _flatten(chunks: list[list[dict]]) -> list[dict]:
    return [s for c in chunks for s in c]


# --------------------------------------------------------------------------- #
# plan_chunks
# --------------------------------------------------------------------------- #


def test_plan_empty_input():
    assert plan_chunks([], []) == []


def test_plan_single_chunk_when_short():
    segments = [_seg(i * 2.0, i * 2.0 + 1.5) for i in range(10)]
    chunks = plan_chunks(segments, [], max_seconds=300.0)
    assert chunks == [segments]


def test_plan_splits_at_instrumental():
    segments = [_seg(i * 60.0, i * 60.0 + 55.0) for i in range(6)]
    chunks = plan_chunks(segments, [_instr(180.0, 182.0)], max_seconds=300.0)
    assert len(chunks) == 2
    assert chunks[0] == segments[:3]
    assert chunks[1] == segments[3:]
    assert _flatten(chunks) == segments


def test_plan_splits_at_last_boundary_without_vad():
    segments = [_seg(i * 60.0, i * 60.0 + 55.0) for i in range(6)]
    chunks = plan_chunks(segments, [], max_seconds=300.0)
    assert chunks[0] == segments[:5]
    assert chunks[1] == segments[5:]


def test_plan_target_seconds_packs_smaller_than_max():
    # The hard-won A40 fact: pack to ~120 s even though the cap is 300 s.
    segments = [_seg(i * 60.0, i * 60.0 + 55.0) for i in range(6)]
    chunks = plan_chunks(segments, [], max_seconds=300.0, target_seconds=120.0)
    for c in chunks:
        assert c[-1]["end"] - c[0]["start"] <= 120.0 or len(c) == 1
    assert _flatten(chunks) == segments


def test_plan_single_oversize_segment_kept_whole():
    segments = [_seg(0.0, 400.0, "long")]
    assert plan_chunks(segments, [], max_seconds=300.0) == [segments]


def test_plan_oversize_then_normal():
    segments = [_seg(0.0, 400.0, "long"), _seg(400.0, 410.0, "short")]
    chunks = plan_chunks(segments, [], max_seconds=300.0)
    assert chunks == [segments[:1], segments[1:]]


def test_plan_preserves_order_and_count():
    segments = [_seg(i * 30.0, i * 30.0 + 25.0) for i in range(40)]
    va = [_instr(150.0, 155.0), _instr(600.0, 610.0), _vox(0.0, 150.0)]
    chunks = plan_chunks(segments, va, max_seconds=300.0)
    assert _flatten(chunks) == segments
    for c in chunks:
        assert c[-1]["end"] - c[0]["start"] <= 300.0 or len(c) == 1


# --------------------------------------------------------------------------- #
# split_at_vad_breaks
# --------------------------------------------------------------------------- #


def test_split_passthrough_single_region():
    out = split_at_vad_breaks([_seg(20.0, 80.0, "alpha beta gamma")], [_vox(10.0, 90.0)])
    assert out == [_seg(20.0, 80.0, "alpha beta gamma")]


def test_split_passthrough_no_vocals_regions():
    segs = [_seg(0.0, 50.0, "hello world")]
    assert split_at_vad_breaks(segs, [_instr(0.0, 50.0)]) == segs


def test_split_single_coarse_segment_at_two_regions():
    # The real qwen3-transcribe shape: one segment spanning the whole song.
    va = [
        _instr(0.0, 20.0),
        _vox(20.0, 80.0),
        _instr(80.0, 120.0),
        _vox(120.0, 220.0),
        _instr(220.0, 222.0),
    ]
    words = [f"w{i}" for i in range(160)]
    out = split_at_vad_breaks([_seg(0.0, 222.0, " ".join(words))], va)
    assert len(out) == 2
    assert (out[0]["start"], out[0]["end"]) == (20.0, 80.0)
    assert (out[1]["start"], out[1]["end"]) == (120.0, 220.0)
    # Proportional: 60/160s of vocal time -> round(160*0.375)=60 words.
    assert len(out[0]["text"].split()) == 60
    assert out[0]["text"].split() + out[1]["text"].split() == words


def test_split_clips_segment_to_regions():
    out = split_at_vad_breaks(
        [_seg(0.0, 100.0, "a b c d e f")], [_vox(10.0, 40.0), _vox(60.0, 90.0)]
    )
    assert len(out) == 2
    assert (out[0]["start"], out[0]["end"]) == (10.0, 40.0)
    assert (out[1]["start"], out[1]["end"]) == (60.0, 90.0)
    assert " ".join(s["text"] for s in out).split() == ["a", "b", "c", "d", "e", "f"]

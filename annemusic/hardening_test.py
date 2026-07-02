"""Mutation-hardening tests (architect-owned).

Each test kills specific mutmut survivors by pinning behavior the unit suite
left loose: routing-table invariants, chunk-packing boundaries, repair_words
window/threshold edges, VAD contract cases, and the ASS output format.
Kept separate from the coder's unit tests per the architect charter.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from annemusic import ass, backends, core, vad
from annemusic.core import (
    SanityError,
    plan_chunks,
    repair_words,
    split_at_vad_breaks,
    synthesize_words,
)


def _seg(start: float, end: float, text: str = "x") -> dict:
    return {"start": start, "end": end, "text": text}


def _vox(start: float, end: float) -> dict:
    return {"start": start, "end": end, "kind": "vocals"}


def _instr(start: float, end: float) -> dict:
    return {"start": start, "end": end, "kind": "instrumental"}


def _item(text: str, start: float, end: float) -> dict:
    return {"text": text, "start": start, "end": end}


def _w(text: str, start: float, end: float) -> dict:
    return {"text": text, "start": start, "end": end}


# --------------------------------------------------------------------------- #
# Routing tables: cross-module invariants, not copied constants
# --------------------------------------------------------------------------- #


def test_aligner_langs_are_a_strict_subset_of_transcribe_langs():
    # Qwen ships an aligner for 11 of the 30 languages its ASR covers; any
    # language we'd align with qwen3 must also be transcribable by it.
    assert core.QWEN_ALIGN_LANGS < core.QWEN_TRANSCRIBE_LANGS


def test_iso_name_map_covers_exactly_the_routed_languages():
    # Qwen3 wants English-name language args: backends must be able to name
    # every language core routes to it, no more, no fewer.
    assert set(backends.ISO_TO_QWEN) == core.QWEN_TRANSCRIBE_LANGS
    assert len(backends.QWEN_TO_ISO) == len(backends.ISO_TO_QWEN)


def test_qwen_chunk_target_env_knob():
    # Subprocess, not importlib.reload: reloading core in-process would swap
    # class identities (SanityError) under every later test in the session.
    proc = subprocess.run(
        [sys.executable, "-c",
         "import annemusic.core as c; assert c.QWEN_TARGET_SECONDS == 90.0"],
        env={**os.environ, "QWEN_CHUNK_TARGET_S": "90"},
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr


# --------------------------------------------------------------------------- #
# plan_chunks boundaries
# --------------------------------------------------------------------------- #


def test_default_hard_cap_is_the_published_300s():
    segs = [_seg(0.0, 150.0), _seg(150.0, 300.5)]
    assert len(plan_chunks(segs, [])) == 2


def test_window_exactly_at_target_still_fits():
    segs = [_seg(0.0, 60.0), _seg(60.0, 120.0)]
    assert plan_chunks(segs, [], target_seconds=120.0) == [segs]


def test_split_prefers_instrumental_break_even_when_more_fits():
    # An instrumental region ending exactly at a segment boundary counts
    # (closed-interval overlap), and beats splitting at the raw fit point.
    segs = [_seg(0.0, 50.0), _seg(60.0, 110.0), _seg(115.0, 290.0)]
    chunks = plan_chunks(segs, [_instr(45.0, 50.0)], target_seconds=120.0)
    assert chunks == [[segs[0]], [segs[1]], [segs[2]]]


# --------------------------------------------------------------------------- #
# split_at_vad_breaks: span-overlap and word-allocation edges
# --------------------------------------------------------------------------- #


def test_region_touching_segment_end_is_ignored():
    seg = _seg(20.0, 80.0, "a b c d")
    out = split_at_vad_breaks([seg], [_vox(30.0, 40.0), _vox(80.0, 90.0)])
    assert out == [seg]


def test_region_touching_segment_start_is_ignored():
    seg = _seg(20.0, 80.0, "a b c d")
    out = split_at_vad_breaks([seg], [_vox(10.0, 20.0), _vox(30.0, 40.0)])
    assert out == [seg]


def test_far_region_cannot_poison_the_split():
    seg = _seg(0.0, 10.0, "a b c d")
    out = split_at_vad_breaks(
        [seg], [_vox(0.0, 5.0), _vox(5.0, 10.0), _vox(200.0, 300.0)]
    )
    assert len(out) == 2


def test_three_region_split_loses_no_words():
    seg = _seg(0.0, 30.0, "a b c d")
    out = split_at_vad_breaks(
        [seg], [_vox(0.0, 10.0), _vox(10.0, 20.0), _vox(20.0, 30.0)]
    )
    assert [s["text"] for s in out] == ["a", "b", "c d"]


def test_tiny_region_gets_zero_words_not_one():
    seg = _seg(0.0, 20.0, "a b c d")
    out = split_at_vad_breaks(
        [seg], [_vox(0.0, 1.0), _vox(1.0, 2.0), _vox(2.0, 20.0)]
    )
    assert [s["text"] for s in out] == ["a b c d"]


def test_empty_text_multi_region_segment_passes_through():
    seg = _seg(0.0, 100.0, "")
    out = split_at_vad_breaks([seg], [_vox(0.0, 40.0), _vox(60.0, 100.0)])
    assert out == [seg]


def test_single_clipping_region_does_not_rewrite_the_segment():
    seg = _seg(0.0, 100.0, "a b")
    out = split_at_vad_breaks([seg], [_vox(10.0, 40.0)])
    assert out == [seg]


def test_subsecond_regions_still_split():
    seg = _seg(0.0, 10.0, "a b")
    out = split_at_vad_breaks([seg], [_vox(1.0, 1.4), _vox(8.0, 8.4)])
    assert len(out) == 2


# --------------------------------------------------------------------------- #
# repair_words: window edges, repair accounting, systemic thresholds
# --------------------------------------------------------------------------- #


def test_word_span_cap_is_12_seconds():
    words = repair_words(
        [_item("held", 0.1, 30.0), _item("next", 31.0, 31.5)], 0.0, 36.0
    )
    assert words[0]["end"] - words[0]["start"] == pytest.approx(12.0)


def test_missing_text_rejects():
    with pytest.raises(SanityError, match="missing"):
        repair_words([{"text": None, "start": 0.5, "end": 1.0}], 0.0, 10.0)


def test_exact_20ms_spans_are_not_counted_as_repairs():
    # start 0.0 so that end - start is EXACTLY the 0.02 literal (any other
    # base start makes the subtraction drift a few ulps off the boundary).
    items = [_item(f"w{i}", 0.0, 0.02) for i in range(12)]
    assert len(repair_words(items, 0.0, 10.0)) == 12


def test_all_overlong_spans_reject():
    items = [_item(f"w{i}", 1.0 + i * 0.1, 31.0 + i * 0.1) for i in range(12)]
    with pytest.raises(SanityError, match="systemic"):
        repair_words(items, 0.0, 50.0)


def test_three_overlong_in_twenty_pass():
    normal = [_item(f"w{i}", 1.0 + i * 0.1, 1.5 + i * 0.1) for i in range(17)]
    overlong = [_item(f"o{i}", 3.0 + i * 0.1, 33.0 + i * 0.1) for i in range(3)]
    assert len(repair_words(normal + overlong, 0.0, 50.0)) == 20


def test_exact_cap_spans_are_not_counted_as_repairs():
    items = [_item(f"w{i}", i * 0.1, i * 0.1 + 12.0) for i in range(12)]
    assert len(repair_words(items, 0.0, 30.0)) == 12


def test_equal_starts_are_in_order():
    words = repair_words(
        [_item("a", 1.0, 1.4), _item("b", 1.0, 1.5)], 0.0, 10.0
    )
    assert [w["text"] for w in words] == ["a", "b"]


def test_word_exactly_at_window_slack_edges_is_kept():
    words = repair_words(
        [_item("lead", -0.05, 0.3), _item("mid", 1.0, 1.5), _item("edge", 10.05, 10.4)],
        0.0,
        10.0,
    )
    assert [w["text"] for w in words] == ["lead", "mid", "edge"]


def test_word_slightly_outside_slack_is_dropped():
    items = [_item(f"w{i}", 1.0 + i * 0.1, 1.1 + i * 0.1) for i in range(9)]
    words = repair_words([_item("early", -0.5, 0.4)] + items, 0.0, 10.0)
    assert [w["text"] for w in words] == [f"w{i}" for i in range(9)]


def test_tail_pull_in_start_positions():
    words = repair_words(
        [_item("early", 1.0, 1.5), _item("crowd", 10.04, 10.4)], 0.0, 10.0
    )
    assert words[1]["start"] == pytest.approx(10.03)
    assert words[1]["end"] == pytest.approx(10.05)


def test_systemic_drops_reject():
    good = [_item(f"w{i}", 1.0 + i * 0.1, 1.1 + i * 0.1) for i in range(6)]
    ghosts = [_item(f"g{i}", 20.0 + i, 20.5 + i) for i in range(4)]
    with pytest.raises(SanityError, match="systemic"):
        repair_words(good + ghosts, 0.0, 10.0)


def test_three_anomalies_in_five_reject():
    good = [_item(f"w{i}", 1.0 + i * 0.1, 1.1 + i * 0.1) for i in range(2)]
    ghosts = [_item(f"g{i}", 20.0 + i, 20.5 + i) for i in range(3)]
    with pytest.raises(SanityError, match="systemic"):
        repair_words(good + ghosts, 0.0, 10.0)


def test_four_anomalies_in_twenty_pass():
    good = [_item(f"w{i}", 1.0 + i * 0.1, 1.1 + i * 0.1) for i in range(16)]
    ghosts = [_item(f"g{i}", 20.0 + i, 20.5 + i) for i in range(4)]
    assert len(repair_words(good + ghosts, 0.0, 10.0)) == 16


def test_tail_clamp_adds_to_the_repair_count():
    ties = [_item(f"t{i}", 1.0 + i * 0.1, 1.0 + i * 0.1) for i in range(4)]
    good = [_item(f"w{i}", 2.0 + i * 0.1, 2.1 + i * 0.1) for i in range(15)]
    tail = [_item("tail", 9.99, 10.4)]
    with pytest.raises(SanityError, match="systemic"):
        repair_words(ties + good + tail, 0.0, 10.0)


# --------------------------------------------------------------------------- #
# synthesize_words
# --------------------------------------------------------------------------- #


def test_synthesize_missing_text_is_skipped():
    assert synthesize_words([{"start": 0.0, "end": 5.0}]) == []


def test_synthesize_continues_past_degenerate_segment():
    words = synthesize_words([_seg(5.0, 5.0, "x"), _seg(0.0, 4.0, "a b")])
    assert [w["text"] for w in words] == ["a", "b"]


def test_synthesize_fractional_step():
    words = synthesize_words([_seg(0.0, 2.0, "a b c d")])
    assert words[0]["end"] == pytest.approx(0.5)
    assert words[1]["start"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# VAD contract cases
# --------------------------------------------------------------------------- #

_SR = 16000


def _tone(seconds: float, rms: float) -> np.ndarray:
    t = np.arange(int(seconds * _SR)) / _SR
    return (rms * np.sqrt(2.0) * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def test_subframe_audio_is_one_instrumental_region():
    regions = vad.detect_regions(np.zeros(100, dtype=np.float32), _SR)
    assert regions == [{"start": 0.0, "end": 100 / _SR, "kind": "instrumental"}]


def test_mono_file_detect(tmp_path):
    import soundfile as sf

    path = tmp_path / "mono.wav"
    sf.write(str(path), _tone(2.0, 0.25), _SR)
    regions = vad.detect(path)
    assert all(r["kind"] == "vocals" for r in regions)


def test_hum_between_thresholds_stays_instrumental():
    # RMS 0.0095 sits inside the hysteresis band (-46..-40 dBFS): without a
    # crossing of the ON threshold, nothing may ever be marked vocals.
    regions = vad.detect_regions(_tone(2.0, 0.0095), _SR)
    assert all(r["kind"] == "instrumental" for r in regions)


def test_two_second_break_survives_gap_merge():
    audio = np.concatenate(
        [_tone(3.0, 0.20), np.zeros(2 * _SR, dtype=np.float32), _tone(3.0, 0.20)]
    )
    regions = vad.detect_regions(audio, _SR)
    kind = next(r["kind"] for r in regions if r["start"] <= 4.0 < r["end"])
    assert kind == "instrumental"


# --------------------------------------------------------------------------- #
# ASS output format pins (players parse this; the text IS the contract)
# --------------------------------------------------------------------------- #

GOLDEN_HEADER = (
    "[Script Info]\n"
    "ScriptType: v4.00+\n"
    "PlayResX: 1920\n"
    "PlayResY: 1080\n"
    "WrapStyle: 2\n"
    "ScaledBorderAndShadow: yes\n\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
    "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
    "Alignment, MarginL, MarginR, MarginV, Encoding\n"
    "Style: Default,Arial,72,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,"
    "-1,0,0,0,100,100,0,0,1,3,2,2,60,60,80,1\n\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
    "Effect, Text\n"
)


def test_empty_build_is_exactly_the_golden_header():
    assert ass.build_ass([]) == GOLDEN_HEADER


def test_dialogue_line_is_pinned():
    doc = ass.build_ass([_w("hello", 1.0, 1.5), _w("world", 2.2, 2.9)])
    dialogue = [ln for ln in doc.splitlines() if ln.startswith("Dialogue:")]
    assert dialogue == [
        "Dialogue: 0,0:00:01.00,0:00:03.20,Default,,0,0,0,,"
        "{\\kf50}hello {\\k70}{\\kf70}world"
    ]


def test_subcentisecond_word_gets_min_1cs_fill():
    assert "{\\kf1}x" in ass.build_ass([_w("x", 1.0, 1.004)])


def test_one_centisecond_gap_is_not_swallowed():
    doc = ass.build_ass([_w("a", 1.0, 1.5), _w("b", 1.51, 2.0)])
    assert "{\\k1}" in doc


def test_unrenderable_token_does_not_stop_the_line():
    doc = ass.build_ass([_w("ok", 1.0, 1.5), _w("\\", 1.6, 2.0), _w("yes", 2.1, 2.6)])
    dialogue = [ln for ln in doc.splitlines() if ln.startswith("Dialogue:")]
    assert "yes" in dialogue[0]


def test_default_line_packing_and_trailing_newline():
    words = [_w(f"w{i}", float(i), float(i) + 0.5) for i in range(9)]
    doc = ass.build_ass(words)
    dialogue = [ln for ln in doc.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogue) == 2  # max_words_per_line defaults to 8
    assert doc.endswith("\n")


def test_default_break_gap_is_1_5s():
    doc = ass.build_ass([_w("a", 0.0, 0.5), _w("b", 2.1, 2.6)])  # gap 1.6 s
    assert sum(ln.startswith("Dialogue:") for ln in doc.splitlines()) == 2


def test_gap_exactly_at_break_gap_does_not_break():
    lines = ass.group_lines([_w("a", 0.0, 0.5), _w("b", 2.0, 2.5)], 8, 1.5)
    assert len(lines) == 1

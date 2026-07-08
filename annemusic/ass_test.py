"""Unit tests for the line-level ASS renderer."""

from __future__ import annotations

from annemusic.ass import build_ass, fmt_time, sanitize


def _ln(text: str, start: float, end: float) -> dict:
    return {"text": text, "start": start, "end": end}


def test_fmt_time_centiseconds():
    assert fmt_time(0.0) == "0:00:00.00"
    assert fmt_time(1.5) == "0:00:01.50"
    assert fmt_time(61.25) == "0:01:01.25"
    assert fmt_time(3600.0) == "1:00:00.00"


def test_fmt_time_never_negative():
    assert fmt_time(-1.0) == "0:00:00.00"


def test_sanitize_strips_ass_control_chars():
    assert sanitize("a\\b{c}d\ne") == "ab(c)d e"


def test_build_ass_one_dialogue_per_line_no_kf():
    doc = build_ass([_ln("hello world", 1.0, 2.2), _ln("second line", 3.0, 4.0)])
    assert "[Script Info]" in doc and "[V4+ Styles]" in doc and "[Events]" in doc
    dialogue = [ln for ln in doc.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogue) == 2
    assert "\\kf" not in doc and "\\k" not in doc  # no fill animation
    # Line runs from its start to end + 0.3 s tail; plain text.
    assert "Dialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,hello world" == dialogue[0]


def test_build_ass_skips_degenerate_lines():
    doc = build_ass([_ln("", 0.0, 1.0), _ln("ok", 1.0, 2.0), _ln("bad", 3.0, 2.0)])
    dialogue = [ln for ln in doc.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogue) == 1
    assert "ok" in dialogue[0] and "bad" not in dialogue[0]


def test_build_ass_empty_is_header_only():
    doc = build_ass([])
    assert "[Events]" in doc and "Dialogue:" not in doc

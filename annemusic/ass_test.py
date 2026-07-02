"""Unit tests for the \\kf ASS renderer (ported from stages/compose ass.ts)."""

from __future__ import annotations

from annemusic.ass import build_ass, fmt_time, group_lines, sanitize


def _w(text: str, start: float, end: float) -> dict:
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


def test_group_lines_breaks_at_max_words():
    words = [_w(f"w{i}", i * 1.0, i * 1.0 + 0.5) for i in range(10)]
    lines = group_lines(words, max_words=4, break_gap=100.0)
    assert [len(ln) for ln in lines] == [4, 4, 2]
    assert [w for ln in lines for w in ln] == words


def test_group_lines_breaks_at_gap():
    words = [_w("a", 0.0, 0.5), _w("b", 0.6, 1.0), _w("c", 5.0, 5.5)]
    lines = group_lines(words, max_words=8, break_gap=1.5)
    assert len(lines) == 2
    assert [w["text"] for w in lines[0]] == ["a", "b"]
    assert [w["text"] for w in lines[1]] == ["c"]


def test_build_ass_has_kf_tags_and_dialogue():
    words = [_w("hello", 1.0, 1.5), _w("world", 1.6, 2.2)]
    doc = build_ass(words)
    assert "[Script Info]" in doc
    assert "[V4+ Styles]" in doc
    assert "[Events]" in doc
    dialogue = [ln for ln in doc.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogue) == 1
    # 50 cs of fill for "hello", a 10 cs dead gap, 60 cs for "world".
    assert "{\\kf50}hello" in dialogue[0]
    assert "{\\k10}" in dialogue[0]
    assert "{\\kf60}world" in dialogue[0]
    # Line runs from first word start to last word end + 0.3 s tail.
    assert "Dialogue: 0,0:00:01.00,0:00:02.50,Default" in dialogue[0]


def test_build_ass_skips_degenerate_words():
    words = [_w("", 0.0, 1.0), _w("ok", 1.0, 2.0), _w("bad", 3.0, 2.0)]
    doc = build_ass(words)
    dialogue = [ln for ln in doc.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogue) == 1
    assert "ok" in dialogue[0]
    assert "bad" not in dialogue[0]


def test_build_ass_empty_words_is_header_only():
    doc = build_ass([])
    assert "[Events]" in doc
    assert "Dialogue:" not in doc

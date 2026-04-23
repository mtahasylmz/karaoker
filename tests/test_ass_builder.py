import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "worker"))

from ass_builder import build_ass, group_lines, _fmt_time, _sanitize


def test_fmt_time():
    assert _fmt_time(0) == "0:00:00.00"
    assert _fmt_time(1.5) == "0:00:01.50"
    assert _fmt_time(62.05) == "0:01:02.05"
    assert _fmt_time(3661.234) == "1:01:01.23"


def test_sanitize_strips_control_chars():
    assert _sanitize("hello") == "hello"
    assert _sanitize("{bad}") == "(bad)"
    assert _sanitize("a\\b") == "ab"
    assert _sanitize("line\nbreak") == "line break"


def test_group_lines_breaks_on_gap():
    words = [
        {"word": "a", "start": 0.0, "end": 0.5},
        {"word": "b", "start": 0.6, "end": 1.0},
        {"word": "c", "start": 3.0, "end": 3.5},  # 2s gap
    ]
    lines = group_lines(words, max_words=8, break_gap=1.5)
    assert len(lines) == 2
    assert [w["word"] for w in lines[0]] == ["a", "b"]
    assert [w["word"] for w in lines[1]] == ["c"]


def test_group_lines_respects_max():
    words = [{"word": str(i), "start": i * 0.3, "end": i * 0.3 + 0.2} for i in range(10)]
    lines = group_lines(words, max_words=4, break_gap=10)
    assert [len(l) for l in lines] == [4, 4, 2]


def test_build_ass_basic():
    words = [
        {"word": "Hello", "start": 1.0, "end": 1.5},
        {"word": "world", "start": 1.5, "end": 2.0},
    ]
    out = build_ass(words)
    assert "[Script Info]" in out
    assert "[V4+ Styles]" in out
    assert "[Events]" in out
    assert "Style: Default" in out
    assert "\\kf50}Hello" in out
    assert "\\kf50}world" in out
    assert "0:00:01.00" in out
    # Tail extends end past last word
    assert "0:00:02.30" in out


def test_build_ass_inserts_silent_gap():
    words = [
        {"word": "one", "start": 0.0, "end": 0.5},
        {"word": "two", "start": 1.0, "end": 1.5},  # 0.5s silent gap
    ]
    out = build_ass(words)
    assert "\\k50}" in out  # silent hold before "two"


def test_build_ass_skips_empty_words():
    words = [
        {"word": "", "start": 0.0, "end": 0.5},
        {"word": "ok", "start": 1.0, "end": 1.5},
    ]
    out = build_ass(words)
    assert "}ok" in out
    # No empty {\kf...} token
    assert "}{\\kf" not in out or " ok" in out


if __name__ == "__main__":
    test_fmt_time()
    test_sanitize_strips_control_chars()
    test_group_lines_breaks_on_gap()
    test_group_lines_respects_max()
    test_build_ass_basic()
    test_build_ass_inserts_silent_gap()
    test_build_ass_skips_empty_words()
    print("all pass")

from annemusic.lines import words_to_lines


def _w(text, start, end):
    return {"text": text, "start": start, "end": end}


# --------------------------------------------------------------------------- #
# words_to_lines — ASR-path line segmentation (Jam-ALT / karaoke conventions)
# --------------------------------------------------------------------------- #


def test_lines_break_after_sentence_punctuation():
    # A period ends a sung line even though only two short words precede it and
    # no gap/char-cap is hit; the next word opens a new line.
    lines = words_to_lines([
        _w("Hello", 0.0, 0.5),
        _w("world.", 0.6, 1.1),
        _w("Again", 1.2, 1.7),
    ])
    assert [ln["text"] for ln in lines] == ["Hello world", "Again"]


def test_lines_break_after_question_and_exclamation():
    lines = words_to_lines([
        _w("Why", 0.0, 0.4),
        _w("now?", 0.5, 0.9),
        _w("Stop!", 1.0, 1.4),
        _w("Go", 1.5, 1.9),
    ])
    assert [ln["text"] for ln in lines] == ["Why now?", "Stop!", "Go"]


def test_lines_break_after_unicode_sentence_punctuation():
    # Ellipsis and full-width CJK sentence marks all end a line.
    lines = words_to_lines([
        _w("Fading", 0.0, 0.4),
        _w("away…", 0.5, 0.9),
        _w("Owari", 1.0, 1.4),
        _w("da。", 1.5, 1.9),
        _w("Doushite", 2.0, 2.4),
        _w("na？", 2.5, 2.9),
        _w("End", 3.0, 3.4),
    ])
    assert [ln["text"] for ln in lines] == [
        "Fading away…", "Owari da。", "Doushite na？", "End"
    ]


def test_lines_break_after_clause_punctuation():
    # A comma is a clause break: the line ends there (and the comma is stripped
    # because it is line-final).
    lines = words_to_lines([
        _w("Wait", 0.0, 0.4),
        _w("here,", 0.5, 0.9),
        _w("I", 1.0, 1.4),
        _w("stay", 1.5, 1.9),
    ])
    assert [ln["text"] for ln in lines] == ["Wait here", "I stay"]


def test_lines_break_on_silence_gap_at_one_second():
    # A >=1 s gap between words breaks the line; 1.0 exactly triggers it.
    lines = words_to_lines([_w("aa", 0.0, 0.5), _w("bb", 1.5, 2.0)])
    assert [ln["text"] for ln in lines] == ["Aa", "Bb"]


def test_lines_do_not_break_on_subsecond_gap():
    lines = words_to_lines([_w("aa", 0.0, 0.5), _w("bb", 1.0, 1.5)])  # gap 0.5 s
    assert [ln["text"] for ln in lines] == ["Aa bb"]


def test_lines_never_break_purely_on_word_count():
    # Twelve short unpunctuated words with no gap and short text stay together:
    # there is no 8-word cap anymore.
    words = [_w(f"la", i * 0.1, i * 0.1 + 0.05) for i in range(12)]
    lines = words_to_lines(words)
    assert len(lines) == 1
    assert lines[0]["text"] == "La " + " ".join(["la"] * 11)


def test_lines_break_at_soft_char_cap_as_last_resort():
    # No punctuation, no gap: only the ~42-char soft cap forces a break.
    words = [_w("wwwwww", i * 0.1, i * 0.1 + 0.05) for i in range(12)]  # 6 chars each
    lines = words_to_lines(words)
    assert len(lines) >= 2
    assert all(len(ln["text"]) <= 48 for ln in lines)  # soft cap ~42 + slack


def test_line_final_comma_and_period_are_stripped():
    lines = words_to_lines([_w("Done.", 0.0, 0.5)])
    assert lines[0]["text"] == "Done"


def test_line_final_question_and_exclamation_are_kept():
    q = words_to_lines([_w("Who?", 0.0, 0.5)])
    e = words_to_lines([_w("Go!", 0.0, 0.5)])
    assert q[0]["text"] == "Who?"
    assert e[0]["text"] == "Go!"


def test_word_internal_period_is_kept():
    # "Mr." mid-line abbreviation: the period is not line-final, keep it.
    lines = words_to_lines([_w("Mr.", 0.0, 0.4), _w("Jones", 0.5, 0.9)])
    assert lines[0]["text"] == "Mr. Jones"


def test_abbreviation_does_not_swallow_a_real_sentence_end():
    # A lowercase multi-letter token ending in '.' is a real sentence break,
    # not an abbreviation.
    lines = words_to_lines([
        _w("stay.", 0.0, 0.4), _w("Then", 0.5, 0.9), _w("go", 1.0, 1.4),
    ])
    assert [ln["text"] for ln in lines] == ["Stay", "Then go"]


def test_first_letter_of_each_line_is_capitalized():
    lines = words_to_lines([_w("hello", 0.0, 0.5), _w("world.", 0.6, 1.1),
                            _w("bye", 1.2, 1.7)])
    assert lines[0]["text"] == "Hello world"
    assert lines[1]["text"] == "Bye"


def test_capitalization_does_not_lowercase_the_rest():
    # Proper-noun / all-caps casing after the first char must survive.
    lines = words_to_lines([_w("iPhone", 0.0, 0.5)])
    assert lines[0]["text"] == "IPhone"


def test_leading_digit_line_is_left_as_is():
    lines = words_to_lines([_w("99", 0.0, 0.5), _w("problems", 0.6, 1.1)])
    assert lines[0]["text"] == "99 problems"


def test_words_are_conserved_up_to_strip_and_capitalize():
    words = [
        _w("first", 0.0, 0.5), _w("line.", 0.6, 1.1),
        _w("second", 1.2, 1.7), _w("one", 1.8, 2.3),
    ]
    lines = words_to_lines(words)
    joined = " ".join(ln["text"] for ln in lines).lower()
    original = " ".join(w["text"] for w in words).rstrip(".").lower()
    # every source word (minus stripped line-final punctuation) survives in order
    assert joined.replace(".", "") == original.replace(".", "")


def test_line_spans_its_member_words():
    lines = words_to_lines([
        _w("a", 1.0, 1.4), _w("b.", 1.5, 2.0),
        _w("c", 3.5, 4.0),
    ])
    assert lines[0]["start"] == 1.0 and lines[0]["end"] == 2.0
    assert lines[1]["start"] == 3.5 and lines[1]["end"] == 4.0


def test_empty_words_yield_no_lines():
    assert words_to_lines([]) == []

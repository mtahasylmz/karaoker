"""Group aligned ASR words into display lines for the ASS subtitles.

``words_to_lines``: aligned ASR words -> display lines, broken at sentence
punctuation, then clause punctuation, then a >= 1 s sung pause, then a
~42-char soft cap — never by word count (Jam-ALT / karaoke conventions).
Line-final commas/periods are stripped and each line is capitalized.

Split out of core.py so the display-segmentation rules don't ride along with
the low-level pipeline primitives. The LRC path skips this entirely (human
lyrics are already punctuated and cased); its even token-spread lives in
core.even_words. ``clean_words`` stays the one shared primitive in repair;
this module depends on it one-directionally.
"""

from __future__ import annotations

from annemusic.repair import clean_words


# Line segmentation for the ASR path (Jam-ALT / karaoke conventions). The LRC
# path never uses this — human lyrics are already punctuated and cased.
LINE_BREAK_GAP_S = 1.0     # a >= 1 s sung pause ends a line
LINE_SOFT_CHARS = 42       # last-resort soft cap before a line grows too long
_SENTENCE_END = ".?!…。？！"  # break AFTER; '.' is line-stripped, '?!…' kept
_CLAUSE_END = ",;:—，；、"    # break AFTER; ',' is line-stripped


# Common title/abbreviation tokens whose trailing period is not a sentence end
# — don't break the line after them (keeps "Mr. Jones" on one line). A closed
# set, not a pattern: a period after a real word like "Go." IS a sentence end.
_ABBREVS = frozenset(
    {"mr.", "mrs.", "ms.", "dr.", "st.", "jr.", "sr.", "prof.", "no.", "vs."}
)


def _line_break_after(word: dict, nxt: dict, chars_so_far: int) -> bool:
    """Should a line end after ``word`` (with ``nxt`` following)? Priority
    ladder, first hit wins: sentence punctuation, clause punctuation, a >= 1 s
    silence gap, then the ~42-char soft cap as a last resort. Never breaks on
    word count."""
    token = word["text"].rstrip()
    tail = token[-1:]
    if tail in _SENTENCE_END and not (tail == "." and token.lower() in _ABBREVS):
        return True
    if tail in _CLAUSE_END:
        return True
    if nxt["start"] - word["end"] >= LINE_BREAK_GAP_S:
        return True
    return chars_so_far + 1 + len(nxt["text"]) > LINE_SOFT_CHARS


def _finalize_line(group: list[dict]) -> dict:
    """Collapse a group of words into one display line: join, strip a line-final
    comma/period run (keep ? ! and word-internal periods), capitalize the first
    letter without lowercasing the rest."""
    text = " ".join(w["text"] for w in group).rstrip(".,")
    if text[:1].isalpha():
        text = text[:1].upper() + text[1:]
    return {"text": text, "start": group[0]["start"], "end": group[-1]["end"]}


def _group_words(cleaned: list[dict]) -> list[list[dict]]:
    """Walk the words once, cutting a new group after each word that
    ``_line_break_after`` marks (running a char count so the soft cap sees the
    line-so-far). Every word lands in exactly one group; the last word never
    breaks (there is no successor to open a new line)."""
    groups: list[list[dict]] = []
    current: list[dict] = []
    chars = -1  # first word adds len+1, leaving chars == the joined length
    for w, nxt in zip(cleaned, cleaned[1:] + [None]):
        current.append(w)
        chars += len(w["text"]) + 1
        if nxt is not None and _line_break_after(w, nxt, chars):
            groups.append(current)
            current, chars = [], -1
    if current:
        groups.append(current)
    return groups


def words_to_lines(words: list[dict]) -> list[dict]:
    """Group aligned words into display lines for the ASR path, breaking at
    sentence punctuation, then clause punctuation, then a >= 1 s silence gap,
    then a ~42-char soft cap — never by word count. Line-final commas/periods
    are stripped and each line is capitalized; word-level manifest content is
    unchanged. The LRC path already has human lines and skips this."""
    return [_finalize_line(g) for g in _group_words(clean_words(words))]

"""Build an ASS subtitle file with karaoke-fill (\\kf) tags from word timings.

ASS karaoke conventions (per libass):
    SecondaryColour = text color BEFORE the sweep passes (unsung)
    PrimaryColour   = text color AFTER  the sweep passes (sung / highlight)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence, TypedDict


class Word(TypedDict):
    word: str
    start: float
    end: float


@dataclass
class Style:
    font: str = "Arial"
    font_size: int = 72
    res_x: int = 1920
    res_y: int = 1080
    # &HAABBGGRR — alpha-blue-green-red, 00 alpha = opaque
    primary_colour: str = "&H0000FFFF"   # yellow — sung
    secondary_colour: str = "&H00FFFFFF"  # white  — unsung
    outline_colour: str = "&H00000000"   # black outline
    back_colour: str = "&H80000000"      # semi-transparent shadow
    outline: int = 3
    shadow: int = 2
    alignment: int = 2  # bottom-center
    margin_v: int = 80
    max_words_per_line: int = 8
    break_gap: float = 1.5
    tail: float = 0.3  # keep line on screen this long after the last word


def _fmt_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _sanitize(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def group_lines(words: Sequence[Word], max_words: int, break_gap: float) -> list[list[Word]]:
    lines: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        if current:
            gap = w["start"] - current[-1]["end"]
            if gap > break_gap or len(current) >= max_words:
                lines.append(current)
                current = []
        current.append(w)
    if current:
        lines.append(current)
    return lines


def _render_line(line: list[Word], tail: float) -> str:
    if not line:
        return ""
    t0 = line[0]["start"]
    t1 = line[-1]["end"] + tail
    parts: list[str] = []
    prev_end = t0
    for w in line:
        gap_cs = max(0, int(round((w["start"] - prev_end) * 100)))
        if gap_cs > 0:
            parts.append(f"{{\\k{gap_cs}}}")  # silent hold
        dur_cs = max(1, int(round((w["end"] - w["start"]) * 100)))
        token = _sanitize(w["word"]).strip()
        if not token:
            continue
        parts.append(f"{{\\kf{dur_cs}}}{token} ")
        prev_end = w["end"]
    text = "".join(parts).rstrip()
    return f"Dialogue: 0,{_fmt_time(t0)},{_fmt_time(t1)},Default,,0,0,0,,{text}"


def build_ass(words: Iterable[Word], style: Style | None = None) -> str:
    s = style or Style()
    words_list = [w for w in words if w.get("word") and w["end"] > w["start"]]
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {s.res_x}\n"
        f"PlayResY: {s.res_y}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{s.font},{s.font_size},{s.primary_colour},{s.secondary_colour},"
        f"{s.outline_colour},{s.back_colour},-1,0,0,0,100,100,0,0,1,{s.outline},{s.shadow},"
        f"{s.alignment},60,60,{s.margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = group_lines(words_list, s.max_words_per_line, s.break_gap)
    events = "\n".join(_render_line(ln, s.tail) for ln in lines if ln) + "\n"
    return header + events


def write_ass(words: Iterable[Word], path: str | Path, style: Style | None = None) -> Path:
    path = Path(path)
    path.write_text(build_ass(words, style), encoding="utf-8")
    return path

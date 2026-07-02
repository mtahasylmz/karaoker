"""Build an ASS subtitle file with \\kf karaoke-fill tags from word timings.

Ported from stages/compose/src/ass.ts. Conventions:
  PrimaryColour   = sung / highlight colour (after the \\kf sweep passes)
  SecondaryColour = unsung / base colour (before the sweep)
ASS colour format is &HAABBGGRR.
"""

from __future__ import annotations

DEFAULT_STYLE = {
    "font": "Arial",
    "font_size": 72,
    "res_x": 1920,
    "res_y": 1080,
    "primary_colour": "&H0000FFFF",
    "secondary_colour": "&H00FFFFFF",
    "outline_colour": "&H00000000",
    "back_colour": "&H80000000",
    "outline": 3,
    "shadow": 2,
    "alignment": 2,          # ASS numpad; 2 = bottom-center
    "margin_v": 80,
    "max_words_per_line": 8,
    "break_gap": 1.5,        # seconds of silence that force a line break
    "tail": 0.3,             # seconds a line lingers after its last word
}


def fmt_time(t: float) -> str:
    """h:mm:ss.cc (centiseconds), clamped at zero."""
    cs = max(0, round(t * 100))
    h, m = divmod(cs, 360_000)
    m, s = divmod(m, 6_000)
    s, rem = divmod(s, 100)
    return f"{h}:{m:02d}:{s:02d}.{rem:02d}"


def sanitize(text: str) -> str:
    return (
        text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")
    )


def group_lines(words: list[dict], max_words: int, break_gap: float) -> list[list[dict]]:
    lines: list[list[dict]] = []
    current: list[dict] = []
    for w in words:
        if current and (
            w["start"] - current[-1]["end"] > break_gap or len(current) >= max_words
        ):
            lines.append(current)
            current = []
        current.append(w)
    if current:
        lines.append(current)
    return lines


def _render_line(line: list[dict], tail: float) -> str:
    if not line:
        return ""
    t0 = line[0]["start"]
    t1 = line[-1]["end"] + tail
    parts: list[str] = []
    prev_end = t0
    for w in line:
        gap_cs = max(0, round((w["start"] - prev_end) * 100))
        if gap_cs > 0:
            parts.append(f"{{\\k{gap_cs}}}")
        dur_cs = max(1, round((w["end"] - w["start"]) * 100))
        tok = sanitize(w["text"]).strip()
        if not tok:
            continue
        parts.append(f"{{\\kf{dur_cs}}}{tok} ")
        prev_end = w["end"]
    text = "".join(parts).rstrip()
    return f"Dialogue: 0,{fmt_time(t0)},{fmt_time(t1)},Default,,0,0,0,,{text}"


def _header(s: dict) -> str:
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {s['res_x']}\n"
        f"PlayResY: {s['res_y']}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{s['font']},{s['font_size']},{s['primary_colour']},"
        f"{s['secondary_colour']},{s['outline_colour']},{s['back_colour']},"
        f"-1,0,0,0,100,100,0,0,1,{s['outline']},{s['shadow']},"
        f"{s['alignment']},60,60,{s['margin_v']},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )


def _events(lines: list[list[dict]], tail: float) -> list[str]:
    return [ev for ev in (_render_line(ln, tail) for ln in lines) if ev]


def build_ass(words: list[dict], style: dict | None = None) -> str:
    s = {**DEFAULT_STYLE, **(style or {})}
    cleaned = [w for w in words if w["text"] and w["end"] > w["start"]]
    lines = group_lines(cleaned, s["max_words_per_line"], s["break_gap"])
    events = _events(lines, s["tail"])
    return _header(s) + ("\n".join(events) + "\n" if events else "")

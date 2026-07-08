"""Build an ASS subtitle file: one styled line shown during its window.

Line-level, no per-word colour-fill animation (even-distribution timing
made the \\kf sweep fake precision — dropped). Takes pre-grouped lines
[{text, start, end}]; word-level timing lives in manifest.json instead.
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
    "tail": 0.3,             # seconds a line lingers after its window
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


def _render_line(line: dict, tail: float) -> str:
    text = sanitize(line.get("text") or "").strip()
    if not text or line["end"] <= line["start"]:
        return ""
    t0 = fmt_time(line["start"])
    t1 = fmt_time(line["end"] + tail)
    return f"Dialogue: 0,{t0},{t1},Default,,0,0,0,,{text}"


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


def build_ass(lines: list[dict], style: dict | None = None) -> str:
    """lines: [{text, start, end}] — one Dialogue event each."""
    s = {**DEFAULT_STYLE, **(style or {})}
    events = [ev for ev in (_render_line(ln, s["tail"]) for ln in lines) if ev]
    return _header(s) + ("\n".join(events) + "\n" if events else "")

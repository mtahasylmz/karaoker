/** Build an ASS subtitle file with \kf karaoke-fill tags from word timings.
 *
 * Ported from MVP worker/ass_builder.py. Same conventions:
 *   PrimaryColour   = sung / highlight colour (after the \kf sweep passes)
 *   SecondaryColour = unsung / base colour (before the sweep)
 *
 * ASS color format is &HAABBGGRR (alpha-blue-green-red).
 */

import type { Word } from "@annemusic/contracts";

export type AssStyle = {
  font: string;
  font_size: number;
  res_x: number;
  res_y: number;
  primary_colour: string;   // sung
  secondary_colour: string; // unsung
  outline_colour: string;
  back_colour: string;
  outline: number;
  shadow: number;
  alignment: number;        // ASS numpad; 2 = bottom-center
  margin_v: number;
  max_words_per_line: number;
  break_gap: number;        // seconds of silence that force a line break
  tail: number;             // seconds a line lingers after its last word
};

export const DEFAULT_STYLE: AssStyle = {
  font: "Arial",
  font_size: 72,
  res_x: 1920,
  res_y: 1080,
  primary_colour: "&H0000FFFF",
  secondary_colour: "&H00FFFFFF",
  outline_colour: "&H00000000",
  back_colour: "&H80000000",
  outline: 3,
  shadow: 2,
  alignment: 2,
  margin_v: 80,
  max_words_per_line: 8,
  break_gap: 1.5,
  tail: 0.3,
};

function fmtTime(t: number): string {
  const cs = Math.max(0, Math.round(t * 100));
  const h = Math.floor(cs / 360_000);
  const m = Math.floor((cs % 360_000) / 6_000);
  const s = Math.floor((cs % 6_000) / 100);
  const rem = cs % 100;
  return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(rem).padStart(2, "0")}`;
}

function sanitize(text: string): string {
  return text.replace(/\\/g, "").replace(/\{/g, "(").replace(/\}/g, ")").replace(/\n/g, " ");
}

export function groupLines(words: Word[], maxWords: number, breakGap: number): Word[][] {
  const lines: Word[][] = [];
  let current: Word[] = [];
  for (const w of words) {
    if (current.length > 0) {
      const prev = current[current.length - 1]!;
      if (w.start - prev.end > breakGap || current.length >= maxWords) {
        lines.push(current);
        current = [];
      }
    }
    current.push(w);
  }
  if (current.length > 0) lines.push(current);
  return lines;
}

function renderLine(line: Word[], tail: number): string {
  if (line.length === 0) return "";
  const t0 = line[0]!.start;
  const t1 = line[line.length - 1]!.end + tail;
  const parts: string[] = [];
  let prevEnd = t0;
  for (const w of line) {
    const gapCs = Math.max(0, Math.round((w.start - prevEnd) * 100));
    if (gapCs > 0) parts.push(`{\\k${gapCs}}`);
    const durCs = Math.max(1, Math.round((w.end - w.start) * 100));
    const tok = sanitize(w.text).trim();
    if (!tok) continue;
    parts.push(`{\\kf${durCs}}${tok} `);
    prevEnd = w.end;
  }
  const text = parts.join("").trimEnd();
  return `Dialogue: 0,${fmtTime(t0)},${fmtTime(t1)},Default,,0,0,0,,${text}`;
}

export function buildAss(words: Word[], styleOverrides: Partial<AssStyle> = {}): string {
  const s: AssStyle = { ...DEFAULT_STYLE, ...styleOverrides };
  const cleaned = words.filter((w) => w.text && w.end > w.start);
  const header =
    "[Script Info]\n" +
    "ScriptType: v4.00+\n" +
    `PlayResX: ${s.res_x}\n` +
    `PlayResY: ${s.res_y}\n` +
    "WrapStyle: 2\n" +
    "ScaledBorderAndShadow: yes\n\n" +
    "[V4+ Styles]\n" +
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, " +
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, " +
    "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, " +
    "Alignment, MarginL, MarginR, MarginV, Encoding\n" +
    `Style: Default,${s.font},${s.font_size},${s.primary_colour},${s.secondary_colour},` +
    `${s.outline_colour},${s.back_colour},-1,0,0,0,100,100,0,0,1,${s.outline},${s.shadow},` +
    `${s.alignment},60,60,${s.margin_v},1\n\n` +
    "[Events]\n" +
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";
  const lines = groupLines(cleaned, s.max_words_per_line, s.break_gap);
  const events = lines.map((ln) => renderLine(ln, s.tail)).filter(Boolean).join("\n") + "\n";
  return header + events;
}

export function lineDuration(words: Word[]): number {
  if (words.length === 0) return 0;
  return words[words.length - 1]!.end - words[0]!.start;
}

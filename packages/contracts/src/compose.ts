import { z } from "zod";
import { GcsUri, MediaUri, StageJobId, StageResult } from "./primitives.js";
import { Word } from "./align.js";
import { VocalActivityRegion } from "./transcribe.js";

// ASS color literal, e.g. "&H00FFFFFF" or "&H0000FFFF". Format is &HAABBGGRR.
const AssColor = z
  .string()
  .regex(/^&H[0-9A-Fa-f]{6,8}$/, "expected ASS &HBBGGRR or &HAABBGGRR color");

export const AssStyle = z
  .object({
    font: z.string(),
    font_size: z.number().int().positive(),
    // ASS [Script Info] PlayResX/PlayResY — libass / JASSUB render coords.
    // Defaults (1920×1080) live on the stage side; override per-request if
    // the source video reports something else.
    res_x: z.number().int().positive(),
    res_y: z.number().int().positive(),
    primary_colour: AssColor, // sung (fills)
    secondary_colour: AssColor, // unsung (base)
    outline_colour: AssColor,
    back_colour: AssColor,
    outline: z.number().int().min(0).max(10),
    shadow: z.number().int().min(0).max(10),
    alignment: z.number().int().min(1).max(9), // ASS numpad: 2 = bottom-center
    margin_v: z.number().int().nonnegative(),
    max_words_per_line: z.number().int().positive(),
    break_gap: z.number().positive(), // seconds of silence that force a new line
    tail: z.number().nonnegative(), // seconds a line lingers after its last word
  })
  .partial(); // every field optional at request time; compose applies defaults
export type AssStyle = z.infer<typeof AssStyle>;

// A line's worth of per-word timings, same grouping that drives the ASS
// events (see stages/compose buildLines / groupLines). The browser's DOM
// lyric overlay renders directly off this tree — no client-side ASS parsing.
export const LyricLine = z.object({
  start: z.number().nonnegative(),
  end: z.number().nonnegative(),
  words: z.array(Word),
});
export type LyricLine = z.infer<typeof LyricLine>;

export const ComposeRequest = z.object({
  job_id: StageJobId,
  words: z.array(Word),
  video_uri: GcsUri,
  instrumental_uri: GcsUri,
  language: z.string().optional(),
  style: AssStyle.optional(),
  // Forwarded from transcribe via align. Surfaces to the browser so the DOM
  // overlay can render "instrumental break" UI (or libass can swap styles)
  // instead of leaving a stale highlight on screen during silent sections.
  vocal_activity: z.array(VocalActivityRegion),
});
export type ComposeRequest = z.infer<typeof ComposeRequest>;

// The JSON the browser consumes. Points at the media URLs + carries the
// lyric tree + VAD regions so a DOM renderer has everything it needs
// without parsing ASS. ass_url remains the fallback/libass/burn-in path.
export const PlaybackManifest = z.object({
  job_id: StageJobId,
  video_url: MediaUri,
  instrumental_url: MediaUri,
  ass_url: MediaUri,
  language: z.string().optional(),
  duration: z.number().positive().optional(),
  created_at: z.number().int(),
  lines: z.array(LyricLine),
  vocal_activity: z.array(VocalActivityRegion),
});
export type PlaybackManifest = z.infer<typeof PlaybackManifest>;

export const ComposeResponse = StageResult.extend({
  stage: z.literal("compose"),
  manifest_uri: GcsUri,
  manifest_url: MediaUri,
  ass_uri: GcsUri,
});
export type ComposeResponse = z.infer<typeof ComposeResponse>;

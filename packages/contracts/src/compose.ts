import { z } from "zod";
import { GcsUri, HttpsUri, StageJobId, StageResult } from "./primitives.js";
import { Word } from "./align.js";

// ASS color literal, e.g. "&H00FFFFFF" or "&H0000FFFF". Format is &HAABBGGRR.
const AssColor = z
  .string()
  .regex(/^&H[0-9A-Fa-f]{6,8}$/, "expected ASS &HBBGGRR or &HAABBGGRR color");

export const AssStyle = z
  .object({
    font: z.string(),
    font_size: z.number().int().positive(),
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

export const ComposeRequest = z.object({
  job_id: StageJobId,
  words: z.array(Word),
  video_uri: GcsUri,
  instrumental_uri: GcsUri,
  language: z.string().optional(),
  style: AssStyle.optional(),
});
export type ComposeRequest = z.infer<typeof ComposeRequest>;

// The JSON the browser consumes. It points at three GCS public URLs that
// render together (video with original audio muted + instrumental audio +
// ASS overlay via JASSUB). No video re-encode; no burning.
export const PlaybackManifest = z.object({
  job_id: StageJobId,
  video_url: HttpsUri,
  instrumental_url: HttpsUri,
  ass_url: HttpsUri,
  language: z.string().optional(),
  duration: z.number().positive().optional(),
  created_at: z.number().int(),
});
export type PlaybackManifest = z.infer<typeof PlaybackManifest>;

export const ComposeResponse = StageResult.extend({
  stage: z.literal("compose"),
  manifest_uri: GcsUri,
  manifest_url: HttpsUri,
  ass_uri: GcsUri,
});
export type ComposeResponse = z.infer<typeof ComposeResponse>;

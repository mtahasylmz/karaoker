import { z } from "zod";
import { GcsUri, StageJobId, StageResult } from "./primitives.js";

// "off": straight amix of recording + instrumental, no pitch DSP.
// "smooth": gentle RubberBand pitch smoothing (no scale detection).
// "snap": scale-detecting snap-to-nearest-note. Not implemented in v1.
export const AutotuneMode = z.enum(["off", "smooth", "snap"]);
export type AutotuneMode = z.infer<typeof AutotuneMode>;

// Tunable mix knobs. All optional at request time so the client can send
// only what the user has changed; the stage applies defaults for the rest.
// Ranges are clamped so a runaway slider can't destroy the output bus.
export const MixParams = z
  .object({
    vocal_gain_db: z.number().min(-12).max(12),
    instrumental_gain_db: z.number().min(-12).max(12),
    reverb_wet: z.number().min(0).max(1),
    duck_db: z.number().min(0).max(12),
    presence_db: z.number().min(-3).max(6),
  })
  .partial();
export type MixParams = z.infer<typeof MixParams>;

export const RecordMixRequest = z.object({
  job_id: StageJobId,
  recording_uri: GcsUri,
  instrumental_uri: GcsUri,
  // Separated vocals stem from stages/separate. Used as the GCC-PHAT sync
  // reference when present (more robust than correlating vs instrumental).
  vocals_uri: GcsUri.optional(),
  autotune: AutotuneMode.default("off"),
  // Run Demucs on the user recording to strip speaker bleed before mixing.
  // Set false when the user confirms headphones were used.
  clean_bleed: z.boolean().default(true),
  // Master output trim applied post-mix, pre-limiter.
  gain_db: z.number().min(-24).max(24).default(0),
  mix: MixParams.optional(),
});
export type RecordMixRequest = z.infer<typeof RecordMixRequest>;

export const RecordMixResponse = StageResult.extend({
  stage: z.literal("record-mix"),
  mix_uri: GcsUri,
});
export type RecordMixResponse = z.infer<typeof RecordMixResponse>;

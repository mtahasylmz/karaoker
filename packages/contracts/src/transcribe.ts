import { z } from "zod";
import { GcsUri, StageJobId, StageResult } from "./primitives.js";

export const Segment = z.object({
  text: z.string(),
  start: z.number().nonnegative(),
  end: z.number().nonnegative(),
});
export type Segment = z.infer<typeof Segment>;

export const TranscribeRequest = z.object({
  job_id: StageJobId,
  vocals_uri: GcsUri,
  // ISO 639-1 language code; if omitted, stage detects.
  language: z.string().length(2).optional(),
  known_lyrics: z.string().optional(),
  // For LRCLIB lookup. If title+artist resolve, stage skips Whisper entirely.
  title: z.string().optional(),
  artist: z.string().optional(),
});
export type TranscribeRequest = z.infer<typeof TranscribeRequest>;

export const TranscribeSource = z.enum(["whisper", "lrclib"]);
export type TranscribeSource = z.infer<typeof TranscribeSource>;

export const TranscribeResponse = StageResult.extend({
  stage: z.literal("transcribe"),
  language: z.string(),
  segments: z.array(Segment),
  source: TranscribeSource,
  model_used: z.string(),
});
export type TranscribeResponse = z.infer<typeof TranscribeResponse>;

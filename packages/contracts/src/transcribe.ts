import { z } from "zod";
import { GcsUri, StageJobId, StageResult } from "./primitives.js";

export const Segment = z.object({
  text: z.string(),
  start: z.number().nonnegative(),
  end: z.number().nonnegative(),
});
export type Segment = z.infer<typeof Segment>;

// Ordered, non-overlapping regions marking whether the vocals stem contains
// singing or is effectively silent (instrumental-only). Karaoke consumers use
// this to render "instrumental break" UI instead of leaving a stale word
// highlight frozen on screen.
export const VocalActivityKind = z.enum(["vocals", "instrumental"]);
export type VocalActivityKind = z.infer<typeof VocalActivityKind>;

export const VocalActivityRegion = z.object({
  start: z.number().nonnegative(),
  end: z.number().nonnegative(),
  kind: VocalActivityKind,
});
export type VocalActivityRegion = z.infer<typeof VocalActivityRegion>;

export const TranscribeRequest = z.object({
  job_id: StageJobId,
  vocals_uri: GcsUri,
  // Original upload (video/audio). Qwen3-ASR was trained with BGM and runs
  // on the full mix; faster-whisper reads the vocals stem. Optional so
  // whisper-only callers can omit.
  source_uri: GcsUri.optional(),
  // ISO 639-1 (2-letter) or ISO 639-3 (3-letter) language code; if omitted,
  // stage detects. flows.ts routes `yue` (Cantonese) and `fil` (Filipino),
  // so 3-letter codes have to pass validation.
  language: z.string().min(2).max(3).optional(),
  // Optional prompt bias for the ASR decoder (first 200 chars used). Helps
  // on rare-vocabulary lyrics; not a transcription replacement.
  known_lyrics: z.string().optional(),
});
export type TranscribeRequest = z.infer<typeof TranscribeRequest>;

// "qwen3" is forward-declared for packages/contracts/flows.ts routing; no
// transcribe backend emits it yet. Until a Qwen3-ASR path lands, expect
// only "whisper" in real responses.
export const TranscribeSource = z.enum(["qwen3", "whisper"]);
export type TranscribeSource = z.infer<typeof TranscribeSource>;

export const TranscribeResponse = StageResult.extend({
  stage: z.literal("transcribe"),
  language: z.string(),
  segments: z.array(Segment),
  // Audio-derived (RMS envelope on the vocals stem). May not cover the full
  // song duration — callers should treat unreported time as "unknown".
  vocal_activity: z.array(VocalActivityRegion),
  source: TranscribeSource,
  model_used: z.string(),
});
export type TranscribeResponse = z.infer<typeof TranscribeResponse>;

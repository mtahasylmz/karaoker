import { z } from "zod";
import { GcsUri, StageJobId, StageResult } from "./primitives.js";
import { Segment, VocalActivityRegion } from "./transcribe.js";

export const Word = z.object({
  text: z.string(),
  start: z.number().nonnegative(),
  end: z.number().nonnegative(),
  score: z.number().min(0).max(1).optional(),
});
export type Word = z.infer<typeof Word>;

// "qwen3" is forward-declared for packages/contracts/flows.ts routing; no
// align backend emits it yet. Until a Qwen3-ForcedAligner path lands,
// expect only "whisperx" | "even-split" in real responses.
export const AlignSource = z.enum(["qwen3", "whisperx", "even-split"]);
export type AlignSource = z.infer<typeof AlignSource>;

export const AlignRequest = z.object({
  job_id: StageJobId,
  vocals_uri: GcsUri,
  segments: z.array(Segment),
  language: z.string(),
  // Forwarded from transcribe (which always produces it). Align does not
  // consume this — wav2vec2 only needs audio + text — it passes the array
  // through verbatim so compose can render instrumental-break UI without
  // a second round-trip to transcribe. Required, matching transcribe's
  // guarantee.
  vocal_activity: z.array(VocalActivityRegion),
});
export type AlignRequest = z.infer<typeof AlignRequest>;

export const AlignResponse = StageResult.extend({
  stage: z.literal("align"),
  words: z.array(Word),
  // Pass-through from the request. Kept required so compose can trust it.
  vocal_activity: z.array(VocalActivityRegion),
  source: AlignSource,
  model_used: z.string(),
});
export type AlignResponse = z.infer<typeof AlignResponse>;

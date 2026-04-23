import { z } from "zod";
import { GcsUri, StageJobId, StageResult } from "./primitives.js";
import { Segment } from "./transcribe.js";

export const Word = z.object({
  text: z.string(),
  start: z.number().nonnegative(),
  end: z.number().nonnegative(),
  score: z.number().min(0).max(1).optional(),
});
export type Word = z.infer<typeof Word>;

export const AlignRequest = z.object({
  job_id: StageJobId,
  vocals_uri: GcsUri,
  segments: z.array(Segment),
  language: z.string(),
});
export type AlignRequest = z.infer<typeof AlignRequest>;

export const AlignResponse = StageResult.extend({
  stage: z.literal("align"),
  words: z.array(Word),
});
export type AlignResponse = z.infer<typeof AlignResponse>;

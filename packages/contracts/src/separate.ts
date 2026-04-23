import { z } from "zod";
import { GcsUri, StageJobId, StageResult } from "./primitives.js";

export const SeparateModel = z.enum([
  "bs-roformer",
  "mel-roformer",
  "htdemucs-ft",
  "htdemucs",
]);
export type SeparateModel = z.infer<typeof SeparateModel>;

export const SeparateRequest = z.object({
  job_id: StageJobId,
  source_uri: GcsUri,
  model: SeparateModel.optional(),
});
export type SeparateRequest = z.infer<typeof SeparateRequest>;

export const SeparateResponse = StageResult.extend({
  stage: z.literal("separate"),
  vocals_uri: GcsUri,
  instrumental_uri: GcsUri,
  sample_rate: z.number().int().positive(),
  model_used: SeparateModel,
});
export type SeparateResponse = z.infer<typeof SeparateResponse>;

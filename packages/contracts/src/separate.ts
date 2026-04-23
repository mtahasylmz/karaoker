import { z } from "zod";
import { GcsUri, StageJobId, StageResult } from "./primitives.js";

// Model slugs are the filenames the separate stage routes on (see
// stages/separate/src/separate/pipeline.py AS_MODEL_FILES / DEMUCS_MODELS).
// Keep in lock-step with the pipeline — adding a model is a one-line edit
// here + a mapping row there.
export const SeparateModel = z.enum([
  "mel_band_roformer_kim",
  "bs_roformer_ep317",
  "htdemucs",
  "htdemucs_ft",
  "htdemucs_6s",
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

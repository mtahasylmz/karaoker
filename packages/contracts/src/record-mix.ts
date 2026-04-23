import { z } from "zod";
import { GcsUri, StageJobId, StageResult } from "./primitives.js";

// "off": just mix the raw recording with the instrumental.
// "smooth": gentle pitch smoothing via RubberBand (no scale detection).
// "snap": scale-detecting snap-to-nearest-note. Deliberately out of scope v1.
export const AutotuneMode = z.enum(["off", "smooth", "snap"]);
export type AutotuneMode = z.infer<typeof AutotuneMode>;

export const RecordMixRequest = z.object({
  job_id: StageJobId,
  recording_uri: GcsUri,
  instrumental_uri: GcsUri,
  autotune: AutotuneMode.default("off"),
  gain_db: z.number().min(-24).max(24).default(0),
});
export type RecordMixRequest = z.infer<typeof RecordMixRequest>;

export const RecordMixResponse = StageResult.extend({
  stage: z.literal("record-mix"),
  mix_uri: GcsUri,
});
export type RecordMixResponse = z.infer<typeof RecordMixResponse>;

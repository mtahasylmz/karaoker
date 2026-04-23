import { z } from "zod";

export const StageName = z.enum([
  "separate",
  "transcribe",
  "align",
  "compose",
  "record-mix",
]);
export type StageName = z.infer<typeof StageName>;

// Job IDs are hex — 12 chars for MVP-style short IDs, up to 32 for UUIDs/sha256 prefixes.
export const StageJobId = z
  .string()
  .regex(/^[a-f0-9]{12,32}$/, "expected 12-32 hex chars");
export type StageJobId = z.infer<typeof StageJobId>;

export const GcsUri = z
  .string()
  .regex(/^gs:\/\/[a-z0-9_.-]+\/[^\0]+$/, "expected gs://bucket/path");
export type GcsUri = z.infer<typeof GcsUri>;

export const HttpsUri = z
  .string()
  .url()
  .refine((v) => v.startsWith("https://"), "must be https");
export type HttpsUri = z.infer<typeof HttpsUri>;

// Public URL for a media artifact the browser will fetch. Prod emits
// https://storage.googleapis.com/... ; dev (DEV_FS_ROOT) emits file:// so
// local manifests can round-trip through this contract without fake-https
// shims.
export const MediaUri = z
  .string()
  .refine(
    (v) => v.startsWith("https://") || v.startsWith("file://"),
    "expected https:// or file:// URL",
  );
export type MediaUri = z.infer<typeof MediaUri>;

// Every stage response carries timing + optional per-stage diagnostics so we
// can trace perf regressions without inventing new fields.
export const StageResult = z.object({
  job_id: StageJobId,
  stage: StageName,
  started_at: z.number().int().nonnegative(),
  finished_at: z.number().int().nonnegative(),
  duration_ms: z.number().int().nonnegative(),
  diagnostics: z.record(z.string(), z.unknown()).optional(),
});
export type StageResult = z.infer<typeof StageResult>;

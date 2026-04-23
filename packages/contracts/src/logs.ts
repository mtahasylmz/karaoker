import { z } from "zod";
import { StageJobId } from "./primitives.js";

export const LogLevel = z.enum(["debug", "info", "warn", "error"]);
export type LogLevel = z.infer<typeof LogLevel>;

// One structured event from a stage. Transport: Upstash Redis Stream
// `logs:{stage}`, also mirrored to stdout as JSON lines.
export const LogEntry = z.object({
  ts: z.number().int().nonnegative(), // ms since epoch
  stage: z.string(),
  job_id: StageJobId.optional(),
  level: LogLevel,
  msg: z.string(),
  data: z.record(z.string(), z.unknown()).optional(),
  err: z
    .object({
      name: z.string(),
      message: z.string(),
      stack: z.string().optional(),
    })
    .optional(),
});
export type LogEntry = z.infer<typeof LogEntry>;

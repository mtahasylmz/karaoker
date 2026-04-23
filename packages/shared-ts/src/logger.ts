/**
 * Structured logger that writes to:
 *   1. stdout (JSON lines — Cloud Run log viewer works unchanged)
 *   2. Upstash Redis Stream `logs:{stage}` (for the `pnpm logs` CLI)
 *
 * XADD is fire-and-forget. If Redis is unreachable, the stdout side still
 * records the event; logging must never crash the caller.
 */

import type { LogEntry, LogLevel } from "@annemusic/contracts";
import { redis } from "./redis.js";

const MAX_STREAM_LEN = 10_000;

type Fields = Record<string, unknown>;

function errObj(err: unknown): LogEntry["err"] {
  if (!err) return undefined;
  if (err instanceof Error) return { name: err.name, message: err.message, stack: err.stack };
  return { name: "NonError", message: String(err) };
}

function flatten(entry: LogEntry): Record<string, string> {
  // Upstash XADD takes a shallow string→string map; serialize nested fields.
  return {
    ts: String(entry.ts),
    stage: entry.stage,
    job_id: entry.job_id ?? "",
    level: entry.level,
    msg: entry.msg,
    data: entry.data ? JSON.stringify(entry.data) : "",
    err: entry.err ? JSON.stringify(entry.err) : "",
  };
}

async function publish(entry: LogEntry) {
  const key = `logs:${entry.stage}`;
  try {
    await redis().xadd(key, "*", flatten(entry), {
      trim: { type: "MAXLEN", threshold: MAX_STREAM_LEN, comparison: "~" },
    });
  } catch (e) {
    // Structured warning so the failure is itself searchable.
    console.error(JSON.stringify({
      ts: Date.now(),
      stage: entry.stage,
      level: "warn",
      msg: "log stream publish failed",
      err: errObj(e),
    }));
  }
}

export type Logger = {
  debug: (job_id: string | undefined, msg: string, data?: Fields) => void;
  info: (job_id: string | undefined, msg: string, data?: Fields) => void;
  warn: (job_id: string | undefined, msg: string, data?: Fields) => void;
  error: (job_id: string | undefined, msg: string, err: unknown, data?: Fields) => void;
};

export function createLogger(stage: string): Logger {
  const emit = (level: LogLevel, job_id: string | undefined, msg: string, data?: Fields, err?: unknown) => {
    const entry: LogEntry = {
      ts: Date.now(),
      stage,
      job_id,
      level,
      msg,
      data,
      err: err ? errObj(err) : undefined,
    };
    // stdout first — never blocks on the network.
    process.stdout.write(JSON.stringify(entry) + "\n");
    // Stream in the background.
    void publish(entry);
  };
  return {
    debug: (job_id, msg, data) => emit("debug", job_id, msg, data),
    info: (job_id, msg, data) => emit("info", job_id, msg, data),
    warn: (job_id, msg, data) => emit("warn", job_id, msg, data),
    error: (job_id, msg, err, data) => emit("error", job_id, msg, data, err),
  };
}

/** Await any in-flight XADDs. Call before process exit in short-lived scripts. */
export async function flushLogs(): Promise<void> {
  // XADDs in this module are all fire-and-forget Promises. We rely on the
  // pnpm/node runtime to await them; in short-lived scripts we also sleep
  // briefly to give the HTTP request time to land.
  await new Promise((r) => setTimeout(r, 200));
}

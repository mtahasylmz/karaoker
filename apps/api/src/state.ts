/** Redis state access for users/uploads/videos/jobs. Shape matches MVP. */

import { redis } from "@annemusic/shared-ts/redis";

const USERNAME_RE = /^[A-Za-z0-9_][A-Za-z0-9_.-]{1,23}$/;
const SHA256_RE = /^[0-9a-f]{64}$/;
const MAX_JOBS_PER_USER = 50;

export const validUsername = (u: string) => USERNAME_RE.test(u ?? "");
export const validSha256 = (s: string) => SHA256_RE.test(s ?? "");

const now = () => String(Date.now());

// ---------- users ----------

function randomHex(bytes: number): string {
  return Array.from(crypto.getRandomValues(new Uint8Array(bytes)))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

// The stored value IS the user's bearer token, returned exactly once at
// registration. Pre-token records hold "1" and can't authenticate — those
// users re-register (acceptable pre-launch; wipe.sh resets dev state).
export async function reserveUsername(username: string): Promise<string | null> {
  const token = randomHex(32);
  const r = await redis().set(`user:${username}`, token, { nx: true });
  return r ? token : null;
}

export async function userExists(username: string): Promise<boolean> {
  return (await redis().exists(`user:${username}`)) === 1;
}

export async function userTokenMatches(
  username: string,
  token: string | undefined,
): Promise<boolean> {
  if (!token) return false;
  const v = await redis().get(`user:${username}`);
  return typeof v === "string" && v.length >= 32 && v === token;
}

// ---------- uploads ----------

export type UploadRecord = {
  sha256: string;
  size: string;
  content_type: string;
  object_path: string;
  created_at: string;
};

export async function getUpload(sha256: string): Promise<UploadRecord | null> {
  const data = (await redis().hgetall(`upload:${sha256}`)) as UploadRecord | null;
  return data && Object.keys(data).length > 0 ? data : null;
}

export async function recordUpload(
  sha256: string,
  size: number,
  content_type: string,
  object_path: string,
): Promise<void> {
  await redis().hset(`upload:${sha256}`, {
    sha256,
    size: String(size),
    content_type,
    object_path,
    created_at: now(),
  });
}

// ---------- video dedup ----------

export async function getVideo(sha256: string): Promise<Record<string, string> | null> {
  const data = (await redis().hgetall(`video:${sha256}`)) as Record<string, string> | null;
  return data && Object.keys(data).length > 0 ? data : null;
}

export async function claimVideo(sha256: string, job_id: string): Promise<boolean> {
  const key = `video:${sha256}`;
  const won = (await redis().hsetnx(key, "status", "queued")) === 1;
  if (won) {
    await redis().hset(key, { job_id, created_at: now() });
    return true;
  }
  // A failed run doesn't block the sha forever: flip failed → queued for the
  // new job. Anything else (queued/processing/done) keeps the existing claim.
  const status = (await redis().hget(key, "status")) as string | null;
  if (status === "failed") {
    await redis().hset(key, { status: "queued", job_id, created_at: now() });
    await redis().hdel(key, "error");
    return true;
  }
  return false;
}

// ---------- jobs ----------

export type JobRecord = {
  job_id: string;
  sha256: string;
  object_path: string;
  username: string;
  status: string;
  created_at: string;
  updated_at: string;
  manifest_url?: string;
  error?: string;
};

export async function createJob(
  job_id: string,
  sha256: string,
  object_path: string,
  username: string,
): Promise<void> {
  await redis().hset(`job:${job_id}`, {
    job_id,
    sha256,
    object_path,
    username,
    status: "queued",
    created_at: now(),
    updated_at: now(),
  });
}

export async function getJob(job_id: string): Promise<JobRecord | null> {
  const data = (await redis().hgetall(`job:${job_id}`)) as JobRecord | null;
  return data && Object.keys(data).length > 0 ? data : null;
}

export async function appendUserJob(username: string, job_id: string): Promise<void> {
  const key = `user:${username}:jobs`;
  await redis().lpush(key, job_id);
  await redis().ltrim(key, 0, MAX_JOBS_PER_USER - 1);
}

export async function listUserJobIds(username: string, limit = 20): Promise<string[]> {
  return ((await redis().lrange(`user:${username}:jobs`, 0, limit - 1)) as string[]) ?? [];
}

// ---------- logs for a single job (read-side) ----------

export type LogEntryRead = {
  ts: number;
  stage: string;
  job_id?: string;
  level: string;
  msg: string;
  data?: unknown;
  err?: unknown;
};

const STAGE_STREAMS = [
  "logs:orchestrator",
  "logs:separate",
  "logs:transcribe",
  "logs:align",
  "logs:compose",
  "logs:record-mix",
];

function asObject(v: unknown): unknown {
  if (v === undefined || v === null || v === "") return undefined;
  if (typeof v === "string") {
    try { return JSON.parse(v); } catch { return v; }
  }
  return v;
}

export async function logsForJob(job_id: string, sinceMs = 0): Promise<LogEntryRead[]> {
  const r = redis();
  const start = sinceMs > 0 ? String(sinceMs) : "-";
  const out: LogEntryRead[] = [];
  for (const stream of STAGE_STREAMS) {
    const rows = (await r.xrange(stream, start, "+")) as Record<string, Record<string, string>>;
    for (const [, fields] of Object.entries(rows)) {
      if (fields["job_id"] !== job_id) continue;
      out.push({
        ts: Number(fields["ts"] ?? 0),
        stage: fields["stage"] ?? stream.replace(/^logs:/, ""),
        job_id: fields["job_id"] || undefined,
        level: fields["level"] ?? "info",
        msg: fields["msg"] ?? "",
        data: asObject(fields["data"]),
        err: asObject(fields["err"]),
      });
    }
  }
  out.sort((a, b) => a.ts - b.ts);
  return out;
}

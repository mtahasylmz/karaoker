#!/usr/bin/env node
/**
 * pnpm logs [options]
 *   --stage <name>       stage to tail; "*" for all (default: "*")
 *   --job <id>           only show entries with this job_id
 *   --level <lvl>        only show entries at/above this level (debug|info|warn|error)
 *   --since <duration>   only show entries newer than this (e.g. 10m, 1h, 30s)
 *   --follow, -f         live-tail, poll every 1.5s
 *   --json               raw JSON output (one entry per line) instead of pretty
 */

import { parseArgs } from "node:util";
import { listStreams, parseDuration, pollOnce, type RawEntry } from "./tail.js";

const LEVEL_ORDER = { debug: 0, info: 1, warn: 2, error: 3 } as const;
type Level = keyof typeof LEVEL_ORDER;

const COLOR = {
  reset: "\x1b[0m",
  dim: "\x1b[2m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  red: "\x1b[31m",
  gray: "\x1b[90m",
};

function color(level: Level): string {
  return { debug: COLOR.gray, info: COLOR.green, warn: COLOR.yellow, error: COLOR.red }[level];
}

function fmtTs(ts: number): string {
  const d = new Date(ts);
  return d.toISOString().slice(11, 23); // HH:MM:SS.mmm
}

type Opts = {
  stage: string;
  job?: string;
  level: Level;
  since: number;
  follow: boolean;
  json: boolean;
};

function parseOpts(): Opts {
  const { values } = parseArgs({
    options: {
      stage: { type: "string", default: "*" },
      job: { type: "string" },
      level: { type: "string", default: "debug" },
      since: { type: "string" },
      follow: { type: "boolean", short: "f", default: false },
      json: { type: "boolean", default: false },
      help: { type: "boolean", short: "h", default: false },
    },
    allowPositionals: false,
  });
  if (values.help) {
    process.stdout.write(
      "pnpm logs [--stage <name>|*] [--job <id>] [--level debug|info|warn|error] [--since 10m] [--follow] [--json]\n",
    );
    process.exit(0);
  }
  const level = (values.level as Level) ?? "debug";
  if (!(level in LEVEL_ORDER)) {
    process.stderr.write(`invalid --level ${level}\n`);
    process.exit(2);
  }
  return {
    stage: (values.stage as string) ?? "*",
    job: values.job as string | undefined,
    level,
    since: parseDuration(values.since as string | undefined),
    follow: Boolean(values.follow),
    json: Boolean(values.json),
  };
}

function entryAllowed(raw: RawEntry, opts: Opts): boolean {
  const level = (raw.fields["level"] as Level) ?? "info";
  if (!(level in LEVEL_ORDER)) return false;
  if (LEVEL_ORDER[level] < LEVEL_ORDER[opts.level]) return false;
  if (opts.job && raw.fields["job_id"] !== opts.job) return false;
  if (opts.since) {
    const ts = Number(raw.fields["ts"] ?? 0);
    if (ts < Date.now() - opts.since) return false;
  }
  return true;
}

function asText(v: unknown): string {
  if (v === undefined || v === null || v === "") return "";
  if (typeof v === "string") return v;
  try { return JSON.stringify(v); } catch { return String(v); }
}

function asObject(v: unknown): unknown {
  if (v === undefined || v === null || v === "") return undefined;
  if (typeof v === "string") {
    try { return JSON.parse(v); } catch { return v; }
  }
  return v;
}

function printEntry(raw: RawEntry, opts: Opts) {
  if (opts.json) {
    const payload = {
      ts: Number(raw.fields["ts"] ?? 0),
      stage: raw.fields["stage"] ?? raw.stream.replace(/^logs:/, ""),
      job_id: raw.fields["job_id"] || undefined,
      level: raw.fields["level"],
      msg: raw.fields["msg"],
      data: asObject(raw.fields["data"]),
      err: asObject(raw.fields["err"]),
    };
    process.stdout.write(JSON.stringify(payload) + "\n");
    return;
  }
  const ts = Number(raw.fields["ts"] ?? 0);
  const level = (raw.fields["level"] as Level) ?? "info";
  const stage = raw.fields["stage"] ?? raw.stream.replace(/^logs:/, "");
  const job = raw.fields["job_id"];
  const dataText = asText(raw.fields["data"]);
  const errText = asText(raw.fields["err"]);
  const line =
    `${COLOR.dim}${fmtTs(ts)}${COLOR.reset} ` +
    `${COLOR.cyan}[${stage}]${COLOR.reset} ` +
    `${color(level)}${level.padEnd(5)}${COLOR.reset} ` +
    (job ? `${COLOR.dim}job=${job}${COLOR.reset} ` : "") +
    `${raw.fields["msg"] ?? ""}` +
    (dataText ? ` ${COLOR.dim}${dataText}${COLOR.reset}` : "") +
    (errText ? `\n  ${COLOR.red}${errText}${COLOR.reset}` : "");
  process.stdout.write(line + "\n");
}

async function resolveStreams(stagePattern: string): Promise<string[]> {
  if (stagePattern === "*") return listStreams();
  return [`logs:${stagePattern}`];
}

async function main() {
  const opts = parseOpts();
  const streams = await resolveStreams(opts.stage);
  const lastIds: Record<string, string> = {};

  // First pass: honor --since by backfilling from either the stream start or
  // the cutoff. With --since, we want history; without it and with --follow,
  // we only want new entries.
  if (opts.since || !opts.follow) {
    // Seed lastIds with the `since` cutoff id (ms form) so xrange picks up from there.
    const seedMs = opts.since ? Date.now() - opts.since : 0;
    for (const s of streams) lastIds[s] = seedMs > 0 ? String(seedMs - 1) : "-";
    // Override "-" placeholder: xrange needs "-" as start, not as lastId.
    const entries = await pollOnce(streams, lastIds);
    entries.filter((e) => entryAllowed(e, opts)).forEach((e) => printEntry(e, opts));
    if (!opts.follow) return;
  } else {
    // Follow mode with no --since: seed lastIds to "now" so we only see new stuff.
    for (const s of streams) lastIds[s] = String(Date.now());
  }

  // Follow loop.
  while (true) {
    await new Promise((r) => setTimeout(r, 1500));
    try {
      const entries = await pollOnce(streams, lastIds);
      entries.filter((e) => entryAllowed(e, opts)).forEach((e) => printEntry(e, opts));
    } catch (e) {
      process.stderr.write(`[logs] poll error: ${(e as Error).message}\n`);
    }
  }
}

main().catch((e) => {
  process.stderr.write(`[logs] fatal: ${(e as Error).message}\n`);
  process.exit(1);
});

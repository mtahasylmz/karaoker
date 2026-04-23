/** XRANGE-polling Redis Stream tail. Works with Upstash's HTTP-based Redis. */

import { redis } from "@annemusic/shared-ts/redis";

export type RawEntry = {
  stream: string; // e.g. "logs:separate"
  id: string; // "<ms>-<seq>"
  fields: Record<string, string>;
};

/** Convert a ms epoch to a Redis Stream ID at or after that time. */
export function msToStreamId(ms: number): string {
  return `${ms}`;
}

/** Turn "10m", "1h", "30s" into ms. "0" / "" returns 0. */
export function parseDuration(s: string | undefined): number {
  if (!s) return 0;
  const m = /^(\d+)\s*(ms|s|m|h|d)?$/.exec(s.trim());
  if (!m) return 0;
  const n = Number(m[1]);
  const unit = m[2] ?? "ms";
  return n * { ms: 1, s: 1_000, m: 60_000, h: 3_600_000, d: 86_400_000 }[unit]!;
}

/** Discover every `logs:*` stream via SCAN. */
export async function listStreams(): Promise<string[]> {
  const r = redis();
  const found = new Set<string>();
  let cursor = "0";
  do {
    const [next, keys] = (await r.scan(cursor, { match: "logs:*", count: 100 })) as [string, string[]];
    keys.forEach((k) => found.add(k));
    cursor = next;
  } while (cursor !== "0");
  return [...found].sort();
}

/**
 * One poll cycle: fetch entries newer than `lastIds[stream]` for each stream.
 * Returns merged entries sorted by ts ascending; updates `lastIds` in place.
 */
export async function pollOnce(
  streams: string[],
  lastIds: Record<string, string>,
): Promise<RawEntry[]> {
  const r = redis();
  const all: RawEntry[] = [];
  for (const stream of streams) {
    const after = lastIds[stream] ?? "-";
    // Upstash xrange: "(id" means exclusive lower bound.
    const start = after === "-" ? "-" : `(${after}`;
    const rows = (await r.xrange(stream, start, "+")) as Record<
      string,
      Record<string, string>
    >;
    for (const [id, fields] of Object.entries(rows)) {
      all.push({ stream, id, fields });
      lastIds[stream] = id;
    }
  }
  all.sort((a, b) => Number(a.fields["ts"] ?? 0) - Number(b.fields["ts"] ?? 0));
  return all;
}

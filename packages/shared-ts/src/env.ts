/** Env var access with explicit failure on missing required values. */

export function required(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`missing required env var: ${name}`);
  return v;
}

export function optional(name: string, fallback = ""): string {
  return process.env[name] ?? fallback;
}

export function optionalNumber(name: string, fallback: number): number {
  const v = process.env[name];
  if (v === undefined || v === "") return fallback;
  const n = Number(v);
  if (!Number.isFinite(n)) throw new Error(`env var ${name}=${v} is not a number`);
  return n;
}

export const isLocal = () =>
  (process.env.NODE_ENV ?? "").toLowerCase() === "local" ||
  process.env.ANNEMUSIC_LOCAL === "1";

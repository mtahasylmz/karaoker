/**
 * Pure manifest assembly. Kept separate from main.ts so tests + bench can
 * import it without triggering the Hono server bootstrap.
 *
 * `assUrl` is whatever the handler got back from uploadBuffer — an https://
 * URL in prod, a file:// URL when DEV_FS_ROOT is set. `publicUrl()` resolves
 * `video_uri` / `instrumental_uri` the same way.
 */

import type { ComposeRequest, PlaybackManifest } from "@annemusic/contracts";
import { publicUrl } from "@annemusic/shared-ts/gcs";

import { buildLines } from "./ass.js";

export function buildManifest(
  req: ComposeRequest,
  assUrl: string,
  now: number = Date.now(),
): PlaybackManifest {
  const lines = buildLines(req.words, req.style ?? {});
  const duration = req.words.length > 0
    ? req.words[req.words.length - 1]!.end
    : undefined;
  return {
    job_id: req.job_id,
    video_url: publicUrl(stripGsPrefix(req.video_uri)),
    instrumental_url: publicUrl(stripGsPrefix(req.instrumental_uri)),
    ass_url: assUrl,
    language: req.language,
    duration,
    created_at: now,
    lines,
    vocal_activity: req.vocal_activity,
  };
}

export function stripGsPrefix(uri: string): string {
  if (uri.startsWith("gs://")) {
    const idx = uri.indexOf("/", "gs://".length);
    return idx >= 0 ? uri.slice(idx + 1) : uri;
  }
  if (uri.startsWith("file://")) return uri.slice("file://".length);
  return uri;
}

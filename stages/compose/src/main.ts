/**
 * Compose stage. Word timings + media URIs → .ass file + manifest JSON,
 * both uploaded to GCS (or DEV_FS_ROOT in dev). No video re-encode — the
 * browser plays the original video and overlays the .ass via JASSUB.
 */

import { serve as nodeServe } from "@hono/node-server";
import { Hono } from "hono";
import {
  ComposeRequest,
  type ComposeResponse,
} from "@annemusic/contracts";
import { createLogger } from "@annemusic/shared-ts/logger";
import { uploadBuffer } from "@annemusic/shared-ts/gcs";

import { buildAss } from "./ass.js";
import { buildManifest } from "./manifest.js";

const log = createLogger("compose");

const app = new Hono();

app.get("/ping", (c) => c.json({ ok: true, service: "compose" }));

app.post("/process", async (c) => {
  const started = Date.now();
  const body = await c.req.json();
  const parsed = ComposeRequest.safeParse(body);
  if (!parsed.success) {
    log.error(undefined, "invalid request", parsed.error, { issues: parsed.error.issues });
    return c.json({ detail: "contract violation", issues: parsed.error.issues }, 400);
  }
  const req = parsed.data;
  const job_id = req.job_id;
  try {
    log.info(job_id, "starting", { words: req.words.length });

    const ass = buildAss(req.words, req.style ?? {});
    const ass_object = `stages/compose/${job_id}/lyrics.ass`;
    const ass_url = await uploadBuffer(ass_object, ass, "text/x-ssa");
    log.debug(job_id, "ass uploaded", { bytes: Buffer.byteLength(ass) });

    const manifest = buildManifest(req, ass_url);
    log.debug(job_id, "manifest shape", {
      lines: manifest.lines.length,
      vocal_activity: manifest.vocal_activity.length,
    });
    const manifest_object = `stages/compose/${job_id}/manifest.json`;
    const manifest_url = await uploadBuffer(
      manifest_object,
      JSON.stringify(manifest, null, 2),
      "application/json",
    );
    log.info(job_id, "done", { duration_ms: Date.now() - started });

    const resp: ComposeResponse = {
      job_id,
      stage: "compose",
      started_at: started,
      finished_at: Date.now(),
      duration_ms: Date.now() - started,
      manifest_uri: toGsUri(manifest_object),
      manifest_url,
      ass_uri: toGsUri(ass_object),
    };
    return c.json(resp);
  } catch (e) {
    log.error(job_id, "compose failed", e);
    return c.json({ detail: `${(e as Error).name}: ${(e as Error).message}` }, 500);
  }
});

function toGsUri(objectPath: string): string {
  const bucket = process.env.GCS_BUCKET ?? "local";
  return `gs://${bucket}/${objectPath}`;
}

const port = Number(process.env.PORT ?? 8104);
nodeServe({ fetch: app.fetch, port });
console.log(JSON.stringify({
  ts: Date.now(),
  stage: "compose",
  level: "info",
  msg: "listening",
  data: { port },
}));

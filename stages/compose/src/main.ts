/**
 * Compose stage. Word timings + media URIs → .ass file + manifest JSON,
 * both uploaded to GCS (or DEV_FS_ROOT in dev). No video re-encode — the
 * browser plays the original video and overlays the .ass via JASSUB.
 */

import { timingSafeEqual } from "node:crypto";

import { serve as nodeServe } from "@hono/node-server";
import { Hono } from "hono";
import {
  ComposeRequest,
  ComposeResponse,
  PlaybackManifest,
} from "@annemusic/contracts";
import { createLogger } from "@annemusic/shared-ts/logger";
import { uploadBuffer } from "@annemusic/shared-ts/gcs";

import { buildAss } from "./ass.js";
import { buildManifest } from "./manifest.js";

const log = createLogger("compose");

const app = new Hono();

app.get("/ping", (c) => c.json({ ok: true, service: "compose" }));

// Stage auth: Cloud Run ingress is open (QStash can't mint OIDC tokens for
// the orchestrator's context.call), so callers prove themselves with a
// shared bearer token instead. Unset STAGE_AUTH_TOKEN = open (local dev).
app.use("/process", async (c, next) => {
  const token = process.env.STAGE_AUTH_TOKEN ?? "";
  if (token) {
    const got = Buffer.from(c.req.header("authorization") ?? "");
    const want = Buffer.from(`Bearer ${token}`);
    if (got.length !== want.length || !timingSafeEqual(got, want)) {
      return c.json({ detail: "missing or invalid stage token" }, 401);
    }
  }
  await next();
});

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

    // Producer-side contract checks: the manifest is the browser's contract
    // and the response is the orchestrator's — fail loudly here, not there.
    const manifest = PlaybackManifest.parse(buildManifest(req, ass_url));
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

    const resp = ComposeResponse.parse({
      job_id,
      stage: "compose",
      started_at: started,
      finished_at: Date.now(),
      duration_ms: Date.now() - started,
      manifest_uri: toGsUri(manifest_object),
      manifest_url,
      ass_uri: toGsUri(ass_object),
    });
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

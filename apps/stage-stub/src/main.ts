/**
 * Phase-C stub: a single Hono app that pretends to be all 5 stages.
 * Each route takes ~1-3 seconds, emits 2-3 log entries, and returns a
 * contract-shaped stub response with fake GCS URIs.
 *
 * Phase D replaces each stage/<name> with its real implementation; this
 * process goes away.
 */

import { serve as nodeServe } from "@hono/node-server";
import { Hono } from "hono";
import {
  AlignRequest,
  AlignResponse,
  ComposeRequest,
  ComposeResponse,
  RecordMixRequest,
  RecordMixResponse,
  SeparateRequest,
  SeparateResponse,
  TranscribeRequest,
  TranscribeResponse,
  type StageJobId,
  type StageName,
} from "@annemusic/contracts";
import { createLogger, type Logger } from "@annemusic/shared-ts/logger";

const loggers: Record<StageName, Logger> = {
  separate: createLogger("separate"),
  transcribe: createLogger("transcribe"),
  align: createLogger("align"),
  compose: createLogger("compose"),
  "record-mix": createLogger("record-mix"),
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const rand = (min: number, max: number) => min + Math.random() * (max - min);

const app = new Hono();

app.get("/ping", (c) => c.json({ ok: true, service: "stage-stub" }));

app.post("/separate/process", async (c) => {
  const log = loggers.separate;
  const started = Date.now();
  try {
    const body = SeparateRequest.parse(await c.req.json());
    log.info(body.job_id, "stub separate: starting", { source: body.source_uri });
    await sleep(rand(800, 2000));
    log.info(body.job_id, "stub separate: demucs complete");
    const bucket = process.env.GCS_BUCKET ?? "local-bucket";
    const resp: SeparateResponse = {
      job_id: body.job_id,
      stage: "separate",
      started_at: started,
      finished_at: Date.now(),
      duration_ms: Date.now() - started,
      vocals_uri: `gs://${bucket}/stub/${body.job_id}/vocals.wav`,
      instrumental_uri: `gs://${bucket}/stub/${body.job_id}/no_vocals.wav`,
      sample_rate: 44100,
      model_used: body.model ?? "htdemucs",
    };
    return c.json(resp);
  } catch (e) {
    loggers.separate.error(undefined, "stub separate error", e);
    return c.json({ error: (e as Error).message }, 500);
  }
});

app.post("/transcribe/process", async (c) => {
  const log = loggers.transcribe;
  const started = Date.now();
  try {
    const body = TranscribeRequest.parse(await c.req.json());
    log.info(body.job_id, "stub transcribe: starting", {
      known_lyrics: !!body.known_lyrics,
      title: body.title,
    });
    await sleep(rand(800, 2000));
    log.info(body.job_id, "stub transcribe: complete");
    const language = body.language ?? "tr";
    const resp: TranscribeResponse = {
      job_id: body.job_id,
      stage: "transcribe",
      started_at: started,
      finished_at: Date.now(),
      duration_ms: Date.now() - started,
      language,
      segments: [
        { text: "stub segment one two three", start: 0.5, end: 3.2 },
        { text: "stub segment four five", start: 4.1, end: 6.8 },
      ],
      source: "whisper",
      model_used: "stub",
    };
    return c.json(resp);
  } catch (e) {
    loggers.transcribe.error(undefined, "stub transcribe error", e);
    return c.json({ error: (e as Error).message }, 500);
  }
});

app.post("/align/process", async (c) => {
  const log = loggers.align;
  const started = Date.now();
  try {
    const body = AlignRequest.parse(await c.req.json());
    log.info(body.job_id, "stub align: starting", { segments: body.segments.length });
    await sleep(rand(500, 1500));
    // Synthesize per-word timings by evenly splitting each segment.
    const words = body.segments.flatMap((seg) => {
      const toks = seg.text.split(/\s+/).filter(Boolean);
      if (toks.length === 0) return [];
      const step = (seg.end - seg.start) / toks.length;
      return toks.map((tok, i) => ({
        text: tok,
        start: seg.start + i * step,
        end: seg.start + (i + 1) * step,
        score: 0.9,
      }));
    });
    log.info(body.job_id, "stub align: complete", { words: words.length });
    const resp: AlignResponse = {
      job_id: body.job_id,
      stage: "align",
      started_at: started,
      finished_at: Date.now(),
      duration_ms: Date.now() - started,
      words,
    };
    return c.json(resp);
  } catch (e) {
    loggers.align.error(undefined, "stub align error", e);
    return c.json({ error: (e as Error).message }, 500);
  }
});

app.post("/compose/process", async (c) => {
  const log = loggers.compose;
  const started = Date.now();
  try {
    const body = ComposeRequest.parse(await c.req.json());
    log.info(body.job_id, "stub compose: starting", { words: body.words.length });
    await sleep(rand(300, 900));
    const bucket = process.env.GCS_BUCKET ?? "local-bucket";
    const manifest_uri = `gs://${bucket}/stub/${body.job_id}/manifest.json`;
    const ass_uri = `gs://${bucket}/stub/${body.job_id}/lyrics.ass`;
    log.info(body.job_id, "stub compose: complete");
    const resp: ComposeResponse = {
      job_id: body.job_id,
      stage: "compose",
      started_at: started,
      finished_at: Date.now(),
      duration_ms: Date.now() - started,
      manifest_uri,
      manifest_url: `https://storage.googleapis.com/${bucket}/stub/${body.job_id}/manifest.json`,
      ass_uri,
    };
    return c.json(resp);
  } catch (e) {
    loggers.compose.error(undefined, "stub compose error", e);
    return c.json({ error: (e as Error).message }, 500);
  }
});

app.post("/record-mix/process", async (c) => {
  const log = loggers["record-mix"];
  const started = Date.now();
  try {
    const body = RecordMixRequest.parse(await c.req.json());
    log.info(body.job_id, "stub record-mix: starting", { autotune: body.autotune });
    await sleep(rand(500, 1500));
    const bucket = process.env.GCS_BUCKET ?? "local-bucket";
    log.info(body.job_id, "stub record-mix: complete");
    const resp: RecordMixResponse = {
      job_id: body.job_id,
      stage: "record-mix",
      started_at: started,
      finished_at: Date.now(),
      duration_ms: Date.now() - started,
      mix_uri: `gs://${bucket}/stub/${body.job_id}/mix.mp3`,
    };
    return c.json(resp);
  } catch (e) {
    loggers["record-mix"].error(undefined, "stub record-mix error", e);
    return c.json({ error: (e as Error).message }, 500);
  }
});

const port = Number(process.env.PORT ?? 8100);
nodeServe({ fetch: app.fetch, port });
console.log(JSON.stringify({
  ts: Date.now(),
  stage: "stage-stub",
  level: "info",
  msg: "listening",
  data: { port, routes: ["/separate", "/transcribe", "/align", "/compose", "/record-mix"] },
}));

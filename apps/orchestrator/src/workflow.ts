/**
 * annemusic workflow: source file → separate → transcribe → align → compose.
 * Each `context.call` hits a stage HTTP endpoint; Upstash Workflow holds the
 * call on their side with retries (up to 12h per step), so stages stay plain
 * request/response servers with no long-poll semantics.
 */

import { serve } from "@upstash/workflow/hono";
import type {
  AlignResponse,
  ComposeResponse,
  SeparateResponse,
  StageJobId,
  TranscribeResponse,
} from "@annemusic/contracts";
import { createLogger } from "@annemusic/shared-ts/logger";
import { required } from "@annemusic/shared-ts/env";

export type WorkflowPayload = {
  job_id: StageJobId;
  username: string;
  sha256: string;
  source_uri: string; // gs://bucket/uploads/<sha>.<ext>
  content_type: string;
  known_lyrics?: string;
  title?: string;
  artist?: string;
  language?: string;
};

const log = createLogger("orchestrator");

const urls = () => ({
  separate: required("SEPARATE_URL"),
  transcribe: required("TRANSCRIBE_URL"),
  align: required("ALIGN_URL"),
  compose: required("COMPOSE_URL"),
});

export const annemusicWorkflow = serve<WorkflowPayload>(async (context) => {
  const p = context.requestPayload;
  const { job_id } = p;
  const u = urls();

  log.info(job_id, "workflow started", { sha: p.sha256, user: p.username });

  const separate = await context.call<SeparateResponse>("separate", {
    url: `${u.separate}/process`,
    method: "POST",
    body: { job_id, source_uri: p.source_uri },
    headers: { "content-type": "application/json" },
    retries: 2,
  });
  if (separate.status < 200 || separate.status >= 300) {
    log.error(job_id, "separate failed", new Error(`status=${separate.status}`), {
      body: separate.body,
    });
    throw new Error(`separate returned ${separate.status}`);
  }
  const sep = separate.body as SeparateResponse;

  const transcribe = await context.call<TranscribeResponse>("transcribe", {
    url: `${u.transcribe}/process`,
    method: "POST",
    body: {
      job_id,
      vocals_uri: sep.vocals_uri,
      language: p.language,
      known_lyrics: p.known_lyrics,
      title: p.title,
      artist: p.artist,
    },
    headers: { "content-type": "application/json" },
    retries: 2,
  });
  if (transcribe.status < 200 || transcribe.status >= 300) {
    log.error(job_id, "transcribe failed", new Error(`status=${transcribe.status}`));
    throw new Error(`transcribe returned ${transcribe.status}`);
  }
  const tr = transcribe.body as TranscribeResponse;

  const align = await context.call<AlignResponse>("align", {
    url: `${u.align}/process`,
    method: "POST",
    body: {
      job_id,
      vocals_uri: sep.vocals_uri,
      segments: tr.segments,
      language: tr.language,
    },
    headers: { "content-type": "application/json" },
    retries: 2,
  });
  if (align.status < 200 || align.status >= 300) {
    log.error(job_id, "align failed", new Error(`status=${align.status}`));
    throw new Error(`align returned ${align.status}`);
  }
  const al = align.body as AlignResponse;

  const compose = await context.call<ComposeResponse>("compose", {
    url: `${u.compose}/process`,
    method: "POST",
    body: {
      job_id,
      words: al.words,
      video_uri: p.source_uri,
      instrumental_uri: sep.instrumental_uri,
      language: tr.language,
    },
    headers: { "content-type": "application/json" },
    retries: 2,
  });
  if (compose.status < 200 || compose.status >= 300) {
    log.error(job_id, "compose failed", new Error(`status=${compose.status}`));
    throw new Error(`compose returned ${compose.status}`);
  }
  const co = compose.body as ComposeResponse;

  // Pipe the manifest URL back to Redis so the API can serve it.
  await context.run("persist-manifest", async () => {
    const { redis } = await import("@annemusic/shared-ts/redis");
    await redis().hset(`job:${job_id}`, {
      status: "done",
      manifest_url: co.manifest_url,
      manifest_uri: co.manifest_uri,
      ass_uri: co.ass_uri,
      vocals_uri: sep.vocals_uri,
      instrumental_uri: sep.instrumental_uri,
      language: tr.language,
      finished_at: String(Date.now()),
    });
    await redis().hset(`video:${p.sha256}`, {
      status: "done",
      manifest_url: co.manifest_url,
      job_id,
    });
  });

  log.info(job_id, "workflow completed", { manifest_url: co.manifest_url });
});

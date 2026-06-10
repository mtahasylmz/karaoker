import { serve as nodeServe } from "@hono/node-server";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { Client } from "@upstash/workflow";
import { createLogger } from "@annemusic/shared-ts/logger";
import { required, optional, optionalNumber, isLocal } from "@annemusic/shared-ts/env";
import {
  signedPutUrl,
  signedGetUrl,
  objectExists,
  publicUrl,
  objectPathFromGsUri,
} from "@annemusic/shared-ts/gcs";
import { proxyStage, type StageName } from "./stage-proxy.js";
import {
  appendUserJob,
  claimVideo,
  createJob,
  getJob,
  getUpload,
  getVideo,
  listUserJobIds,
  logsForJob,
  recordUpload,
  reserveUsername,
  userExists,
  validSha256,
  validUsername,
} from "./state.js";

const log = createLogger("api");

const MAX_UPLOAD_BYTES = 500 * 1024 * 1024;
const EXT_BY_CONTENT_TYPE: Record<string, string> = {
  "video/mp4": "mp4",
  "video/quicktime": "mov",
  "video/webm": "webm",
  "video/x-matroska": "mkv",
};

const app = new Hono();

app.use(
  "*",
  cors({
    origin: optional("CORS_ORIGINS", "*") === "*" ? "*" : optional("CORS_ORIGINS").split(","),
    allowMethods: ["GET", "POST", "OPTIONS"],
    allowHeaders: ["content-type", "x-manual-key"],
  }),
);

app.onError((err, c) => {
  log.error(undefined, "unhandled", err, { path: c.req.path });
  return c.json({ detail: `${err.name}: ${err.message}` }, 500);
});

app.get("/ping", (c) => c.json({ ok: true, service: "api" }));

// ---------- users ----------

app.post("/users", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const username = (body?.username ?? "").trim();
  if (!validUsername(username)) return c.json({ detail: "invalid username" }, 400);
  const ok = await reserveUsername(username);
  if (!ok) return c.json({ detail: "username taken" }, 409);
  log.info(undefined, "user registered", { username });
  return c.json({ username }, 201);
});

app.get("/users/:username", async (c) => {
  const u = c.req.param("username");
  if (!(await userExists(u))) return c.json({ detail: "unknown username" }, 404);
  return c.json({ username: u });
});

// Dev-only: fires the workflow with a synthetic job so we can prove the
// pipeline shape without GCS plumbing. Disabled in prod.
app.post("/dev/trigger", async (c) => {
  if (!isLocal()) return c.json({ detail: "dev only" }, 403);
  const body = await c.req.json().catch(() => ({}));
  const username = body.username ?? "dev";
  if (!(await userExists(username))) await reserveUsername(username);

  // If the caller didn't specify a sha256, look for the newest file in
  // $DEV_FS_ROOT/uploads/ and reuse its name (sha256.ext) so the staged
  // fixture is exercised instead of a phantom one.
  let sha256 = body.sha256 as string | undefined;
  let object_path: string | undefined;
  if (!sha256) {
    const root = process.env.DEV_FS_ROOT;
    if (root) {
      try {
        const fs = await import("node:fs");
        const path = await import("node:path");
        const dir = path.join(root, "uploads");
        const entries = fs
          .readdirSync(dir, { withFileTypes: true })
          .filter((e) => e.isFile() && /^[a-f0-9]{64}\.(mp4|mov|webm|mkv)$/.test(e.name))
          .map((e) => ({ name: e.name, mtime: fs.statSync(path.join(dir, e.name)).mtimeMs }))
          .sort((a, b) => b.mtime - a.mtime);
        if (entries[0]) {
          sha256 = entries[0].name.split(".")[0]!;
          object_path = `uploads/${entries[0].name}`;
        }
      } catch { /* fall through */ }
    }
  }
  if (!sha256) {
    sha256 = Array.from(crypto.getRandomValues(new Uint8Array(32)))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  }
  if (!object_path) object_path = `uploads/${sha256}.mp4`;
  await recordUpload(sha256, 1024, "video/mp4", object_path);

  const job_id = newJobId();
  await createJob(job_id, sha256, object_path, username);
  await claimVideo(sha256, job_id);
  await appendUserJob(username, job_id);

  const bucket = required("GCS_BUCKET");
  const source_uri = `gs://${bucket}/${object_path}`;

  const client = new Client({
    baseUrl: required("QSTASH_URL"),
    token: required("QSTASH_TOKEN"),
  });
  const orchestratorUrl = required("ORCHESTRATOR_URL");
  const { workflowRunId } = await client.trigger({
    url: `${orchestratorUrl}/workflow`,
    body: {
      job_id,
      username,
      sha256,
      source_uri,
      content_type: "video/mp4",
      title: body.title ?? "Dev Test Song",
      artist: body.artist ?? "Phase C Stub",
    },
    retries: 1,
  });
  log.info(job_id, "dev workflow triggered", { workflowRunId });
  return c.json({ job_id, sha256, workflowRunId });
});

app.get("/users/:username/jobs", async (c) => {
  const u = c.req.param("username");
  if (!(await userExists(u))) return c.json({ detail: "unknown username" }, 404);
  const limit = Number(c.req.query("limit") ?? 20);
  const ids = await listUserJobIds(u, limit);
  const jobs = [];
  for (const id of ids) {
    const j = await getJob(id);
    if (j) jobs.push(j);
  }
  return c.json({ jobs });
});

// ---------- uploads ----------

app.post("/uploads", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const { username, sha256, size, content_type, known_lyrics, title, artist, language } = body;
  if (!(await userExists(username))) return c.json({ detail: "unknown username" }, 404);
  if (!validSha256(sha256)) return c.json({ detail: "sha256 must be 64 hex chars" }, 400);
  if (typeof size !== "number" || size < 1 || size > MAX_UPLOAD_BYTES) {
    return c.json({ detail: `size must be 1..${MAX_UPLOAD_BYTES}` }, 400);
  }
  const ext = EXT_BY_CONTENT_TYPE[content_type];
  if (!ext) {
    return c.json({
      detail: `unsupported content_type; accept one of: ${Object.keys(EXT_BY_CONTENT_TYPE).join(", ")}`,
    }, 400);
  }

  // 1. already done → skip everything.
  const video = await getVideo(sha256);
  if (video && video["status"] === "done" && video["manifest_url"]) {
    return c.json({
      cached: true,
      status: "done",
      sha256,
      manifest_url: video["manifest_url"],
      need_upload: false,
    });
  }

  const bucket = required("GCS_BUCKET");

  // 2. upload record exists AND object present → skip PUT.
  const existing = await getUpload(sha256);
  if (existing?.object_path && (await objectExists(existing.object_path))) {
    return c.json({
      cached: false,
      status: "uploaded",
      sha256,
      object_path: existing.object_path,
      gs_uri: `gs://${bucket}/${existing.object_path}`,
      need_upload: false,
    });
  }

  // 3. Mint a fresh signed PUT URL.
  const object_path = `uploads/${sha256}.${ext}`;
  const signed_put_url = await signedPutUrl(object_path, content_type, 900);
  await recordUpload(sha256, size, content_type, object_path);
  // Stash lyrics hints on the upload so /jobs can pass them to the workflow.
  if (known_lyrics || title || artist || language) {
    const { redis } = await import("@annemusic/shared-ts/redis");
    await redis().hset(`upload:${sha256}`, {
      ...(known_lyrics ? { known_lyrics } : {}),
      ...(title ? { title } : {}),
      ...(artist ? { artist } : {}),
      ...(language ? { language } : {}),
    });
  }

  log.info(undefined, "signed put url minted", { sha256, size, content_type });
  return c.json({
    cached: false,
    status: "pending_upload",
    sha256,
    object_path,
    gs_uri: `gs://${bucket}/${object_path}`,
    need_upload: true,
    signed_put_url,
    expires_in: 900,
  });
});

// ---------- jobs ----------

function newJobId(): string {
  // 12 hex chars — fits StageJobId regex.
  return Array.from(crypto.getRandomValues(new Uint8Array(6)))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

app.post("/jobs", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const { username, sha256 } = body;
  if (!(await userExists(username))) return c.json({ detail: "unknown username" }, 404);
  if (!validSha256(sha256)) return c.json({ detail: "sha256 must be 64 hex chars" }, 400);

  const upload = await getUpload(sha256);
  if (!upload?.object_path) {
    return c.json({ detail: "no upload recorded; call POST /uploads first" }, 404);
  }
  if (!isLocal() && !(await objectExists(upload.object_path))) {
    return c.json({ detail: "upload file not found in storage; re-upload" }, 412);
  }

  // Dedup: running or done job on the same sha?
  const existing = await getVideo(sha256);
  if (existing) {
    if (existing["status"] === "done") {
      return c.json({
        status: "done",
        sha256,
        manifest_url: existing["manifest_url"],
        cached: true,
      });
    }
    if (existing["status"] && existing["status"] !== "failed") {
      return c.json({
        status: existing["status"],
        sha256,
        job_id: existing["job_id"],
        cached: true,
      });
    }
  }

  const job_id = newJobId();
  await createJob(job_id, sha256, upload.object_path, username);
  await claimVideo(sha256, job_id);
  await appendUserJob(username, job_id);

  const bucket = required("GCS_BUCKET");
  const source_uri = `gs://${bucket}/${upload.object_path}`;

  const client = new Client({
    baseUrl: required("QSTASH_URL"),
    token: required("QSTASH_TOKEN"),
  });
  const orchestratorUrl = required("ORCHESTRATOR_URL");

  try {
    const { workflowRunId } = await client.trigger({
      url: `${orchestratorUrl}/workflow`,
      body: {
        job_id,
        username,
        sha256,
        source_uri,
        content_type: upload.content_type,
        known_lyrics: upload["known_lyrics" as keyof typeof upload],
        title: upload["title" as keyof typeof upload],
        artist: upload["artist" as keyof typeof upload],
        language: upload["language" as keyof typeof upload],
      },
      retries: 2,
    });
    log.info(job_id, "workflow triggered", { workflowRunId, orchestratorUrl });
  } catch (e) {
    log.error(job_id, "trigger failed", e);
    // leave job in queued state; user can retry
  }

  return c.json({ status: "queued", sha256, job_id, cached: false });
});

app.get("/jobs/:job_id", async (c) => {
  const id = c.req.param("job_id");
  const job = await getJob(id);
  if (!job) return c.json({ detail: "unknown job" }, 404);
  // Recent logs for UI (all stages, sorted by ts).
  const sinceMs = Number(c.req.query("since_ms") ?? job.created_at) || 0;
  const logs = await logsForJob(id, sinceMs);
  return c.json({ ...job, logs });
});

// ---------- manual test harness (/manual) ----------
//
// The /manual browser UI calls a single stage at a time so a human can
// observe each intermediate artifact before advancing. Routes here proxy
// to the stage endpoints with OIDC auth (prod) or pass through (local dev),
// mint signed GET URLs for GCS artifacts, and — in dev — stream local files
// so browser audio players can read them.

const MANUAL_STAGES: readonly StageName[] = [
  "separate", "transcribe", "align", "compose", "record-mix",
] as const;

// The /manual surface can drive arbitrarily many real (GPU-priced) stage
// runs and mint signed GET URLs, so it is closed by default outside local
// dev: set MANUAL_TOKEN to enable it in prod, and the browser must echo it
// in x-manual-key (the /manual UI prompts once and stores it).
app.use("/manual/*", async (c, next) => {
  const token = optional("MANUAL_TOKEN", "");
  if (token) {
    if ((c.req.header("x-manual-key") ?? "") !== token) {
      return c.json({ detail: "manual key required" }, 401);
    }
  } else if (!isLocal()) {
    return c.json({ detail: "manual harness disabled (set MANUAL_TOKEN to enable)" }, 403);
  }
  await next();
});

app.post("/manual/job-id", (c) => c.json({ job_id: newJobId() }));

app.post("/manual/signed-get", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const uri = typeof body?.uri === "string" ? body.uri : "";
  if (!uri) return c.json({ detail: "uri required" }, 400);
  let path: string;
  try {
    path = objectPathFromGsUri(uri);
  } catch (e) {
    return c.json({ detail: `invalid uri: ${(e as Error).message}` }, 400);
  }
  if (process.env.DEV_FS_ROOT) {
    // Dev mode: the browser can't open file:// via fetch, so point it at the
    // streaming endpoint below. Dev takes precedence over signing so the same
    // harness works without real GCS credentials.
    return c.json({ url: `/api/manual/file?path=${encodeURIComponent(path)}` });
  }
  // GCS_URL_MODE=public → hand back the stable public URL (infra/setup.sh
  // grants allUsers:objectViewer on the annemusic bucket, so this works
  // without any credentials on the browser side and without the api having
  // to sign).
  if (optional("GCS_URL_MODE", "public") !== "signed") {
    return c.json({ url: publicUrl(path) });
  }
  const url = await signedGetUrl(path, 900);
  return c.json({ url });
});

app.get("/manual/file", async (c) => {
  // Dev-mode only: stream files from DEV_FS_ROOT. Restrict to paths under
  // uploads/ and stages/ so nothing else on disk is exposed.
  const root = process.env.DEV_FS_ROOT;
  if (!root) return c.json({ detail: "dev only (DEV_FS_ROOT not set)" }, 403);
  const raw = c.req.query("path") ?? "";
  if (!raw.startsWith("uploads/") && !raw.startsWith("stages/")) {
    return c.json({ detail: "path must be under uploads/ or stages/" }, 403);
  }
  if (raw.includes("..")) return c.json({ detail: "no .." }, 403);
  const fs = await import("node:fs");
  const nodePath = await import("node:path");
  const abs = nodePath.join(root, raw);
  if (!fs.existsSync(abs)) return c.json({ detail: "not found" }, 404);
  const ext = nodePath.extname(abs).toLowerCase();
  const mime =
    ext === ".wav" ? "audio/wav"
    : ext === ".mp3" ? "audio/mpeg"
    : ext === ".mp4" ? "video/mp4"
    : ext === ".webm" ? "video/webm"
    : ext === ".ass" ? "text/plain; charset=utf-8"
    : ext === ".json" ? "application/json"
    : "application/octet-stream";
  const stream = fs.createReadStream(abs);
  const web = new ReadableStream({
    start(ctrl) {
      stream.on("data", (chunk) => ctrl.enqueue(chunk));
      stream.on("end", () => ctrl.close());
      stream.on("error", (err) => ctrl.error(err));
    },
    cancel() { stream.destroy(); },
  });
  return new Response(web, { headers: { "content-type": mime } });
});

app.post("/manual/stages/:stage/process", async (c) => {
  const stage = c.req.param("stage") as StageName;
  if (!MANUAL_STAGES.includes(stage)) {
    return c.json({ detail: `unknown stage: ${stage}` }, 404);
  }
  const body = await c.req.json().catch(() => ({}));
  try {
    const { status, body: out } = await proxyStage(stage, body);
    return c.json(out as Record<string, unknown>, status as 200);
  } catch (e) {
    log.error(undefined, "stage proxy failed", e, { stage });
    return c.json({ detail: `${(e as Error).name}: ${(e as Error).message}` }, 502);
  }
});

const port = optionalNumber("PORT", 8082);
nodeServe({ fetch: app.fetch, port });
console.log(JSON.stringify({
  ts: Date.now(),
  stage: "api",
  level: "info",
  msg: "listening",
  data: { port },
}));

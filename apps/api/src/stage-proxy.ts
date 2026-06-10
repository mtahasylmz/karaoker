/**
 * Forwards a per-stage /process request to its Cloud Run URL (or localhost
 * in dev). Stages are IAM-gated on Cloud Run; we mint an OIDC id-token
 * scoped to the target audience and attach it as a Bearer token. Localhost
 * URLs skip auth. Token clients are cached per audience.
 */

import { Agent, setGlobalDispatcher } from "undici";
import { GoogleAuth } from "google-auth-library";
import type { IdTokenClient } from "google-auth-library";
import { required } from "@annemusic/shared-ts/env";

// Node's default fetch (undici) has 5-min headers + body timeouts. Long CPU
// stage runs (separate ≈ 28 min on CPU for a 4-min song) easily blow past
// that. Disable both so proxyStage can wait for the Cloud Run service's
// own --timeout (we set it to 3600s on each stage).
setGlobalDispatcher(new Agent({ headersTimeout: 0, bodyTimeout: 0 }));

export type StageName =
  | "separate"
  | "transcribe"
  | "align"
  | "compose"
  | "record-mix";

const STAGE_URL_ENV: Record<StageName, string> = {
  separate: "SEPARATE_URL",
  transcribe: "TRANSCRIBE_URL",
  align: "ALIGN_URL",
  compose: "COMPOSE_URL",
  "record-mix": "RECORD_MIX_URL",
};

const _tokenClients = new Map<string, Promise<IdTokenClient>>();
let _auth: GoogleAuth | undefined;

function auth(): GoogleAuth {
  if (!_auth) _auth = new GoogleAuth();
  return _auth;
}

function needsOidc(url: string): boolean {
  return /^https:\/\//.test(url) && !url.includes("localhost") && !url.includes("127.0.0.1");
}

async function authHeader(url: string): Promise<Record<string, string>> {
  // Preferred path: the shared stage bearer token (stages run with open
  // Cloud Run ingress and verify this at the app layer — same credential
  // the orchestrator's context.call sends). OIDC below is the legacy
  // IAM-gated fallback for stages not yet redeployed with a token.
  const token = process.env.STAGE_AUTH_TOKEN;
  if (token) return { authorization: `Bearer ${token}` };
  if (!needsOidc(url)) return {};
  const audience = new URL(url).origin;
  let client = _tokenClients.get(audience);
  if (!client) {
    client = auth().getIdTokenClient(audience);
    _tokenClients.set(audience, client);
  }
  const c = await client;
  const headers = await c.getRequestHeaders();
  // Normalise to a flat Record<string,string> — getRequestHeaders can return
  // either a plain object (older lib) or a Headers-like instance.
  if (headers instanceof Headers) {
    const out: Record<string, string> = {};
    headers.forEach((v, k) => { out[k] = v; });
    return out;
  }
  return headers as Record<string, string>;
}

export type ProxyResult = {
  status: number;
  body: unknown;
};

export async function proxyStage(
  stage: StageName,
  body: unknown,
): Promise<ProxyResult> {
  const base = required(STAGE_URL_ENV[stage]);
  const url = `${base.replace(/\/$/, "")}/process`;
  const extra = await authHeader(url);
  // CPU stage runs (separate / transcribe / align) routinely take 5-30 min.
  // Node's default fetch timeout (~5 min) drops the connection mid-inference;
  // bump to match the Cloud Run service timeout (--timeout 3600).
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...extra,
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(60 * 60 * 1000),
  });
  let parsed: unknown;
  try {
    parsed = await res.json();
  } catch {
    parsed = { detail: `non-JSON response from stage ${stage}` };
  }
  return { status: res.status, body: parsed };
}

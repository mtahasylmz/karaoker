// Thin wrappers for the /manual/* routes on apps/api. Parallel to
// ../api.ts which drives the orchestrated flow — kept separate so the
// manual harness is additive and easy to delete once the UI use-case dies.

const BASE = "/api";
const MANUAL_KEY_STORAGE = "annemusic_manual_key";

// Deployed APIs gate /manual/* behind MANUAL_TOKEN (sent as x-manual-key).
// On the first 401 we prompt once and persist the key in localStorage.
function manualKeyHeader(): Record<string, string> {
  const key = localStorage.getItem(MANUAL_KEY_STORAGE);
  return key ? { "x-manual-key": key } : {};
}

async function post(path: string, body: unknown, retried = false): Promise<unknown> {
  const res = await fetch(BASE + path, {
    method: "POST",
    headers: { "content-type": "application/json", ...manualKeyHeader() },
    body: body == null ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (res.status === 401 && !retried) {
    const key = window.prompt("Manual harness key (MANUAL_TOKEN on the api):");
    if (key) {
      localStorage.setItem(MANUAL_KEY_STORAGE, key.trim());
      return post(path, body, true);
    }
  }
  if (!res.ok) {
    throw new Error(
      typeof data?.detail === "string" ? data.detail : `${res.status} ${res.statusText}`,
    );
  }
  return data;
}

export type StageName =
  | "separate"
  | "transcribe"
  | "align"
  | "compose"
  | "record-mix";

export const manualApi = {
  newJobId: () => post("/manual/job-id", {}) as Promise<{ job_id: string }>,
  signedGet: (uri: string) => post("/manual/signed-get", { uri }) as Promise<{ url: string }>,
  callStage: <T = unknown>(stage: StageName, body: unknown) =>
    post(`/manual/stages/${stage}/process`, body) as Promise<T>,
};

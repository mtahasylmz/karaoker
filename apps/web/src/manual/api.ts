// Thin wrappers for the /manual/* routes on apps/api. Parallel to
// ../api.ts which drives the orchestrated flow — kept separate so the
// manual harness is additive and easy to delete once the UI use-case dies.

const BASE = "/api";

async function post(path: string, body: unknown): Promise<unknown> {
  const res = await fetch(BASE + path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: body == null ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
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

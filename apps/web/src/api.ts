const BASE = "/api";

async function req(path: string, opts: RequestInit = {}) {
  const res = await fetch(BASE + path, {
    ...opts,
    headers: { "content-type": "application/json", ...(opts.headers ?? {}) },
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) throw new Error(data?.detail ?? res.statusText);
  return data;
}

export const api = {
  ping: () => req("/ping"),
  registerUser: (username: string) => req("/users", { method: "POST", body: JSON.stringify({ username }) }),
  getUser: (username: string) => req(`/users/${encodeURIComponent(username)}`),
  requestUpload: (body: {
    username: string;
    sha256: string;
    size: number;
    content_type: string;
    title?: string;
    artist?: string;
    language?: string;
    known_lyrics?: string;
  }) => req("/uploads", { method: "POST", body: JSON.stringify(body) }),
  createJob: (username: string, sha256: string) =>
    req("/jobs", { method: "POST", body: JSON.stringify({ username, sha256 }) }),
  getJob: (job_id: string) => req(`/jobs/${encodeURIComponent(job_id)}`),
  listUserJobs: (username: string) => req(`/users/${encodeURIComponent(username)}/jobs`),
  devTrigger: (username: string, title?: string, artist?: string) =>
    req("/dev/trigger", { method: "POST", body: JSON.stringify({ username, title, artist }) }),
};

/** Stream-hash a file with SubtleCrypto (4 MB chunks). */
export async function sha256File(
  file: File,
  onProgress?: (p: number) => void,
): Promise<string> {
  const chunkSize = 4 * 1024 * 1024;
  const chunks: Uint8Array[] = [];
  for (let offset = 0; offset < file.size; offset += chunkSize) {
    const buf = await file.slice(offset, offset + chunkSize).arrayBuffer();
    chunks.push(new Uint8Array(buf));
    onProgress?.(Math.min(1, (offset + chunkSize) / file.size));
  }
  // Web Crypto doesn't do streaming SHA-256; reassemble for the digest call.
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const combined = new Uint8Array(total);
  let pos = 0;
  for (const c of chunks) {
    combined.set(c, pos);
    pos += c.length;
  }
  const digest = await crypto.subtle.digest("SHA-256", combined);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export function putWithProgress(
  url: string,
  file: File,
  contentType: string,
  onProgress: (p: number) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url, true);
    xhr.setRequestHeader("Content-Type", contentType);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => (xhr.status >= 200 && xhr.status < 300 ? resolve() : reject(new Error(`PUT ${xhr.status}`)));
    xhr.onerror = () => reject(new Error("network error"));
    xhr.send(file);
  });
}

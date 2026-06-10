import { useEffect, useMemo, useState } from "react";
import { api, sha256File } from "../api.ts";
import { manualApi } from "./api.ts";

/**
 * Per-stage test harness. Five doors, five tables.
 *
 *   Upload → Door 1 Separate → Table 1 → Door 2 Transcribe → Table 2 →
 *   Door 3 Align → Table 3 → Door 4 Compose → Table 4 →
 *   Door 5 Record-mix → Table 5.
 *
 * Every door's inputs are editable. Paste a gs:// URI in any slot to skip
 * earlier stages. The upload widget only runs the bucket PUT; the job_id
 * and stage URIs thread through doors via component state.
 */

// ---------- shared UI primitives ----------

function JsonDump({ value }: { value: unknown }) {
  return <pre className="json-dump">{JSON.stringify(value, null, 2)}</pre>;
}

function SignedAudio({ uri, label }: { uri?: string; label: string }) {
  const [src, setSrc] = useState<string | undefined>();
  const [err, setErr] = useState<string | undefined>();
  useEffect(() => {
    if (!uri) { setSrc(undefined); return; }
    manualApi.signedGet(uri)
      .then((r) => setSrc(r.url))
      .catch((e) => setErr(String(e)));
  }, [uri]);
  if (!uri) return null;
  return (
    <div className="media-row">
      <div className="media-label">{label}</div>
      {err && <div className="small err">{err}</div>}
      {src && <audio controls src={src} />}
      <div className="small muted mono truncate">{uri}</div>
    </div>
  );
}

function SignedVideo({ uri, label }: { uri?: string; label: string }) {
  const [src, setSrc] = useState<string | undefined>();
  useEffect(() => {
    if (!uri) { setSrc(undefined); return; }
    manualApi.signedGet(uri).then((r) => setSrc(r.url));
  }, [uri]);
  if (!uri) return null;
  return (
    <div className="media-row">
      <div className="media-label">{label}</div>
      {src && <video controls src={src} style={{ width: "100%", maxHeight: 360 }} />}
      <div className="small muted mono truncate">{uri}</div>
    </div>
  );
}

function SignedText({ uri, label }: { uri?: string; label: string }) {
  const [text, setText] = useState<string | undefined>();
  useEffect(() => {
    if (!uri) { setText(undefined); return; }
    manualApi.signedGet(uri)
      .then((r) => fetch(r.url).then((res) => res.text()))
      .then(setText)
      .catch((e) => setText(`[fetch failed: ${e}]`));
  }, [uri]);
  if (!uri) return null;
  return (
    <div className="media-row">
      <div className="media-label">{label}</div>
      <textarea className="ass-dump" readOnly value={text ?? ""} />
      <div className="small muted mono truncate">{uri}</div>
    </div>
  );
}

type Vad = { start: number; end: number; kind: "vocals" | "instrumental" };

function VadTimeline({ regions }: { regions: Vad[] }) {
  if (!regions.length) return <div className="small muted">(no regions)</div>;
  const total = regions[regions.length - 1]!.end - regions[0]!.start || 1;
  return (
    <div className="vad-row" title="Vocals (green) / instrumental (grey)">
      {regions.map((r, i) => (
        <div
          key={i}
          className={`vad-seg ${r.kind}`}
          style={{ flexGrow: Math.max(0.01, (r.end - r.start) / total) }}
          title={`${r.kind}: ${r.start.toFixed(1)}s–${r.end.toFixed(1)}s`}
        />
      ))}
    </div>
  );
}

// ---------- upload entry ----------

type UploadState = { source_uri: string; sha256: string; content_type: string };

function UploadEntry({
  username,
  onUploaded,
}: {
  username: string;
  onUploaded: (u: UploadState) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState<string | undefined>();
  const [manualUri, setManualUri] = useState("");

  async function doUpload() {
    if (!file) return;
    setRunning(true);
    setErr(undefined);
    try {
      const sha = await sha256File(file, (p) => setProgress(p * 0.3));
      const r = await api.requestUpload({
        username,
        sha256: sha,
        size: file.size,
        content_type: file.type || "video/mp4",
      });
      if (r.need_upload && r.signed_put_url) {
        await putFile(r.signed_put_url, file, (p) => setProgress(0.3 + p * 0.7));
      }
      setProgress(1);
      // /uploads now echoes the full gs:// URI so the UI doesn't need to
      // know the bucket name. Dev mode (DEV_FS_ROOT) returns "gs://mock/..."
      // which /manual/signed-get treats as a local-fs path.
      const source_uri: string = r.gs_uri ?? `gs://${r.bucket ?? "mock"}/${r.object_path}`;
      onUploaded({ source_uri, sha256: sha, content_type: r.content_type ?? file.type });
    } catch (e) {
      setErr(String(e));
    } finally {
      setRunning(false);
    }
  }

  function submitManualUri(e: React.FormEvent) {
    e.preventDefault();
    const u = manualUri.trim();
    if (!u.startsWith("gs://")) return setErr("expected gs://… URI");
    onUploaded({ source_uri: u, sha256: "pasted", content_type: "unknown" });
  }

  return (
    <section className="panel">
      <h3>Entrance · source video</h3>
      <label>Upload a video</label>
      <input
        type="file"
        accept="video/*"
        onChange={(e) => setFile(e.currentTarget.files?.[0] ?? null)}
        disabled={running}
      />
      {progress > 0 && (
        <div className="progress"><div style={{ width: `${Math.round(progress * 100)}%` }} /></div>
      )}
      <button onClick={doUpload} disabled={!file || running}>
        {running ? "uploading…" : "upload → get source_uri"}
      </button>
      <div className="divider" />
      <form onSubmit={submitManualUri}>
        <label>…or paste a gs:// source URI (skip upload)</label>
        <input
          type="text"
          placeholder="gs://your-bucket/uploads/...mp4"
          value={manualUri}
          onChange={(e) => setManualUri(e.currentTarget.value)}
        />
        <button type="submit" className="ghost">use this URI</button>
      </form>
      {err && <div className="status failed">{err}</div>}
    </section>
  );
}

async function putFile(url: string, file: File, onProgress: (p: number) => void): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);
    xhr.setRequestHeader("content-type", file.type || "application/octet-stream");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => (xhr.status >= 200 && xhr.status < 300 ? resolve() : reject(new Error(`put ${xhr.status}`)));
    xhr.onerror = () => reject(new Error("put failed"));
    xhr.send(file);
  });
}

// ---------- door primitive ----------

type DoorProps<TReq, TRes> = {
  title: string;
  stage: import("./api.ts").StageName;
  fields: Array<{
    key: keyof TReq & string;
    label: string;
    placeholder?: string;
    type?: "text" | "number" | "select";
    options?: string[];
    optional?: boolean;
  }>;
  initial: Partial<TReq>;
  onResult: (res: TRes) => void;
};

function Door<TReq extends Record<string, unknown>, TRes>({
  title, stage, fields, initial, onResult,
}: DoorProps<TReq, TRes>) {
  const [values, setValues] = useState<Record<string, unknown>>(initial as Record<string, unknown>);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState<string | undefined>();
  const [response, setResponse] = useState<TRes | undefined>();

  // Re-seed inputs when parent changes them (e.g., after a "Send to next door").
  useEffect(() => {
    setValues((prev) => ({ ...prev, ...(initial as Record<string, unknown>) }));
  }, [JSON.stringify(initial)]);

  async function fire() {
    setRunning(true);
    setErr(undefined);
    setResponse(undefined);
    try {
      // Clean empty optional fields so the stage's schema doesn't get "" where
      // it expects undefined.
      const body: Record<string, unknown> = {};
      for (const f of fields) {
        const v = values[f.key];
        if (v === "" || v == null) {
          if (!f.optional) throw new Error(`missing: ${f.label}`);
          continue;
        }
        body[f.key] = f.type === "number" ? Number(v) : v;
      }
      const res = await manualApi.callStage<TRes>(stage, body);
      setResponse(res);
      onResult(res);
    } catch (e) {
      setErr(String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <section className="panel door">
      <h3>🚪 {title}</h3>
      {fields.map((f) => (
        <div key={f.key}>
          <label>{f.label}{f.optional ? " (optional)" : ""}</label>
          {f.type === "select" ? (
            <select
              value={(values[f.key] as string) ?? ""}
              onChange={(e) => setValues((p) => ({ ...p, [f.key]: e.currentTarget.value }))}
            >
              <option value="">(default)</option>
              {(f.options ?? []).map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          ) : (
            <input
              type={f.type === "number" ? "number" : "text"}
              placeholder={f.placeholder}
              value={(values[f.key] as string | number) ?? ""}
              onChange={(e) => setValues((p) => ({ ...p, [f.key]: e.currentTarget.value }))}
            />
          )}
        </div>
      ))}
      <button onClick={fire} disabled={running}>
        {running ? "running…" : "open →"}
      </button>
      {err && <div className="status failed">{err}</div>}
      {response != null && (
        <details className="response-details"><summary>raw response</summary><JsonDump value={response} /></details>
      )}
    </section>
  );
}

// ---------- page ----------

type SeparateRes = {
  vocals_uri: string;
  instrumental_uri: string;
  sample_rate: number;
  model_used: string;
};
type TranscribeRes = {
  language: string;
  segments: Array<{ text: string; start: number; end: number }>;
  vocal_activity: Vad[];
  source: string;
  model_used: string;
};
type AlignRes = {
  words: Array<{ text: string; start: number; end: number; score?: number }>;
  vocal_activity: Vad[];
  source: string;
  model_used: string;
};
type ComposeRes = {
  manifest_uri: string;
  manifest_url: string;
  ass_uri: string;
};
type RecordMixRes = { mix_uri: string };

export function Manual() {
  const username = useMemo(() => localStorage.getItem("annemusic.username") ?? "", []);
  const [upload, setUpload] = useState<UploadState | null>(null);
  const [jobId, setJobId] = useState<string>("");
  const [separate, setSeparate] = useState<SeparateRes | undefined>();
  const [transcribe, setTranscribe] = useState<TranscribeRes | undefined>();
  const [align, setAlign] = useState<AlignRes | undefined>();
  const [compose, setCompose] = useState<ComposeRes | undefined>();
  const [recordMix, setRecordMix] = useState<RecordMixRes | undefined>();
  const [recording, setRecording] = useState<string>(""); // gs:// uri for user recording

  useEffect(() => {
    manualApi.newJobId().then((r) => setJobId(r.job_id)).catch(() => {});
  }, []);

  if (!username) {
    return (
      <main>
        <h1>annemusic · manual 🚪</h1>
        <p className="subtitle">
          No <code>annemusic.username</code> in localStorage. Register on the
          main app first: <a href="/">open /</a>.
        </p>
      </main>
    );
  }

  return (
    <main className="manual">
      <h1>annemusic · manual 🚪</h1>
      <p className="subtitle">
        5 doors. 5 tables. Observe each stage's output, then advance.
      </p>

      <section className="panel">
        <div className="row">
          <span>user: <strong>{username}</strong> · job: <code>{jobId || "…"}</code></span>
          <a className="ghost-link" href="/">← main app</a>
        </div>
      </section>

      <UploadEntry username={username} onUploaded={setUpload} />

      {upload && (
        <>
          <Door<
            { job_id: string; source_uri: string; model?: string },
            SeparateRes
          >
            title="Door 1 · Separate"
            stage="separate"
            fields={[
              { key: "job_id", label: "job_id" },
              { key: "source_uri", label: "source_uri" },
              {
                key: "model",
                label: "model",
                type: "select",
                options: [
                  "mel_band_roformer_kim",
                  "bs_roformer_ep317",
                  "htdemucs",
                  "htdemucs_ft",
                  "htdemucs_6s",
                ],
                optional: true,
              },
            ]}
            initial={{ job_id: jobId, source_uri: upload.source_uri }}
            onResult={setSeparate}
          />

          {separate && (
            <section className="panel table">
              <h3>🪑 Table 1 · separate output</h3>
              <SignedAudio uri={separate.vocals_uri} label="vocals.wav" />
              <SignedAudio uri={separate.instrumental_uri} label="instrumental (no_vocals).wav" />
              <div className="small muted">
                sample_rate {separate.sample_rate} · model {separate.model_used}
              </div>
            </section>
          )}
        </>
      )}

      {separate && (
        <>
          <Door<
            { job_id: string; vocals_uri: string; source_uri?: string; language?: string; known_lyrics?: string },
            TranscribeRes
          >
            title="Door 2 · Transcribe"
            stage="transcribe"
            fields={[
              { key: "job_id", label: "job_id" },
              { key: "vocals_uri", label: "vocals_uri (from Table 1)" },
              { key: "source_uri", label: "source_uri (original upload)", optional: true },
              { key: "language", label: "language (ISO 639-1/3)", placeholder: "tr", optional: true },
              { key: "known_lyrics", label: "known_lyrics (bias)", optional: true },
            ]}
            initial={{
              job_id: jobId,
              vocals_uri: separate.vocals_uri,
              source_uri: upload?.source_uri,
              language: "tr",
            }}
            onResult={setTranscribe}
          />

          {transcribe && (
            <section className="panel table">
              <h3>🪑 Table 2 · transcribe output</h3>
              <div className="small muted">
                language {transcribe.language} · source {transcribe.source} · model {transcribe.model_used}
              </div>
              <div className="media-label">vocal_activity</div>
              <VadTimeline regions={transcribe.vocal_activity} />
              <div className="media-label">segments ({transcribe.segments.length})</div>
              <ul className="segments">
                {transcribe.segments.map((s, i) => (
                  <li key={i}>
                    <span className="ts">{s.start.toFixed(1)}–{s.end.toFixed(1)}s</span>
                    <span>{s.text}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}

      {transcribe && (
        <>
          <Door<
            { job_id: string; vocals_uri: string; segments: unknown; language: string; vocal_activity: unknown },
            AlignRes
          >
            title="Door 3 · Align"
            stage="align"
            fields={[
              { key: "job_id", label: "job_id" },
              { key: "vocals_uri", label: "vocals_uri" },
              { key: "language", label: "language" },
            ]}
            initial={{
              job_id: jobId,
              vocals_uri: separate?.vocals_uri,
              language: transcribe.language,
              // segments + vocal_activity are complex — inject directly rather
              // than routing through a text input.
              segments: transcribe.segments,
              vocal_activity: transcribe.vocal_activity,
            }}
            onResult={setAlign}
          />

          {align && (
            <section className="panel table">
              <h3>🪑 Table 3 · align output</h3>
              <div className="small muted">
                source {align.source} · model {align.model_used} · {align.words.length} words
              </div>
              <div className="words">
                {align.words.slice(0, 200).map((w, i) => {
                  const conf = w.score ?? 1;
                  const opacity = 0.4 + conf * 0.6;
                  return (
                    <span key={i} className="word" style={{ opacity }} title={`${w.start.toFixed(2)}–${w.end.toFixed(2)}s score=${conf.toFixed(2)}`}>
                      {w.text}
                    </span>
                  );
                })}
                {align.words.length > 200 && (
                  <span className="small muted"> +{align.words.length - 200} more…</span>
                )}
              </div>
            </section>
          )}
        </>
      )}

      {align && upload && (
        <>
          <Door<
            {
              job_id: string; words: unknown; video_uri: string;
              instrumental_uri: string; language?: string; vocal_activity: unknown;
            },
            ComposeRes
          >
            title="Door 4 · Compose"
            stage="compose"
            fields={[
              { key: "job_id", label: "job_id" },
              { key: "video_uri", label: "video_uri (upload)" },
              { key: "instrumental_uri", label: "instrumental_uri" },
              { key: "language", label: "language", optional: true },
            ]}
            initial={{
              job_id: jobId,
              words: align.words,
              video_uri: upload.source_uri,
              instrumental_uri: separate?.instrumental_uri,
              language: transcribe?.language,
              vocal_activity: align.vocal_activity,
            }}
            onResult={setCompose}
          />

          {compose && (
            <section className="panel table">
              <h3>🪑 Table 4 · compose output</h3>
              <SignedVideo uri={upload.source_uri} label="source video" />
              <SignedAudio uri={separate?.instrumental_uri} label="instrumental for playback" />
              <SignedText uri={compose.ass_uri} label=".ass (preview as text — JASSUB overlay TBD)" />
              <a
                className="ghost-link"
                href={compose.manifest_url}
                target="_blank"
                rel="noreferrer"
              >open manifest.json ↗</a>
            </section>
          )}
        </>
      )}

      {compose && separate && (
        <>
          <section className="panel door">
            <h3>🚪 Door 5 · Record-mix</h3>
            <div className="small muted">
              Needs a user recording (separate upload). Paste a gs:// URI for the
              recording; everything else prefills from Table 1 / this job.
            </div>
            <label>recording_uri</label>
            <input
              type="text"
              placeholder="gs://your-bucket/recordings/..."
              value={recording}
              onChange={(e) => setRecording(e.currentTarget.value)}
            />
            <RecordMixButton
              jobId={jobId}
              recordingUri={recording}
              instrumentalUri={separate.instrumental_uri}
              vocalsUri={separate.vocals_uri}
              onResult={setRecordMix}
            />
          </section>

          {recordMix && (
            <section className="panel table">
              <h3>🪑 Table 5 · record-mix output</h3>
              <SignedAudio uri={recordMix.mix_uri} label="mix.wav" />
            </section>
          )}
        </>
      )}
    </main>
  );
}

function RecordMixButton({
  jobId, recordingUri, instrumentalUri, vocalsUri, onResult,
}: {
  jobId: string; recordingUri: string; instrumentalUri: string; vocalsUri: string;
  onResult: (r: RecordMixRes) => void;
}) {
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState<string | undefined>();
  async function fire() {
    if (!recordingUri.startsWith("gs://")) return setErr("recording_uri must be gs://…");
    setRunning(true); setErr(undefined);
    try {
      const res = await manualApi.callStage<RecordMixRes>("record-mix", {
        job_id: jobId,
        recording_uri: recordingUri,
        instrumental_uri: instrumentalUri,
        vocals_uri: vocalsUri,
      });
      onResult(res);
    } catch (e) {
      setErr(String(e));
    } finally {
      setRunning(false);
    }
  }
  return (
    <>
      <button onClick={fire} disabled={running || !recordingUri}>
        {running ? "mixing…" : "open →"}
      </button>
      {err && <div className="status failed">{err}</div>}
    </>
  );
}

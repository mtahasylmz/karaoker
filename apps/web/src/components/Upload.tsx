import { useState } from "react";
import { api, putWithProgress, sha256File } from "../api.ts";

const MAX_BYTES = 500 * 1024 * 1024;

export function Upload({
  username,
  onJobCreated,
}: {
  username: string;
  onJobCreated: (jobId: string) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [artist, setArtist] = useState("");
  const [hashPct, setHashPct] = useState<number | null>(null);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [statusClass, setStatusClass] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit() {
    if (!file) return;
    if (file.size > MAX_BYTES) {
      setStatus(`File too big (${(file.size / 1024 / 1024).toFixed(0)} MB, max 500)`);
      setStatusClass("failed");
      return;
    }
    setBusy(true);
    setStatus("Hashing file…");
    setStatusClass("running");
    try {
      const sha256 = await sha256File(file, (p) => setHashPct(p));
      setHashPct(null);

      setStatus("Checking cache…");
      const u = await api.requestUpload({
        username,
        sha256,
        size: file.size,
        content_type: file.type,
        title: title || undefined,
        artist: artist || undefined,
      });

      if (u.need_upload && u.signed_put_url) {
        setStatus("Uploading…");
        await putWithProgress(u.signed_put_url, file, file.type, (p) => setUploadPct(p));
        setUploadPct(null);
      }

      setStatus("Queuing job…");
      const j = await api.createJob(username, sha256);
      if (j.job_id) {
        onJobCreated(j.job_id);
      } else if (j.status === "done") {
        // TODO: show cached manifest playback in JobView
        onJobCreated("cached-" + sha256.slice(0, 12));
      }
    } catch (e) {
      setStatus((e as Error).message);
      setStatusClass("failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel">
      <label htmlFor="file">
        Pick a music video (mp4, mov, webm, mkv — up to 500 MB)
      </label>
      <input
        id="file"
        type="file"
        accept="video/mp4,video/quicktime,video/webm,video/x-matroska"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
      />

      <div style={{ marginTop: "0.75rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
        <input
          type="text"
          placeholder="Song title (optional — helps lyrics lookup)"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />
        <input
          type="text"
          placeholder="Artist (optional)"
          value={artist}
          onChange={(e) => setArtist(e.target.value)}
        />
      </div>

      <button onClick={submit} disabled={busy || !file}>
        Karaoke it
      </button>
      <button
        className="ghost"
        style={{ marginLeft: "0.5rem" }}
        onClick={async () => {
          setBusy(true);
          setStatus("Triggering stub job…");
          setStatusClass("running");
          try {
            const r = await api.devTrigger(username, title || undefined, artist || undefined);
            onJobCreated(r.job_id);
          } catch (e) {
            setStatus((e as Error).message);
            setStatusClass("failed");
          } finally {
            setBusy(false);
          }
        }}
        disabled={busy}
        title="Phase C: trigger the pipeline with a synthetic job, no file needed"
      >
        or: trigger stub flow
      </button>

      {hashPct !== null && (
        <div className="progress">
          <div style={{ width: `${(hashPct * 100).toFixed(0)}%` }} />
        </div>
      )}
      {uploadPct !== null && (
        <div className="progress">
          <div style={{ width: `${(uploadPct * 100).toFixed(0)}%` }} />
        </div>
      )}
      {status && <div className={`status ${statusClass}`}>{status}</div>}
    </section>
  );
}

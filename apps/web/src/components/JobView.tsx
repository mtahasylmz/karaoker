import { useEffect, useRef, useState } from "react";
import { api } from "../api.ts";

type LogEntry = {
  ts: number;
  stage: string;
  job_id?: string;
  level: "debug" | "info" | "warn" | "error";
  msg: string;
  data?: unknown;
  err?: unknown;
};

type Job = {
  job_id: string;
  sha256: string;
  status: string;
  created_at: string;
  updated_at: string;
  manifest_url?: string;
  error?: string;
  logs?: LogEntry[];
};

const STAGES = ["separate", "transcribe", "align", "compose"] as const;
type Stage = (typeof STAGES)[number];

type StageState = "pending" | "running" | "done" | "failed";

function deriveStageStates(logs: LogEntry[]): Record<Stage, StageState> {
  const state: Record<Stage, StageState> = {
    separate: "pending",
    transcribe: "pending",
    align: "pending",
    compose: "pending",
  };
  for (const l of logs) {
    if (!STAGES.includes(l.stage as Stage)) continue;
    const s = l.stage as Stage;
    if (l.level === "error") state[s] = "failed";
    else if (/start|starting|begin/i.test(l.msg) && state[s] === "pending") state[s] = "running";
    else if (/complete|done|finish/i.test(l.msg)) state[s] = "done";
  }
  return state;
}

function fmtTs(ts: number) {
  const d = new Date(ts);
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
}

export function JobView({ jobId, onReset }: { jobId: string; onReset: () => void }) {
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const logsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const j = (await api.getJob(jobId)) as Job;
        if (cancelled) return;
        setJob(j);
        setErr(null);
        // keep polling until terminal
        if (j.status !== "done" && j.status !== "failed") {
          setTimeout(poll, 1500);
        }
      } catch (e) {
        if (cancelled) return;
        setErr((e as Error).message);
        setTimeout(poll, 3000);
      }
    }
    poll();
    return () => { cancelled = true; };
  }, [jobId]);

  // auto-scroll logs
  useEffect(() => {
    if (logsRef.current) logsRef.current.scrollTop = logsRef.current.scrollHeight;
  }, [job?.logs?.length]);

  if (err && !job) return <section className="panel status failed">{err}</section>;
  if (!job) return <section className="panel">Loading…</section>;

  const states = deriveStageStates(job.logs ?? []);

  const overallClass =
    job.status === "done" ? "done" :
    job.status === "failed" ? "failed" : "running";

  return (
    <>
      <section className="panel">
        <div className="row">
          <div>
            <div className="small">Job {job.job_id}</div>
            <div className={`status ${overallClass}`}>
              Status: <strong>{job.status}</strong>
            </div>
          </div>
          <button className="ghost" onClick={onReset}>new upload</button>
        </div>

        <div className="pipeline">
          {STAGES.map((s) => (
            <div key={s} className={`stage ${states[s]}`}>
              <span className="dot" />
              <div className="name">{s}</div>
              <div className="small">{states[s]}</div>
            </div>
          ))}
        </div>

        {job.manifest_url && (
          <div className="small">
            Manifest:{" "}
            <a href={job.manifest_url} target="_blank" rel="noreferrer">
              {job.manifest_url}
            </a>
          </div>
        )}
        {job.error && <div className="status failed">{job.error}</div>}
      </section>

      <section className="panel">
        <div className="small" style={{ marginBottom: "0.5rem" }}>
          Live logs ({job.logs?.length ?? 0})
        </div>
        <div className="logs" ref={logsRef}>
          {(job.logs ?? []).map((l, i) => (
            <div className="log-line" key={i}>
              <span className="ts">{fmtTs(l.ts)}</span>
              <span className="stage">{l.stage}</span>
              <span className={`level-${l.level}`}>{l.level}</span>
              <span className="msg">
                {l.msg}
                {l.data != null && (
                  <span className="data"> {JSON.stringify(l.data)}</span>
                )}
              </span>
            </div>
          ))}
          {(job.logs ?? []).length === 0 && (
            <div className="log-line">
              <span className="msg small">(waiting for first event…)</span>
            </div>
          )}
        </div>
      </section>
    </>
  );
}

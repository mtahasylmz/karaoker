import { useState } from "react";
import { api } from "../api.ts";

export function Register({ onRegistered }: { onRegistered: (u: string) => void }) {
  const [username, setUsername] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit() {
    const u = username.trim();
    if (!u) return;
    setBusy(true);
    setErr(null);
    try {
      await api.registerUser(u);
      onRegistered(u);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel">
      <label htmlFor="username">
        Pick a username (2–24 chars, letters/digits/_.-)
      </label>
      <input
        id="username"
        type="text"
        autoComplete="off"
        placeholder="e.g. taha"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && submit()}
      />
      <button onClick={submit} disabled={busy}>
        Continue
      </button>
      {err && <div className="status failed">{err}</div>}
    </section>
  );
}

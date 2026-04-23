import { useEffect, useState } from "react";
import { Register } from "./components/Register.tsx";
import { Upload } from "./components/Upload.tsx";
import { JobView } from "./components/JobView.tsx";
import { api } from "./api.ts";

const USER_KEY = "annemusic.username";

export function App() {
  const [username, setUsername] = useState<string | null>(
    () => localStorage.getItem(USER_KEY),
  );
  const [booted, setBooted] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  useEffect(() => {
    if (!username) {
      setBooted(true);
      return;
    }
    api
      .getUser(username)
      .then(() => setBooted(true))
      .catch(() => {
        localStorage.removeItem(USER_KEY);
        setUsername(null);
        setBooted(true);
      });
  }, [username]);

  if (!booted) return null;

  function handleRegistered(u: string) {
    localStorage.setItem(USER_KEY, u);
    setUsername(u);
  }

  function logout() {
    localStorage.removeItem(USER_KEY);
    setUsername(null);
    setActiveJobId(null);
  }

  return (
    <main>
      <h1>annemusic 🎤</h1>
      <p className="subtitle">
        Upload a music video, watch each stage process it in real time.
      </p>

      {!username && <Register onRegistered={handleRegistered} />}

      {username && (
        <section className="panel">
          <div className="row">
            <span>
              Signed in as <strong>{username}</strong>
            </span>
            <button className="ghost" onClick={logout}>
              switch user
            </button>
          </div>
        </section>
      )}

      {username && !activeJobId && (
        <Upload username={username} onJobCreated={setActiveJobId} />
      )}

      {username && activeJobId && (
        <JobView
          jobId={activeJobId}
          onReset={() => setActiveJobId(null)}
        />
      )}
    </main>
  );
}

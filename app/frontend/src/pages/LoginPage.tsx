import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    const ok = await login(username, password);
    if (ok) {
      navigate("/chat");
    } else {
      setError("Invalid username or password");
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center">
      <form
        onSubmit={handleSubmit}
        className="stamp-shadow w-80 rounded border-2 border-rule bg-surface p-7"
      >
        <p className="text-center font-wire text-[10px] uppercase tracking-[0.3em] text-ink-muted">
          Est. Today · Vol. I
        </p>
        <h1 className="mb-1 text-center font-display text-3xl text-ink">The NewsQA</h1>
        <p className="mb-6 text-center font-wire text-[10px] uppercase tracking-wide text-moss">
          Retrieval-Augmented Desk
        </p>

        <label className="mb-1 block font-wire text-[10px] uppercase tracking-wide text-ink-muted">
          Username
        </label>
        <input
          className="mb-4 w-full rounded border-2 border-rule bg-paper px-3 py-2 text-sm text-ink outline-none focus:border-accent"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <label className="mb-1 block font-wire text-[10px] uppercase tracking-wide text-ink-muted">
          Password
        </label>
        <input
          type="password"
          className="mb-5 w-full rounded border-2 border-rule bg-paper px-3 py-2 text-sm text-ink outline-none focus:border-accent"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {error && <p className="mb-3 text-sm text-accent">{error}</p>}
        <button
          type="submit"
          className="w-full rounded bg-accent px-3 py-2 font-wire text-xs uppercase tracking-wide text-surface hover:bg-accent-hover"
        >
          Enter the Newsroom
        </button>
      </form>
    </div>
  );
}

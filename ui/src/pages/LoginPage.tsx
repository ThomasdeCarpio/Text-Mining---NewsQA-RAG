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
    <div className="flex min-h-screen items-center justify-center bg-gray-50">
      <form onSubmit={handleSubmit} className="w-80 rounded-lg border border-gray-200 p-6 shadow-sm">
        <h1 className="mb-4 text-xl font-semibold">NewsQA-RAG Login</h1>
        <label className="mb-1 block text-sm text-gray-600">Username</label>
        <input
          className="mb-3 w-full rounded border border-gray-300 px-3 py-2 text-sm"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <label className="mb-1 block text-sm text-gray-600">Password</label>
        <input
          type="password"
          className="mb-4 w-full rounded border border-gray-300 px-3 py-2 text-sm"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {error && <p className="mb-3 text-sm text-red-600">{error}</p>}
        <button
          type="submit"
          className="w-full rounded bg-purple-600 px-3 py-2 text-sm font-medium text-white hover:bg-purple-700"
        >
          Log in
        </button>
      </form>
    </div>
  );
}

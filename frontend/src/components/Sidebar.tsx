import { NavLink } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export function Sidebar() {
  const { user, logout } = useAuth();
  if (!user) return null;

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `block rounded px-3 py-2 font-wire text-xs uppercase tracking-wide ${
      isActive
        ? "bg-accent text-surface"
        : "text-ink-muted hover:bg-paper-dark hover:text-ink"
    }`;

  return (
    <aside className="flex w-60 shrink-0 flex-col border-r-2 border-rule bg-surface p-5">
      <div className="mb-8 border-b-2 border-rule pb-4">
        <p className="font-display text-2xl leading-none text-ink">The NewsQA</p>
        <p className="font-wire text-[10px] uppercase tracking-[0.2em] text-ink-muted">
          Late Edition · RAG Desk
        </p>
      </div>

      <div className="mb-6">
        <p className="font-wire text-[10px] uppercase tracking-wide text-ink-muted">Signed in as</p>
        <p className="font-display text-lg text-ink">{user.username}</p>
        <p className="font-wire text-[10px] uppercase tracking-wide text-moss">{user.role}</p>
      </div>

      <nav className="flex flex-col gap-1">
        <NavLink to="/chat" className={linkClass}>
          💬 News Chat
        </NavLink>
        {user.role === "admin" && (
          <>
            <NavLink to="/dashboard" className={linkClass}>
              📊 Evaluation Desk
            </NavLink>
            <NavLink to="/retrieval" className={linkClass}>
              🔍 Retrieval Playground
            </NavLink>
          </>
        )}
      </nav>

      <button
        onClick={logout}
        className="stamp-shadow mt-auto rounded border-2 border-rule bg-paper px-3 py-2 font-wire text-xs uppercase tracking-wide text-ink hover:border-accent hover:text-accent"
      >
        Log out
      </button>
    </aside>
  );
}

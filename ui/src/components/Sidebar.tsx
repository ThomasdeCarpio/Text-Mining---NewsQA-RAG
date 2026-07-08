import { NavLink } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export function Sidebar() {
  const { user, logout } = useAuth();
  if (!user) return null;

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `block rounded px-3 py-2 text-sm font-medium ${
      isActive ? "bg-purple-100 text-purple-800" : "text-gray-600 hover:bg-gray-100"
    }`;

  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-gray-200 p-4">
      <div className="mb-6">
        <p className="text-sm text-gray-500">Logged in as</p>
        <p className="font-semibold">{user.username}</p>
        <p className="text-xs text-gray-400">{user.role}</p>
      </div>

      <nav className="flex flex-col gap-1">
        <NavLink to="/chat" className={linkClass}>
          💬 News Chat
        </NavLink>
        {user.role === "admin" && (
          <NavLink to="/dashboard" className={linkClass}>
            📊 Evaluation Dashboard
          </NavLink>
        )}
      </nav>

      <button
        onClick={logout}
        className="mt-auto rounded border border-gray-300 px-3 py-2 text-sm text-gray-700 hover:bg-gray-100"
      >
        Logout
      </button>
    </aside>
  );
}

import { ExperimentPanel } from "../components/ExperimentPanel";

export function DashboardPage() {
  return (
    <main className="flex-1 overflow-y-auto p-6">
      <h1 className="font-display text-2xl text-ink">Evaluation Desk</h1>
      <p className="mb-6 mt-1 text-sm text-ink-muted">
        Preview a locked experiment, run missing work, or load its saved report.
      </p>
      <ExperimentPanel />
    </main>
  );
}

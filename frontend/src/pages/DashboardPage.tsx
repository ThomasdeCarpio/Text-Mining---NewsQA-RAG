import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import {
  getFailureCases,
  getMetrics,
  getPipelineLogs,
  getSearchComparison,
  triggerCrawler,
} from "../api/client";
import type { AgentEvent, FailureCase, SearchComparisonRow } from "../api/types";
import { MetricCard } from "../components/MetricCard";

export function DashboardPage() {
  const [metrics, setMetrics] = useState<Record<string, number>>({});
  const [comparison, setComparison] = useState<SearchComparisonRow[]>([]);
  const [failures, setFailures] = useState<FailureCase[]>([]);
  const [logs, setLogs] = useState<AgentEvent[]>([]);
  const [crawlerMessage, setCrawlerMessage] = useState<string | null>(null);

  useEffect(() => {
    getMetrics().then(setMetrics);
    getSearchComparison().then(setComparison);
    getFailureCases().then(setFailures);
    getPipelineLogs().then(setLogs);
  }, []);

  async function handleTrigger() {
    const triggered = await triggerCrawler();
    setCrawlerMessage(triggered ? "Crawler triggered (mock)." : "Failed to trigger crawler.");
  }

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <h1 className="mb-6 font-display text-2xl text-ink">Evaluation Desk</h1>

      <div className="mb-8 grid grid-cols-2 gap-4 md:grid-cols-4">
        {Object.entries(metrics).map(([name, value]) => (
          <MetricCard key={name} name={name} value={value} />
        ))}
      </div>

      <h2 className="mb-2 font-wire text-[10px] uppercase tracking-wide text-ink-muted">
        Vector Search vs Hybrid Search
      </h2>
      <div className="stamp-shadow mb-8 h-64 rounded border-2 border-rule bg-surface p-2">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={comparison}>
            <CartesianGrid strokeDasharray="3 3" stroke="#d8cbb0" />
            <XAxis dataKey="metric" tick={{ fontSize: 12, fill: "#6b6355" }} />
            <YAxis domain={[0, 1]} tick={{ fill: "#6b6355" }} />
            <Tooltip contentStyle={{ background: "#fffcf5", border: "2px solid #d8cbb0" }} />
            <Legend />
            <Bar dataKey="vector_search" name="Vector Search" fill="#5f6b4a" />
            <Bar dataKey="hybrid_search" name="Hybrid Search" fill="#a13d2b" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <h2 className="mb-2 font-wire text-[10px] uppercase tracking-wide text-ink-muted">Failure Analysis</h2>
      <div className="stamp-shadow mb-8 overflow-x-auto rounded border-2 border-rule bg-surface">
        <table className="w-full text-left text-sm">
          <thead className="border-b-2 border-rule bg-paper font-wire text-[10px] uppercase tracking-wide text-ink-muted">
            <tr>
              <th className="p-2">Question</th>
              <th className="p-2">Expected</th>
              <th className="p-2">Retrieved</th>
              <th className="p-2">Reason</th>
            </tr>
          </thead>
          <tbody>
            {failures.map((f, i) => (
              <tr key={i} className="border-t border-rule text-ink">
                <td className="p-2">{f.question}</td>
                <td className="p-2">{f.expected}</td>
                <td className="p-2">{f.retrieved}</td>
                <td className="p-2 text-ink-muted">{f.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2 className="mb-2 font-wire text-[10px] uppercase tracking-wide text-ink-muted">
        Pipeline Logs (Agent Thought Process)
      </h2>
      <div className="stamp-shadow mb-8 max-h-48 overflow-y-auto rounded border-2 border-rule bg-surface p-2 font-wire text-xs text-ink-muted">
        {logs.length === 0 && <p className="text-ink-muted">No agent activity recorded yet.</p>}
        {logs.map((event, i) => (
          <p key={i}>
            [{event.timestamp}] {event.type}
            {event.tool_name ? ` [${event.tool_name}]` : ""}: {event.content}
          </p>
        ))}
      </div>

      <h2 className="mb-2 font-wire text-[10px] uppercase tracking-wide text-ink-muted">Manual Crawler Trigger</h2>
      <button
        onClick={handleTrigger}
        className="rounded bg-accent px-4 py-2 font-wire text-xs uppercase tracking-wide text-surface hover:bg-accent-hover"
      >
        Trigger Crawler
      </button>
      {crawlerMessage && <p className="mt-2 text-sm text-moss">{crawlerMessage}</p>}
    </div>
  );
}

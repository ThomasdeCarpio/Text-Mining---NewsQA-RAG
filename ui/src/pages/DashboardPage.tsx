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
      <h1 className="mb-4 text-lg font-semibold">Evaluation Dashboard</h1>

      <div className="mb-6 grid grid-cols-2 gap-4 md:grid-cols-4">
        {Object.entries(metrics).map(([name, value]) => (
          <MetricCard key={name} name={name} value={value} />
        ))}
      </div>

      <h2 className="mb-2 text-sm font-semibold text-gray-700">Vector Search vs Hybrid Search</h2>
      <div className="mb-6 h-64 rounded border border-gray-200 p-2">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={comparison}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="metric" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Legend />
            <Bar dataKey="vector_search" name="Vector Search" fill="#a78bfa" />
            <Bar dataKey="hybrid_search" name="Hybrid Search" fill="#7c3aed" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <h2 className="mb-2 text-sm font-semibold text-gray-700">Failure Analysis</h2>
      <div className="mb-6 overflow-x-auto rounded border border-gray-200">
        <table className="w-full text-left text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="p-2">Question</th>
              <th className="p-2">Expected</th>
              <th className="p-2">Retrieved</th>
              <th className="p-2">Reason</th>
            </tr>
          </thead>
          <tbody>
            {failures.map((f, i) => (
              <tr key={i} className="border-t border-gray-100">
                <td className="p-2">{f.question}</td>
                <td className="p-2">{f.expected}</td>
                <td className="p-2">{f.retrieved}</td>
                <td className="p-2">{f.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2 className="mb-2 text-sm font-semibold text-gray-700">Pipeline Logs (Agent Thought Process)</h2>
      <div className="mb-6 max-h-48 overflow-y-auto rounded border border-gray-200 p-2 text-xs text-gray-600">
        {logs.length === 0 && <p className="text-gray-400">No agent activity recorded yet.</p>}
        {logs.map((event, i) => (
          <p key={i}>
            [{event.timestamp}] {event.type}
            {event.tool_name ? ` [${event.tool_name}]` : ""}: {event.content}
          </p>
        ))}
      </div>

      <h2 className="mb-2 text-sm font-semibold text-gray-700">Manual Crawler Trigger</h2>
      <button
        onClick={handleTrigger}
        className="rounded bg-purple-600 px-4 py-2 text-sm font-medium text-white hover:bg-purple-700"
      >
        Trigger Crawler
      </button>
      {crawlerMessage && <p className="mt-2 text-sm text-green-700">{crawlerMessage}</p>}
    </div>
  );
}

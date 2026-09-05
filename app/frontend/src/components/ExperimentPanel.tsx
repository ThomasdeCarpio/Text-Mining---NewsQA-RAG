import { useEffect, useState } from "react";
import {
  getExperimentResults,
  getExperiments,
  previewExperiment,
  runExperiment,
} from "../api/client";
import type { ExperimentResults, ExperimentSummary } from "../api/types";
import { ExperimentResultsView } from "./ExperimentResultsView";

export function ExperimentPanel() {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selected, setSelected] = useState("");
  const [preview, setPreview] = useState<ExperimentSummary | null>(null);
  const [results, setResults] = useState<ExperimentResults | null>(null);
  const [message, setMessage] = useState("");

  async function refresh() {
    try {
      const items = await getExperiments();
      setExperiments(items);
      setSelected((current) => current || items[0]?.filename || "");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2500);
    return () => window.clearInterval(timer);
  }, []);

  const current = experiments.find((item) => item.filename === selected);

  async function handlePreview() {
    if (!selected) return;
    setMessage("");
    try {
      setPreview(await previewExperiment(selected));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleRun() {
    if (!current) return;
    const rebuild = current.status === "complete";
    const question = rebuild
      ? "Rebuild the comparison from saved run artifacts? No retrieval will be repeated."
      : `Run or resume ${current.run_count} benchmark configurations?`;
    if (!window.confirm(question)) return;
    setMessage("");
    try {
      await runExperiment(current.filename);
      setMessage(
        rebuild
          ? "Rebuilding the saved comparison."
          : "Experiment started. Successful questions will be reused.",
      );
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleResults() {
    if (!selected) return;
    setMessage("");
    try {
      setResults(await getExperimentResults(selected));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  return (
    <section className="stamp-shadow rounded border-2 border-rule bg-surface p-4">
      <div className="flex flex-wrap items-end gap-3">
        <label className="min-w-72 flex-1 font-wire text-[10px] uppercase tracking-wide text-ink-muted">
          Experiment
          <select
            className="mt-1 block w-full rounded border-2 border-rule bg-paper px-3 py-2 text-sm normal-case text-ink"
            value={selected}
            onChange={(event) => {
              setSelected(event.target.value);
              setPreview(null);
              setResults(null);
              setMessage("");
            }}
          >
            {experiments.map((item) => (
              <option key={item.filename} value={item.filename}>{item.name}</option>
            ))}
          </select>
        </label>
        <button className="rounded border-2 border-rule bg-paper px-3 py-2 font-wire text-xs uppercase" onClick={handlePreview}>
          Preview config
        </button>
        <button
          className="rounded bg-accent px-3 py-2 font-wire text-xs uppercase text-surface disabled:opacity-50"
          disabled={!current || current.status === "running"}
          onClick={handleRun}
        >
          {current?.status === "running"
            ? "Working…"
            : current?.status === "complete"
              ? "Rebuild summary"
              : "Run / resume"}
        </button>
        <button
          className="rounded border-2 border-moss px-3 py-2 font-wire text-xs uppercase text-moss disabled:opacity-50"
          disabled={!current?.result_ready}
          onClick={handleResults}
        >
          Load saved results
        </button>
      </div>

      {current && (
        <div className="mt-3 text-sm text-ink-muted">
          <p>{current.description}</p>
          <p className="mt-1">
            <strong className="text-ink">{current.run_count} configurations · {current.status}</strong>
            {Object.keys(current.status_counts).length > 0 && ` · ${JSON.stringify(current.status_counts)}`}
          </p>
          {current.result_ready && !results && (
            <p className="mt-1 text-moss">A saved report is available. It is not loaded until you request it.</p>
          )}
        </div>
      )}
      {current?.error && <p className="mt-2 text-sm text-accent">{current.error}</p>}
      {message && <p className="mt-2 text-sm text-moss">{message}</p>}

      {preview?.runs && (
        <div className="mt-4 border-t border-rule pt-3 text-sm">
          <h2 className="font-display text-lg">Locked configuration</h2>
          {preview.partitions && Object.entries(preview.partitions).map(([index, partitions]) => (
            <div key={index} className="mt-2 text-ink-muted">
              <strong className="text-ink">Index: {index}</strong>
              {Object.entries(partitions).map(([name, value]) => (
                <p key={name}>{name}: {value.articles} articles · questions {JSON.stringify(value.questions)}</p>
              ))}
            </div>
          ))}
          <div className="mt-3 max-h-48 overflow-auto font-wire text-xs">
            {preview.runs.map((run) => (
              <p key={run.run_id} className="mb-1">
                {run.parameters.variant} · {run.parameters.retriever} · {run.parameters.reranker}
                {` · top_k=${run.parameters.top_k} · top_n=${run.parameters.rerank_top_n}`}
              </p>
            ))}
          </div>
        </div>
      )}

      {results
        ? <ExperimentResultsView filename={selected} results={results} />
        : (
          <div className="mt-4 rounded border-2 border-dashed border-rule bg-paper p-6 text-center text-sm text-ink-muted">
            Chưa nạp report. Hãy chạy/resume experiment hoặc bấm <strong className="text-ink">Load saved results</strong>.
          </div>
        )}

      <details className="mt-4 border-t border-rule pt-3 text-sm text-ink-muted">
        <summary className="cursor-pointer font-medium text-ink">Run behavior</summary>
        <p className="mt-2">Run/resume only executes missing work. Rebuild summary recalculates the comparison from saved artifacts.</p>
        <p>A completed experiment is immutable; copy its YAML and use a new experiment ID for a genuinely fresh run.</p>
      </details>
    </section>
  );
}

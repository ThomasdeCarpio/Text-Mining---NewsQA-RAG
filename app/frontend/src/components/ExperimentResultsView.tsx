import { useEffect, useState } from "react";
import { getExperimentRun } from "../api/client";
import type { ExperimentResults, ExperimentRunDetail } from "../api/types";

type ResultRow = ExperimentResults["runs"][number];

function value(row: ResultRow, key: string) {
  return typeof row[key] === "number" ? row[key] : null;
}

function format(row: ResultRow, key: string, digits = 4) {
  const result = value(row, key);
  return result === null ? "—" : result.toFixed(digits);
}

function label(row: ResultRow) {
  return `${row.variant} · ${row.retriever} · ${row.reranker}`;
}

const metrics = [
  ["MRR@5", "retrieval.mrr@5.mean", "Vị trí của gold chunk đầu tiên; càng cao càng tốt."],
  ["NDCG@5", "retrieval.ndcg@5.mean", "Thưởng cho gold chunk xuất hiện gần đầu danh sách."],
  ["Recall@5", "retrieval.recall@5.mean", "Tỷ lệ gold chunks được tìm thấy trong top 5."],
] as const;

export function ExperimentResultsView({
  filename,
  results,
}: {
  filename: string;
  results: ExperimentResults;
}) {
  const [runId, setRunId] = useState(String(results.runs[0]?.run_id ?? ""));
  const [detail, setDetail] = useState<ExperimentRunDetail | null>(null);
  const [error, setError] = useState("");
  const selected = results.runs.find((row) => row.run_id === runId) ?? results.runs[0];
  const history = results.history.find((item) => item.run_id === runId);

  useEffect(() => {
    if (!runId) return;
    setDetail(null);
    setError("");
    getExperimentRun(filename, runId)
      .then(setDetail)
      .catch((reason) => setError(reason instanceof Error ? reason.message : String(reason)));
  }, [filename, runId]);

  if (!selected) return <p className="mt-4 text-sm text-ink-muted">Report này chưa có run hoàn tất.</p>;

  return (
    <div className="mt-4 border-t border-rule pt-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="font-display text-xl">Kết quả đã lưu</h2>
          <p className="text-xs text-ink-muted">Experiment: {results.experiment_id} · tạo lúc {new Date(results.generated_at).toLocaleString()}</p>
        </div>
        <label className="font-wire text-[10px] uppercase text-ink-muted">
          Run đang xem
          <select className="mt-1 block rounded border-2 border-rule bg-paper px-3 py-2 text-sm normal-case text-ink" value={runId} onChange={(event) => setRunId(event.target.value)}>
            {results.runs.map((row) => <option key={String(row.run_id)} value={String(row.run_id)}>{label(row)}</option>)}
          </select>
        </label>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
        {metrics.map(([name, key, help]) => (
          <div key={key} className="rounded border-2 border-rule bg-paper p-3">
            <p className="font-wire text-[10px] uppercase text-ink-muted">{name}</p>
            <p className="font-display text-2xl">{format(selected, key)}</p>
            <p className="mt-1 text-xs text-ink-muted">{help}</p>
          </div>
        ))}
        <div className="rounded border-2 border-rule bg-paper p-3">
          <p className="font-wire text-[10px] uppercase text-ink-muted">P95 latency</p>
          <p className="font-display text-2xl">{format(selected, "latency.total.p95_ms", 1)} ms</p>
          <p className="mt-1 text-xs text-ink-muted">95% câu hoàn tất trong thời gian này, gồm cả cold start.</p>
        </div>
        <div className="rounded border-2 border-rule bg-paper p-3">
          <p className="font-wire text-[10px] uppercase text-ink-muted">Coverage</p>
          <p className="font-display text-2xl">{format(selected, "coverage.success_rate")}</p>
          <p className="mt-1 text-xs text-ink-muted">Tỷ lệ câu chạy thành công trong run này.</p>
        </div>
        <div className="rounded border-2 border-rule bg-paper p-3">
          <p className="font-wire text-[10px] uppercase text-ink-muted">Run time</p>
          <p className="font-display text-2xl">{history?.wall_time_seconds?.toFixed(1) ?? "—"} s</p>
          <p className="mt-1 text-xs text-ink-muted">Thời gian toàn bộ cấu hình này chạy.</p>
        </div>
      </div>

      <h3 className="mb-2 mt-6 font-display text-lg">So sánh các run trong experiment</h3>
      <div className="space-y-3 rounded border-2 border-rule bg-paper p-3">
        {results.runs.map((row) => (
          <div key={String(row.run_id)} className={row.run_id === runId ? "rounded bg-accent-soft p-2" : "p-2"}>
            <p className="mb-1 font-wire text-xs">{label(row)}{results.pareto_run_ids.includes(String(row.run_id)) ? " ★" : ""}</p>
            {metrics.map(([name, key]) => {
              const score = value(row, key) ?? 0;
              return (
                <div key={key} className="grid grid-cols-[4rem_1fr_3rem] items-center gap-2 text-xs">
                  <span className="text-ink-muted">{name}</span>
                  <div className="h-2 overflow-hidden rounded bg-rule"><div className="h-full bg-moss" style={{ width: `${Math.max(0, Math.min(score, 1)) * 100}%` }} /></div>
                  <span>{score.toFixed(3)}</span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <p className="mt-1 text-xs text-ink-muted">★ = không bị run khác vượt cả MRR@5 lẫn P95 latency.</p>

      <h3 className="mb-2 mt-6 font-display text-lg">Failure Analysis — {label(selected)}</h3>
      {error && <p className="text-sm text-accent">{error}</p>}
      {!detail && !error && <p className="text-sm text-ink-muted">Đang tải report của run…</p>}
      {detail && detail.failures.length === 0 && <p className="rounded border-2 border-rule bg-paper p-3 text-sm">Run này không ghi nhận failure.</p>}
      {detail && detail.failures.length > 0 && (
        <div className="overflow-x-auto rounded border-2 border-rule">
          <table className="w-full text-left text-sm">
            <thead className="bg-paper font-wire text-[10px] uppercase text-ink-muted"><tr><th className="p-2">Question</th><th className="p-2">Expected</th><th className="p-2">Reason</th></tr></thead>
            <tbody>{detail.failures.map((failure) => (
              <tr key={failure.question_id} className="border-t border-rule"><td className="p-2">{failure.question}</td><td className="p-2">{failure.expected}</td><td className="p-2 text-ink-muted">{failure.reason}</td></tr>
            ))}</tbody>
          </table>
        </div>
      )}
    </div>
  );
}

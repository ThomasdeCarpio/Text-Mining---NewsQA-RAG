import { useState } from "react";
import type { RetrievalResult } from "../api/types";

export function RetrievalResultCard({ result, rank }: { result: RetrievalResult; rank: number }) {
  const [open, setOpen] = useState(false);
  const title = result.metadata.title || result.id;
  const meta = [result.metadata.publisher, result.metadata.publish_date, result.metadata.author]
    .filter((v) => v && v !== "Unknown Publisher" && v !== "Unknown Date" && v !== "Unknown Author")
    .join(" · ");

  return (
    <div className="stamp-shadow rounded border-2 border-rule bg-surface">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-start gap-3 p-3 text-left"
      >
        <span className="font-wire text-xs text-ink-muted">#{rank}</span>
        <div className="min-w-0 flex-1">
          <p className="font-display text-base text-ink">{title}</p>
          {meta && <p className="font-wire text-[10px] uppercase tracking-wide text-ink-muted">{meta}</p>}
          {!open && <p className="mt-1 line-clamp-2 text-sm text-ink-muted">{result.text}</p>}
        </div>
        <span className="shrink-0 font-wire text-[10px] text-ink-muted">
          {result.distance.toFixed(3)}
          <br />
          {open ? "▲" : "▼"}
        </span>
      </button>

      {open && (
        <div className="border-t-2 border-rule bg-paper p-3">
          <p className="whitespace-pre-wrap text-sm text-ink">{result.text}</p>
          <p className="mt-3 font-wire text-[10px] text-ink-muted">
            {Object.entries(result.metadata)
              .map(([k, v]) => `${k}: ${v}`)
              .join(" · ")}
          </p>
        </div>
      )}
    </div>
  );
}

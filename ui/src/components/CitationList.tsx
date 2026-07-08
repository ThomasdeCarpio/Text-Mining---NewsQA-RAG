import { useState } from "react";
import type { Citation } from "../api/types";

export function CitationList({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState(false);
  if (citations.length === 0) return null;

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(!open)}
        className="font-wire text-[10px] uppercase tracking-wide text-accent hover:text-accent-hover"
      >
        {open ? "▲ Hide Sources" : "▼ View Sources"}
      </button>
      {open && (
        <div className="mt-2 flex flex-col gap-2">
          {citations.map((c, i) => (
            <div key={i} className="rounded border-2 border-rule bg-paper p-2 text-left text-sm">
              <p className="font-display font-semibold text-ink">
                {c.title}{" "}
                <span className="font-wire text-[10px] font-normal text-ink-muted">
                  {c.source} · {c.date}
                </span>
              </p>
              <p className="mt-1 italic text-ink-muted">"{c.chunk_text}"</p>
              <a href={c.url} target="_blank" rel="noreferrer" className="font-wire text-[10px] text-accent hover:underline">
                {c.url}
              </a>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

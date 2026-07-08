import { useState } from "react";
import type { Citation } from "../api/types";

export function CitationList({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState(false);
  if (citations.length === 0) return null;

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(!open)}
        className="text-xs font-medium text-purple-700 hover:underline"
      >
        {open ? "Hide Sources" : "View Sources"}
      </button>
      {open && (
        <div className="mt-2 flex flex-col gap-2">
          {citations.map((c, i) => (
            <div key={i} className="rounded border border-gray-200 bg-gray-50 p-2 text-left text-sm">
              <p className="font-semibold">
                {c.source} <span className="font-normal text-gray-500">({c.date})</span> — {c.title}
              </p>
              <p className="mt-1 italic text-gray-600">"{c.chunk_text}"</p>
              <a href={c.url} target="_blank" rel="noreferrer" className="text-xs text-purple-700 hover:underline">
                {c.url}
              </a>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

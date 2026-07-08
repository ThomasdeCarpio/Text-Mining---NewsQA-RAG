import { useEffect, useState } from "react";
import { getAlgorithms, getCollectionStats, searchRetrieval } from "../api/client";
import type { AlgorithmOption, CollectionStats, RetrievalResult, RetrievalTiming } from "../api/types";
import { RetrievalResultCard } from "../components/RetrievalResultCard";

export function RetrievalPage() {
  const [algorithms, setAlgorithms] = useState<AlgorithmOption[]>([]);
  const [stats, setStats] = useState<CollectionStats | null>(null);
  const [algorithm, setAlgorithm] = useState("dense");
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState<RetrievalResult[]>([]);
  const [timing, setTiming] = useState<RetrievalTiming | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getAlgorithms().then(setAlgorithms);
    getCollectionStats().then(setStats);
  }, []);

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setIsSearching(true);
    setError(null);
    setResults([]);
    setTiming(null);
    try {
      const data = await searchRetrieval(query, algorithm, topK);
      setResults(data.results);
      setTiming(data.timing);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setIsSearching(false);
    }
  }

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <h1 className="mb-4 font-display text-2xl text-ink">Retrieval Playground</h1>

      <div className="stamp-shadow mb-6 rounded border-2 border-rule bg-surface p-3 font-wire text-xs text-ink-muted">
        {stats ? (
          stats.exists ? (
            <p>
              Collection <span className="text-accent">{stats.name}</span> — {stats.count} chunks
              indexed
              {typeof stats.embedding_info.model_name === "string" &&
                ` · embedding model: ${stats.embedding_info.model_name}`}
            </p>
          ) : (
            <p>Collection not found yet — run the ingestion pipeline first.</p>
          )
        ) : (
          <p>Loading collection stats...</p>
        )}
      </div>

      <form onSubmit={handleSearch} className="mb-6 flex flex-col gap-3">
        <input
          className="rounded border-2 border-rule bg-surface px-3 py-2 text-sm text-ink outline-none focus:border-accent"
          placeholder="Enter a query to search the corpus..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />

        <div className="flex flex-wrap gap-3">
          {algorithms.map((a) => (
            <label
              key={a.id}
              className={`flex items-center gap-2 rounded border-2 px-3 py-2 font-wire text-xs uppercase tracking-wide ${
                a.available
                  ? "border-rule bg-surface text-ink has-checked:border-accent has-checked:text-accent"
                  : "cursor-not-allowed border-rule/50 bg-paper text-ink-muted/60"
              }`}
            >
              <input
                type="radio"
                name="algorithm"
                value={a.id}
                checked={algorithm === a.id}
                disabled={!a.available}
                onChange={() => setAlgorithm(a.id)}
              />
              {a.label}
              {!a.available && <span className="italic">(coming soon)</span>}
            </label>
          ))}
        </div>

        <div className="flex items-center gap-3">
          <label className="font-wire text-xs uppercase tracking-wide text-ink-muted">Top K</label>
          <input
            type="number"
            min={1}
            max={50}
            className="w-20 rounded border-2 border-rule bg-surface px-2 py-1 text-sm text-ink"
            value={topK}
            onChange={(e) => setTopK(Math.min(50, Math.max(1, Number(e.target.value))))}
          />
          <button
            type="submit"
            disabled={isSearching}
            className="rounded bg-accent px-4 py-2 font-wire text-xs uppercase tracking-wide text-surface hover:bg-accent-hover disabled:opacity-50"
          >
            {isSearching ? "Searching..." : "Search"}
          </button>
        </div>
      </form>

      {error && <p className="mb-4 text-sm text-accent">{error}</p>}

      {timing && (
        <div className="stamp-shadow mb-4 flex flex-wrap items-center gap-x-4 gap-y-1 rounded border-2 border-rule bg-surface p-2 font-wire text-[10px] uppercase tracking-wide text-ink-muted">
          <span>
            Embed: <span className="text-ink">{timing.embed_ms}ms</span>
            {timing.model_cold_start && <span className="text-accent"> (cold start)</span>}
          </span>
          <span>
            DB query: <span className="text-ink">{timing.db_query_ms}ms</span>
          </span>
          <span>
            Total: <span className="text-ink">{timing.total_ms}ms</span>
          </span>
        </div>
      )}

      <div className="flex flex-col gap-3">
        {results.map((r, i) => (
          <RetrievalResultCard key={r.id} result={r} rank={i + 1} />
        ))}
      </div>
    </div>
  );
}

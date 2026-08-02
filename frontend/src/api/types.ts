export interface Citation {
  source: string;
  title: string;
  date: string;
  url: string;
  chunk_text: string;
}

export type AgentEventType = "thought" | "tool_call" | "tool_result" | "final_answer";

export interface AgentEvent {
  type: AgentEventType;
  content: string;
  tool_name: string | null;
  citations: Citation[] | null;
  timestamp: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
}

export interface LoginResponse {
  session_id: string;
  username: string;
  role: "user" | "admin";
}

export interface AlgorithmOption {
  id: string;
  label: string;
  available: boolean;
}

export interface CollectionStats {
  exists: boolean;
  name: string;
  count: number;
  metadata: Record<string, unknown>;
  embedding_info: Record<string, unknown>;
}

export interface RetrievalResult {
  id: string;
  text: string;
  distance: number;
  metadata: Record<string, string>;
}

export interface RetrievalTiming {
  model_cold_start: boolean;
  embed_ms: number;
  db_query_ms: number;
  total_ms: number;
}

export interface RetrievalSearchResponse {
  results: RetrievalResult[];
  timing: RetrievalTiming;
}

export interface ExperimentRun {
  run_id: string;
  parameters: Record<string, string | number | boolean>;
}

export interface ExperimentSummary {
  filename: string;
  id: string;
  name: string;
  description: string;
  run_count: number;
  status: "pending" | "running" | "complete" | "failed";
  status_counts: Record<string, number>;
  result_ready: boolean;
  error: string | null;
  runs?: ExperimentRun[];
  partitions?: Record<
    string,
    Record<string, { articles: number; questions: Record<string, number> }>
  >;
}

export interface ExperimentResults {
  experiment_id: string;
  generated_at: string;
  runs: Array<Record<string, string | number | boolean | null>>;
  pareto_run_ids: string[];
  history: Array<{
    run_id: string;
    status: string;
    wall_time_seconds?: number;
  }>;
}

export interface ExperimentRunDetail {
  run_id: string;
  parameters: Record<string, string | number | boolean>;
  generated_at: string | null;
  coverage: Record<string, number>;
  failures: Array<{
    question_id: string;
    question: string;
    expected: string;
    reason: string;
  }>;
}

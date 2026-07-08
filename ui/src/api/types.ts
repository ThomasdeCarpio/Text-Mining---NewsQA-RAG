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

export interface FailureCase {
  question: string;
  expected: string;
  retrieved: string;
  reason: string;
}

export interface SearchComparisonRow {
  metric: string;
  vector_search: number;
  hybrid_search: number;
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

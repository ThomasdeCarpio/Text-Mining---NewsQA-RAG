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

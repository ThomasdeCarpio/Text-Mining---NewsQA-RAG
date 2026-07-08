import type {
  AgentEvent,
  ChatMessage,
  FailureCase,
  LoginResponse,
  SearchComparisonRow,
} from "./types";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function getJson<T>(path: string): Promise<T> {
  const resp = await fetch(`${BASE_URL}${path}`);
  if (!resp.ok) throw new Error(`GET ${path} failed: ${resp.status}`);
  return resp.json();
}

async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const resp = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!resp.ok) throw new Error(`POST ${path} failed: ${resp.status}`);
  return resp.json();
}

export async function login(username: string, password: string): Promise<LoginResponse | null> {
  const resp = await fetch(`${BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (resp.status !== 200) return null;
  return resp.json();
}

export function getHistory(sessionId: string): Promise<ChatMessage[]> {
  return getJson(`/chat/history/${sessionId}`);
}

export function clearChat(sessionId: string): Promise<{ cleared: boolean }> {
  return postJson(`/chat/clear/${sessionId}`);
}

export function getMetrics(): Promise<Record<string, number>> {
  return getJson("/admin/metrics");
}

export function getSearchComparison(): Promise<SearchComparisonRow[]> {
  return getJson("/admin/search-comparison");
}

export function getFailureCases(): Promise<FailureCase[]> {
  return getJson("/admin/failure-cases");
}

export function getPipelineLogs(): Promise<AgentEvent[]> {
  return getJson("/admin/pipeline-logs");
}

export async function triggerCrawler(): Promise<boolean> {
  const data = await postJson<{ triggered: boolean }>("/admin/trigger-crawler");
  return data.triggered;
}

/**
 * The backend streams SSE ("data: {json}\n\n" frames) over a POST response,
 * so the browser's EventSource API (GET-only, no body) can't be used here —
 * read the response body stream manually instead.
 */
export async function askStream(
  sessionId: string,
  question: string,
  onEvent: (event: AgentEvent) => void,
): Promise<void> {
  const resp = await fetch(`${BASE_URL}/chat/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, question }),
  });
  if (!resp.ok || !resp.body) throw new Error(`POST /chat/ask failed: ${resp.status}`);

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let separatorIndex;
    while ((separatorIndex = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, separatorIndex);
      buffer = buffer.slice(separatorIndex + 2);
      if (frame.startsWith("data: ")) {
        onEvent(JSON.parse(frame.slice("data: ".length)));
      }
    }
  }
}

import { useEffect, useRef, useState } from "react";
import { askStream, clearChat, getHistory } from "../api/client";
import type { AgentEvent, ChatMessage, Citation } from "../api/types";
import { useAuth } from "../context/AuthContext";
import { ChatBubble } from "../components/ChatBubble";
import { CitationList } from "../components/CitationList";

export function ChatPage() {
  const { user } = useAuth();
  const sessionId = user!.sessionId;

  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const [steps, setSteps] = useState<AgentEvent[]>([]);
  const [pendingAnswer, setPendingAnswer] = useState<{ content: string; citations: Citation[] } | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  function refreshHistory() {
    getHistory(sessionId).then(setHistory);
  }

  useEffect(refreshHistory, [sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, steps, pendingAnswer]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (!q) return;
    setQuestion("");
    setHistory((h) => [...h, { role: "user", content: q, citations: [] }]);
    setIsThinking(true);
    setSteps([]);
    setPendingAnswer(null);

    await askStream(sessionId, q, (event) => {
      if (event.type === "final_answer") {
        setPendingAnswer({ content: event.content, citations: event.citations ?? [] });
      } else {
        setSteps((s) => [...s, event]);
      }
    });

    setIsThinking(false);
    refreshHistory();
    setSteps([]);
    setPendingAnswer(null);
  }

  async function handleClear() {
    await clearChat(sessionId);
    setHistory([]);
  }

  return (
    <div className="flex h-screen flex-1 flex-col">
      <div className="flex items-center justify-between border-b-2 border-rule bg-surface p-4">
        <h1 className="font-display text-xl text-ink">News Chat</h1>
        <button
          onClick={handleClear}
          className="font-wire text-[10px] uppercase tracking-wide text-ink-muted hover:text-accent"
        >
          Clear chat
        </button>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {history.map((m, i) => (
          <ChatBubble key={i} message={m} />
        ))}

        {isThinking && (
          <div className="stamp-shadow rounded border-2 border-dashed border-rule bg-surface p-3 text-left text-sm text-ink-muted">
            <p className="mb-1 font-wire text-[10px] uppercase tracking-wide text-moss">
              Agent is thinking... Searching database...
            </p>
            {steps.map((s, i) => (
              <p key={i} className="font-wire text-xs text-ink-muted">
                <span className="font-semibold text-ink">{s.type}</span>: {s.content}
              </p>
            ))}
            {pendingAnswer && (
              <div className="mt-2 rounded border-2 border-rule bg-paper p-2">
                <p className="text-sm text-ink">{pendingAnswer.content}</p>
                <CitationList citations={pendingAnswer.citations} />
              </div>
            )}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2 border-t-2 border-rule bg-surface p-4">
        <input
          className="flex-1 rounded border-2 border-rule bg-paper px-3 py-2 text-sm text-ink outline-none focus:border-accent"
          placeholder="Ask a question about the news..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button
          type="submit"
          className="rounded bg-accent px-4 py-2 font-wire text-xs uppercase tracking-wide text-surface hover:bg-accent-hover"
        >
          Send
        </button>
      </form>
    </div>
  );
}

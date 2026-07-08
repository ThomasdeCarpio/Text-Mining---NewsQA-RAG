import type { ChatMessage } from "../api/types";
import { CitationList } from "./CitationList";

export function ChatBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[75%] rounded px-4 py-2 text-left ${
          isUser
            ? "bg-accent text-surface"
            : "stamp-shadow border-2 border-rule bg-surface text-ink"
        }`}
      >
        <p className="whitespace-pre-wrap text-sm">{message.content}</p>
        {!isUser && <CitationList citations={message.citations} />}
      </div>
    </div>
  );
}

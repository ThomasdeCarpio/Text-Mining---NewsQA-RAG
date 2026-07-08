import type { ChatMessage } from "../api/types";
import { CitationList } from "./CitationList";

export function ChatBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[75%] rounded-lg px-4 py-2 text-left ${
          isUser ? "bg-purple-600 text-white" : "bg-gray-100 text-gray-900"
        }`}
      >
        <p className="whitespace-pre-wrap text-sm">{message.content}</p>
        {!isUser && <CitationList citations={message.citations} />}
      </div>
    </div>
  );
}

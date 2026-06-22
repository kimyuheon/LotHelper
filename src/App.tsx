import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import Markdown from "./Markdown";
import "./App.css";

type Role = "user" | "assistant";

interface Message {
  id: number;
  role: Role;
  content: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 0,
      role: "assistant",
      content: "안녕하세요! CppAI입니다. 무엇을 도와드릴까요?",
    },
  ]);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const nextId = useRef(1);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pending]);

  async function send() {
    const text = input.trim();
    if (!text || pending) return;

    const userMsg: Message = { id: nextId.current++, role: "user", content: text };
    const history = [...messages, userMsg];
    setMessages(history);
    setInput("");
    setPending(true);

    try {
      const reply = await invoke<string>("chat", {
        messages: history.map(({ role, content }) => ({ role, content })),
      });
      setMessages((prev) => [
        ...prev,
        { id: nextId.current++, role: "assistant", content: reply },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: nextId.current++,
          role: "assistant",
          content: `오류가 발생했습니다: ${String(err)}`,
        },
      ]);
    } finally {
      setPending(false);
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="chat">
      <header className="chat-header">CppAI</header>

      <div className="chat-messages">
        {messages.map((m) => (
          <div key={m.id} className={`bubble-row ${m.role}`}>
            <div className={`bubble ${m.role}`}>
              {m.role === "assistant" ? (
                <Markdown content={m.content} />
              ) : (
                m.content
              )}
            </div>
          </div>
        ))}
        {pending && (
          <div className="bubble-row assistant">
            <div className="bubble assistant typing">…</div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form
        className="chat-input"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.currentTarget.value)}
          onKeyDown={onKeyDown}
          placeholder="메시지를 입력하세요... (Enter 전송, Shift+Enter 줄바꿈)"
          rows={1}
        />
        <button type="submit" disabled={pending || !input.trim()}>
          전송
        </button>
      </form>
    </div>
  );
}

export default App;

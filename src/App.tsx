import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import Markdown from "./Markdown";
import "./App.css";

type Role = "user" | "assistant";
type Mode = "chat" | "propose" | "auto" | "agent";

interface FileBlock {
  path: string;
  content: string;
}

interface Message {
  id: number;
  role: Role;
  content: string;
  actions?: string[];
  proposals?: FileBlock[];
}

interface AgentResult {
  reply: string;
  actions: string[];
  changed: string[];
}

interface LlamaStatus {
  state: string; // "starting" | "choose" | "no_model"
  models: string[];
}

const MODE_LABELS: Record<Mode, string> = {
  chat: "채팅",
  propose: "제안→승인",
  auto: "자동 적용",
  agent: "에이전트",
};

const MODE_HINTS: Record<Mode, string> = {
  chat: "그냥 대화합니다. 파일은 건드리지 않습니다.",
  propose: "파일 변경을 제안하면 '적용' 버튼으로 반영합니다.",
  auto: "파일 변경 제안을 자동으로 저장합니다.",
  agent: "AI가 스스로 파일을 읽고/쓰고, 빌드해서 오류를 고쳐 성공할 때까지 반복합니다.",
};

// Parse ```file:PATH\n...``` blocks out of an assistant reply.
function parseFileBlocks(text: string): FileBlock[] {
  const re = /```file:([^\n`]+)\n([\s\S]*?)```/g;
  const blocks: FileBlock[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    blocks.push({ path: m[1].trim(), content: m[2].replace(/\n$/, "") });
  }
  return blocks;
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
  const [ready, setReady] = useState(false);
  const [modelChoice, setModelChoice] = useState<string[] | null>(null);
  const [noModel, setNoModel] = useState(false);
  const [mode, setMode] = useState<Mode>("chat");
  const [workspace, setWorkspace] = useState<string | null>(null);
  const [files, setFiles] = useState<string[]>([]);
  const [showFiles, setShowFiles] = useState(false);
  const [applied, setApplied] = useState<Set<string>>(new Set());
  const nextId = useRef(1);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pending]);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      while (!cancelled) {
        const ok = await invoke<boolean>("llama_ready").catch(() => false);
        if (ok) {
          if (!cancelled) setReady(true);
          return;
        }
        await new Promise((r) => setTimeout(r, 1500));
      }
    }
    poll();
    invoke<string | null>("current_workspace").then((w) => w && setWorkspace(w));
    invoke<LlamaStatus>("llama_status")
      .then((s) => {
        if (s.state === "choose") setModelChoice(s.models);
        else if (s.state === "no_model") setNoModel(true);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  async function pickModel(name: string) {
    setModelChoice(null);
    try {
      await invoke("start_model", { name });
    } catch (err) {
      addMessage({ role: "assistant", content: `모델 시작 실패: ${String(err)}` });
    }
  }

  async function refreshFiles() {
    try {
      setFiles(await invoke<string[]>("workspace_files"));
    } catch {
      setFiles([]);
    }
  }

  async function selectWorkspace() {
    const w = await invoke<string | null>("select_workspace").catch(() => null);
    if (w) {
      setWorkspace(w);
      await refreshFiles();
    }
  }

  function addMessage(msg: Omit<Message, "id">) {
    setMessages((prev) => [...prev, { id: nextId.current++, ...msg }]);
  }

  async function applyBlock(msgId: number, b: FileBlock) {
    try {
      await invoke("write_file", { path: b.path, content: b.content });
      setApplied((prev) => new Set(prev).add(`${msgId}:${b.path}`));
      await refreshFiles();
    } catch (err) {
      addMessage({ role: "assistant", content: `저장 실패: ${String(err)}` });
    }
  }

  async function send() {
    const text = input.trim();
    if (!text || pending || !ready) return;
    if (mode !== "chat" && !workspace) {
      addMessage({
        role: "assistant",
        content: "먼저 상단의 **폴더 선택**으로 작업 폴더를 지정해주세요.",
      });
      return;
    }

    const userMsg: Message = { id: nextId.current++, role: "user", content: text };
    const history = [...messages, userMsg];
    setMessages(history);
    setInput("");
    setPending(true);

    const wire = history
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map(({ role, content }) => ({ role, content }));

    try {
      if (mode === "agent") {
        const res = await invoke<AgentResult>("agent_chat", { messages: wire });
        addMessage({ role: "assistant", content: res.reply, actions: res.actions });
        if (res.changed.length) await refreshFiles();
      } else {
        const editBlocks = mode === "propose" || mode === "auto";
        const reply = await invoke<string>("chat", { messages: wire, editBlocks });
        const proposals = editBlocks ? parseFileBlocks(reply) : [];

        if (mode === "auto" && proposals.length) {
          for (const b of proposals) {
            await invoke("write_file", { path: b.path, content: b.content });
          }
          await refreshFiles();
        }

        const msgId = nextId.current++;
        setMessages((prev) => [
          ...prev,
          { id: msgId, role: "assistant", content: reply, proposals },
        ]);
        if (mode === "auto") {
          setApplied((prev) => {
            const next = new Set(prev);
            proposals.forEach((b) => next.add(`${msgId}:${b.path}`));
            return next;
          });
        }
      }
    } catch (err) {
      addMessage({ role: "assistant", content: `오류가 발생했습니다: ${String(err)}` });
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
      <header className="chat-header">
        <span className="brand">CppAI</span>
        <div className="mode-tabs">
          {(Object.keys(MODE_LABELS) as Mode[]).map((m) => (
            <button
              key={m}
              className={`mode-tab ${mode === m ? "active" : ""}`}
              onClick={() => setMode(m)}
              title={MODE_HINTS[m]}
            >
              {MODE_LABELS[m]}
            </button>
          ))}
        </div>
      </header>

      {mode !== "chat" && (
        <div className="workspace-bar">
          <button className="ws-btn" onClick={selectWorkspace}>
            📁 폴더 선택
          </button>
          {workspace ? (
            <>
              <span className="ws-path" title={workspace}>
                {workspace}
              </span>
              <button className="ws-btn" onClick={() => setShowFiles((s) => !s)}>
                파일 {files.length}개
              </button>
              <button className="ws-btn" onClick={() => invoke("open_in_vscode", { path: null })}>
                VSCode로 열기
              </button>
            </>
          ) : (
            <span className="ws-hint">작업 폴더가 지정되지 않았습니다</span>
          )}
        </div>
      )}

      {mode !== "chat" && showFiles && workspace && (
        <div className="file-list">
          {files.length === 0 && <span className="ws-hint">파일 없음</span>}
          {files.map((f) => (
            <button key={f} className="file-item" onClick={() => invoke("open_in_vscode", { path: f })}>
              {f}
            </button>
          ))}
        </div>
      )}

      {!ready && modelChoice && modelChoice.length > 0 ? (
        <div className="model-picker">
          <div className="model-picker-title">실행할 모델을 선택하세요</div>
          {modelChoice.map((m) => (
            <button key={m} className="model-option" onClick={() => pickModel(m)}>
              {m}
            </button>
          ))}
        </div>
      ) : !ready ? (
        <div className="status-banner">
          {noModel
            ? "⚠️ models/ 폴더에 .gguf 모델이 없습니다. 모델을 넣거나 llama-server를 수동 실행하세요."
            : "⏳ 모델을 불러오는 중입니다… (llama-server 준비 대기)"}
        </div>
      ) : null}

      <div className="chat-messages">
        {messages.map((m) => (
          <div key={m.id} className={`bubble-row ${m.role}`}>
            <div className={`bubble ${m.role}`}>
              {m.role === "assistant" ? <Markdown content={m.content} /> : m.content}

              {m.actions && m.actions.length > 0 && (
                <div className="actions">
                  {m.actions.map((a, i) => (
                    <div key={i} className="action-line">
                      {a}
                    </div>
                  ))}
                </div>
              )}

              {m.proposals && m.proposals.length > 0 && (
                <div className="proposals">
                  {m.proposals.map((b) => {
                    const key = `${m.id}:${b.path}`;
                    const done = applied.has(key);
                    return (
                      <div key={key} className="proposal">
                        <span className="proposal-path">📄 {b.path}</span>
                        <button
                          className="apply-btn"
                          disabled={done}
                          onClick={() => applyBlock(m.id, b)}
                        >
                          {done ? "✓ 적용됨" : "적용"}
                        </button>
                      </div>
                    );
                  })}
                </div>
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
          placeholder={
            ready ? `[${MODE_LABELS[mode]}] 메시지 입력... (Enter 전송)` : "모델 준비 중입니다..."
          }
          rows={1}
          disabled={!ready}
        />
        <button type="submit" disabled={!ready || pending || !input.trim()}>
          전송
        </button>
      </form>
    </div>
  );
}

export default App;

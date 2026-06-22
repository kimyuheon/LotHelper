import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import "./Markdown.css";

// Recursively pull the raw text out of a hast node (for the copy button).
function nodeText(node: any): string {
  if (!node) return "";
  if (node.type === "text") return node.value ?? "";
  if (Array.isArray(node.children)) return node.children.map(nodeText).join("");
  return "";
}

function CodeBlock({ node, children, ...props }: any) {
  const [copied, setCopied] = useState(false);
  const code = nodeText(node);

  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      /* clipboard unavailable */
    }
  }

  return (
    <div className="code-block">
      <button className="copy-btn" onClick={copy} type="button">
        {copied ? "복사됨" : "복사"}
      </button>
      <pre {...props}>{children}</pre>
    </div>
  );
}

export default function Markdown({ content }: { content: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{ pre: CodeBlock }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

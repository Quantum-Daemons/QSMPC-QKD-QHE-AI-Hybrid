import AgenticAIPanel from "../components/AgenticAIPanel";
import { DEMO_BANNER, isClientDemo } from "../lib/demoMode";

const DEMO = isClientDemo();

// Standalone Agentic-AI page: a generic in-browser agent over the client quantum-safe MCP channel.
// The same AgenticAIPanel is embedded (per-industry, fed by the live run) in each use-case page.
export default function AgenticAI() {
  return (
    <div>
      <h1 className="page-title">Agentic AI · quantum-safe MCP</h1>
      <p className="page-sub">
        A small language model runs <b>entirely in your browser</b> (WebGPU, no server); its request and
        response are sealed and signed over a client-side <b>quantum-safe MCP</b> channel (ML-KEM-768 +
        ML-DSA-65 + AES-256-GCM). The first run streams model weights from a CDN (browser-cached after).
      </p>
      {DEMO && <div className="note" style={{ marginBottom: 12 }}>🖥️ {DEMO_BANNER}</div>}

      <AgenticAIPanel />

      <div className="note" style={{ marginTop: 16, fontSize: 12 }}>
        In-browser AI via <b>Transformers.js</b> (onnxruntime-web) + <b>WebLLM</b> (WebGPU) — both Apache-2.0,
        loaded on demand as async chunks. Mirrors the "MCP delegate-and-verify" agentic-AI layer with a
        client-side quantum-safe channel — zero server compute. Each target-industry use-case page embeds this
        same panel with its own prompt + MCP tool, fed by that run's live metrics.
      </div>
    </div>
  );
}

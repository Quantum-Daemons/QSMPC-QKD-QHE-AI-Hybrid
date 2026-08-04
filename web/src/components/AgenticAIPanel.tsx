import { useEffect, useState } from "react";
import { type AgentRun, type Engine, hasWebGPU, runAgent, TFJS_MODELS, WEBLLM_MODELS } from "../lib/clientAI";
import { clientSecureWrap, type McpResult } from "../lib/clientMcp";
import { cryptoCfg } from "../lib/config";
import { downloadJSON } from "../lib/export";

// Reusable in-browser agentic-AI + quantum-safe MCP panel. A small LLM runs entirely client-side
// (WebGPU via WebLLM / Transformers.js) and its request+response are sealed+signed over the client
// quantum-safe MCP channel (clientMcp) — zero server. Used standalone on the Agentic-AI page AND
// embedded in each use-case page (per-industry prompt + MCP tool, fed by the live run's metrics) so
// the report's MCP "delegate-and-verify" layer is demonstrated contextually per target industry.

export interface RunContext { metrics?: Record<string, any> | null; benchmark?: any | null; }

export interface AgenticAIPanelProps {
  usecase?: string;          // industry context (finance/medical/...) — labels the MCP tool call
  defaultPrompt?: string;    // industry default prompt
  mcpToolName?: string;      // per-industry MCP tool name (score_transaction / query_cohort / …)
  runContext?: RunContext | null;   // live-run metrics/benchmark → composed into the prompt + payload
  inline?: boolean;          // compact rendering inside a use-case page (h3, no page intro)
}

const FALLBACK_PROMPT =
  "In one sentence, why does agentic AI accessing multiple data silos need a quantum-safe MCP layer?";

// Compose a compact, factual context line from the live run so the agent "explains THIS run".
function runContextLine(usecase: string | undefined, rc?: RunContext | null): string {
  if (!rc) return "";
  const m = rc.metrics || {};
  const bits: string[] = [];
  if (m.fhe_linear_ms != null) bits.push(`FHE-linear ${m.fhe_linear_ms} ms`);
  if (m.mpc_nonlinear_ms != null) bits.push(`MPC-non-linear ${m.mpc_nonlinear_ms} ms`);
  if (m.qber != null) bits.push(`QBER ${m.qber}`);
  if (m.federated_accuracy != null) bits.push(`federated accuracy ${m.federated_accuracy}`);
  if (m.psi_intersection_size != null) bits.push(`PSI |A∩B| ${m.psi_intersection_size}`);
  if (m.secure_key_bits != null) bits.push(`${m.secure_key_bits} secure key bits`);
  const adv = rc.benchmark?.advantage;
  if (adv?.score != null) bits.push(`quantum advantage ${adv.score}/10`);
  if (!bits.length) return "";
  return `Context from this ${usecase ?? "pipeline"} run — ${bits.join(", ")}.`;
}

export default function AgenticAIPanel({
  usecase, defaultPrompt, mcpToolName = "ai_inference", runContext = null, inline = false,
}: AgenticAIPanelProps) {
  const [engine, setEngine] = useState<Engine>("transformers");
  const [model, setModel] = useState(TFJS_MODELS[0].id);
  const [prompt, setPrompt] = useState(defaultPrompt || FALLBACK_PROMPT);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState("");
  const [run, setRun] = useState<AgentRun | null>(null);
  const [mcp, setMcp] = useState<(McpResult & { echoedPayload: any }) | null>(null);
  const [mcpQkd, setMcpQkd] = useState(true);
  const [combiner, setCombiner] = useState<string>(cryptoCfg().hybrid_combiner_mode ?? "hkdf");   // PQC+QKD key combiner
  const [gpu, setGpu] = useState(false);                 // genuine WebGPU (adapter obtainable), resolved async
  useEffect(() => { hasWebGPU().then(setGpu); }, []);
  const models = engine === "webllm" ? WEBLLM_MODELS : TFJS_MODELS;

  const onEngine = (e: Engine) => {
    setEngine(e);
    setModel((e === "webllm" ? WEBLLM_MODELS : TFJS_MODELS)[0].id);
  };

  const go = async () => {
    setBusy(true); setProgress(""); setRun(null); setMcp(null);
    const ctxLine = runContextLine(usecase, runContext);
    const effectivePrompt = ctxLine ? `${ctxLine}\n\n${prompt}` : prompt;
    const r = await runAgent(engine, effectivePrompt, { model, onProgress: setProgress });
    setRun(r);
    // wrap {prompt, answer, run-context} through the client quantum-safe MCP channel, tagged with the
    // per-industry tool name (signed + sealed + tamper-tested)
    const m = await clientSecureWrap(
      { prompt: effectivePrompt, answer: r.text || r.note || "", engine: r.engine, model: r.model,
        usecase: usecase ?? null, run_context: ctxLine || null },
      mcpQkd ? new Uint8Array(32).fill(0x71) : undefined, mcpToolName, combiner);
    setMcp(m); setBusy(false);
  };

  const ctxLine = runContextLine(usecase, runContext);

  return (
    <div className={inline ? "card" : undefined} style={inline ? { marginTop: 16 } : undefined}>
      {inline && (
        <>
          <h3 style={{ marginTop: 0 }}>Agentic AI · quantum-safe MCP</h3>
          <p className="muted" style={{ fontSize: 12, marginTop: -4 }}>
            A small LLM runs <b>in your browser</b>; its request + response are sealed and signed over a
            client-side quantum-safe MCP channel as a <code>{mcpToolName}</code> tool call for this {usecase} use case.
            {ctxLine ? " The live run's metrics are folded into the prompt." : " Run the simulator above to feed it live metrics."}
          </p>
        </>
      )}
      {!gpu && (
        <div className="note" style={{ marginBottom: 12 }}>
          ⚠ WebGPU not detected — WebLLM is disabled; Transformers.js will run on the slower WASM backend.
        </div>
      )}

      <div className={inline ? undefined : "card"}>
        <div className="btn-row">
          <label className="muted" htmlFor={`ai-engine-${usecase ?? "std"}`}>Engine</label>
          <select id={`ai-engine-${usecase ?? "std"}`} value={engine} onChange={(e) => onEngine(e.target.value as Engine)}>
            <option value="transformers">Transformers.js (ONNX · WebGPU/WASM)</option>
            <option value="webllm" disabled={!gpu}>WebLLM (chat LLM · WebGPU)</option>
          </select>
          <label className="muted" htmlFor={`ai-model-${usecase ?? "std"}`}>Model</label>
          <select id={`ai-model-${usecase ?? "std"}`} value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
          <label className="muted">
            <input type="checkbox" checked={mcpQkd} onChange={(e) => setMcpQkd(e.target.checked)} /> mix QKD key
          </label>
          <label className="muted" htmlFor={`ai-combiner-${usecase ?? "std"}`}>Combiner</label>
          <select id={`ai-combiner-${usecase ?? "std"}`} value={combiner} disabled={!mcpQkd}
            onChange={(e) => setCombiner(e.target.value)}
            title="PQC+QKD key combiner (crypto.hybrid_combiner_mode). HKDF = dual-PRF: the session key holds if either input does, under computational assumptions. Toeplitz+QLHL = a modified-Toeplitz strong extractor. Its leakage-resilience property — if the combined output AND one input key leak, the other retains quantifiable min-entropy — holds only under the paper's stated precondition that the input and seed are conditionally independent given any unbounded adversary (Giestinhas & Spiller, arXiv:2603.26907, Lemma 2). That precondition is assumed here, not established, so the guarantee is conditional — at the cost of longer keys (arXiv:2603.26907). Applies when 'mix QKD key' is on; the derived-key fingerprint below changes with the choice.">
            <option value="hkdf">HKDF (dual-PRF)</option>
            <option value="its_toeplitz">Toeplitz+QLHL (ITS · leakage-resilient)</option>
          </select>
        </div>
        {ctxLine && <div className="note" style={{ marginTop: 10, fontSize: 12 }}>🔗 {ctxLine}</div>}
        <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3}
          style={{ width: "100%", marginTop: 10 }} aria-label="Prompt" />
        <div className="btn-row" style={{ marginTop: 10 }}>
          <button className="btn" onClick={go} disabled={busy}>{busy ? "Running…" : "▶ Run in-browser agent"}</button>
          {busy && progress && (
            <span className="muted" style={{ fontSize: 12, alignSelf: "center" }}>{progress}</span>
          )}
        </div>
      </div>

      {(run || mcp) && (
        <div className="grid cols-2" style={{ marginTop: 16 }}>
          <div className="card">
            <h3>In-browser AI response</h3>
            {run?.note && <div className="note" style={{ marginBottom: 8 }}>{run.note}</div>}
            {run?.text && <pre className="mono" style={{ whiteSpace: "pre-wrap", fontSize: 13 }}>{run.text}</pre>}
            {run && (
              <table style={{ marginTop: 8 }}>
                <tbody>
                  <tr><td>engine</td><td>{run.engine}</td></tr>
                  <tr><td>model</td><td className="mono">{run.model}</td></tr>
                  <tr><td>backend</td><td>{run.backend}{run.webgpu ? " (WebGPU)" : ""}</td></tr>
                  <tr><td>elapsed</td><td>{run.elapsedMs} ms</td></tr>
                </tbody>
              </table>
            )}
          </div>
          <div className="card">
            <h3>Quantum-safe MCP envelope</h3>
            <p className="muted" style={{ fontSize: 12 }}>
              The AI request + response wrapped as a signed, sealed MCP tool call (no trusted third party).
            </p>
            {mcp && (
              <table style={{ marginTop: 8 }}>
                <tbody>
                  <tr><td>session</td><td className="mono">{mcp.session}</td></tr>
                  <tr><td>tool</td><td className="mono">{mcpToolName}</td></tr>
                  <tr><td>KEM / QKD-mixed</td><td>{mcp.kem} / {String(mcp.qkd_mixed)}</td></tr>
                  <tr><td>PQC+QKD combiner</td><td className="mono">{mcp.combiner}</td></tr>
                  <tr><td>derived key fingerprint</td><td className="mono">{mcp.session_key_fp}</td></tr>
                  <tr><td>method</td><td className="mono">{mcp.method}</td></tr>
                  <tr><td>authenticated ({mcp.signature_alg})</td>
                    <td style={{ color: mcp.authenticated ? "var(--accent-2)" : "var(--danger)" }}>{String(mcp.authenticated)}</td></tr>
                  <tr><td>tampered message rejected</td>
                    <td style={{ color: mcp.tamper_rejected ? "var(--accent-2)" : "var(--danger)" }}>{String(mcp.tamper_rejected)}</td></tr>
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}

      <div className="btn-row" style={{ marginTop: 16 }}>
        <button className="btn ghost" disabled={!run}
          onClick={() => downloadJSON({ usecase, tool: mcpToolName, run, mcp }, `qsh-agentic-${usecase ?? "ai"}.json`)}>
          ⬇ Download results (JSON)
        </button>
      </div>
    </div>
  );
}

import { useEffect, useState } from "react";
import { type AgentRun, type Engine, hasWebGPU, runAgent, TFJS_MODELS, WEBLLM_MODELS } from "../lib/clientAI";
import { clientSecureWrap, type McpResult } from "../lib/clientMcp";
import { isSoftwareAdapter, webgpuInfo } from "../lib/onnxInference";
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
  // Qualified, because this string is fed to a language model as run context. "federated accuracy
  // 0.74" invites it to describe the model as good or bad; the untrained score and the row count are
  // what make the number mean anything, and their absence here is what `test_federated_scope.py`'s
  // docstring calls out as the accuracy being handed over "with no qualifier".
  if (m.federated_accuracy != null) {
    bits.push(`federated accuracy ${m.federated_accuracy}`
      + (m.federated_baseline_accuracy != null ? ` vs ${m.federated_baseline_accuracy} untrained` : "")
      + (m.federated_eval_rows != null ? ` on ${m.federated_eval_rows} held-out rows` : ""));
  }
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
  // The context line the currently-displayed result was actually produced from. `go()` snapshots
  // runContext at click time, so pressing the pipeline's Run afterwards used to leave a rendered --
  // and downloadable -- MCP envelope describing a superseded run, sitting under the new run's
  // numbers with nothing to indicate it. Kept so the two can be compared on every render.
  const [ranWithCtx, setRanWithCtx] = useState<string | null>(null);
  const [combiner, setCombiner] = useState<string>(cryptoCfg().hybrid_combiner_mode ?? "hkdf");   // PQC+QKD key combiner
  const [gpu, setGpu] = useState(false);                 // genuine WebGPU (adapter obtainable), resolved async
  // "absent" and "software" are different facts and the note below used to call both of them
  // "not detected". Since hasWebGPU() began declining software rasterisers - they run this class of
  // model about 300x slower than the wasm build - that wording would have been actively wrong on a
  // machine where an adapter exists and was refused on purpose.
  const [gpuKind, setGpuKind] = useState<"usable" | "software" | "absent">("absent");
  useEffect(() => {
    hasWebGPU().then(setGpu);
    webgpuInfo().then((i: Record<string, unknown>) => {
      const info = i.info as Record<string, unknown> | undefined;
      setGpuKind(!info ? "absent" : isSoftwareAdapter(info) ? "software" : "usable");
    });
  }, []);
  const models = engine === "webllm" ? WEBLLM_MODELS : TFJS_MODELS;

  const onEngine = (e: Engine) => {
    setEngine(e);
    setModel((e === "webllm" ? WEBLLM_MODELS : TFJS_MODELS)[0].id);
  };

  const go = async () => {
    setBusy(true); setProgress(""); setRun(null); setMcp(null);
    const ctxLine = runContextLine(usecase, runContext);
  // A result is stale once the pipeline has produced different numbers than it was generated from.
  const stale = run !== null && ranWithCtx !== null && ranWithCtx !== ctxLine;
    const effectivePrompt = ctxLine ? `${ctxLine}\n\n${prompt}` : prompt;
    const r = await runAgent(engine, effectivePrompt, { model, onProgress: setProgress });
    setRun(r);
    // wrap {prompt, answer, run-context} through the client quantum-safe MCP channel, tagged with the
    // per-industry tool name (signed + sealed + tamper-tested)
    const m = await clientSecureWrap(
      { prompt: effectivePrompt, answer: r.text || r.note || "", engine: r.engine, model: r.model,
        usecase: usecase ?? null, run_context: ctxLine || null },
      mcpQkd ? new Uint8Array(32).fill(0x71) : undefined, mcpToolName, combiner);
    setMcp(m); setRanWithCtx(ctxLine); setBusy(false);
  };

  const ctxLine = runContextLine(usecase, runContext);
  // A result is stale once the pipeline has produced different numbers than it was generated from.
  const stale = run !== null && ranWithCtx !== null && ranWithCtx !== ctxLine;

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
      {/* EU AI Act Art. 50(1). Applicable since 2 August 2026, and this panel is squarely in scope:
          a system "intended to interact directly with natural persons" must be designed so the
          person is informed they are interacting with an AI. The Commission's final guidelines
          landed 20 July 2026; the only grace period runs to 2 December 2026 and covers Art. 50(2)
          synthetic-content marking for systems placed on the market before 2 August, which is a
          different obligation from this one.

          Shown unconditionally rather than behind the disclosure toggle, because the requirement is
          that the information reach the person, not that it be available to them. The project's own
          /value page already told EU visitors this obligation was live while this panel said
          nothing, which is the doc-versus-code divergence class this repo treats as most serious. */}
      <div className="note" style={{ marginBottom: 12, fontSize: 12 }}>
        You are interacting with an AI system. The replies below are generated by a small language
        model running in this browser tab, and may be inaccurate — treat them as a demonstration of
        the quantum-safe MCP channel, not as analysis to rely on. Disclosure under Article 50(1) of
        the EU AI Act, applicable since 2 August 2026.
      </div>
      {!gpu && (
        <div className="note" style={{ marginBottom: 12 }}>
          {gpuKind === "software" ? (
            <>
              ⚠ The WebGPU adapter here is a software rasteriser, so it is declined on purpose —
              measured about 300x slower than WebAssembly on this class of model, because every
              shader dispatch is emulated on the CPU. WebLLM is disabled and Transformers.js runs on
              the WASM backend, which is the faster of the two CPU paths.
            </>
          ) : (
            <>
              ⚠ WebGPU not detected — WebLLM is disabled; Transformers.js will run on the slower WASM
              backend.
            </>
          )}
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
          {/* Labelled for what it mixes. The analytic BB84/E91 models return bit COUNTS - a secret
              key RATE and a secure-bit total - not key material, so there is no run-derived key to
              feed here and manufacturing one would be worse than a stated constant. This mixes the
              same fixed QKD-domain vector the backend uses (b"q"*32), which exercises the combiner
              path and keeps the two implementations' fingerprints comparable. */}
          <label className="muted"
            title={"Mixes a FIXED QKD-domain constant (0x71 repeated 32 times, matching the backend's "
              + "b'q'*32), not this run's key: the BB84/E91 simulation is analytic and yields secure-bit "
              + "COUNTS rather than key bytes. What it demonstrates is the PQC+QKD combiner path and "
              + "that the derived-key fingerprint below changes when a second input is mixed in."}>
            <input type="checkbox" checked={mcpQkd} onChange={(e) => setMcpQkd(e.target.checked)} /> mix QKD-domain key
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

      {stale && (
        <div className="note" style={{ marginTop: 12, borderColor: "var(--warn)", color: "var(--warn)" }}>
          ⚠ The pipeline has run again since this answer was produced, so the response and the signed
          MCP envelope below describe an earlier run. Press <b>Run in-browser agent</b> to regenerate
          them against the current numbers.
        </div>
      )}
      {(run || mcp) && (
        <div className="grid cols-2" style={{ marginTop: 16, opacity: stale ? 0.6 : 1 }}>
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
        {/* A disabled control has to say WHY, at the point of disablement - the four simulator
            exports above already do this. Without it the button just reads as broken. */}
        <button className="btn ghost" disabled={!run}
          title={!run ? "Run the in-browser agent first - there is no result to download yet" : undefined}
          onClick={() => downloadJSON({
            usecase, tool: mcpToolName, run, mcp,
            // The download outlives the page, so it has to carry its own provenance: which run's
            // numbers produced it, and whether the page had moved on by the time it was saved.
            run_context: ranWithCtx,
            stale_at_download: stale,
            // Provenance metadata on the downloaded artifact. Deliberately NOT labelled as an
            // Article 50(2) marking layer, because researching the actual instruments showed this
            // system does not need one - and because a marking layer without a detector is worse
            // than none.
            //
            // The Commission's Code of Practice on Transparency of AI-generated Content (final
            // 2026-06-10, assessed adequate 8-9 July 2026) is the recognised route to 50(2). Its
            // chain for this exact case: FREE-FORM TEXT - "the text that is shown on a website or
            // within a chat" - cannot transport metadata, so watermarking is the only layer; and
            // Sub-measure 1.1.2 excepts "very short text", which the glossary defines as text
            // SHORTER THAN 200 TOKENS. This panel generates 96 tokens (Transformers.js) or 160
            // (WebLLM), both under that threshold.
            //
            // And the Guidelines (point 70) are explicit that marking without the means to detect
            // it "will not suffice". Shipping a marking layer we could not detect would create an
            // obligation rather than discharge one. So this block is voluntary provenance, useful
            // for anyone auditing a saved artifact, and it says so.
            //
            // The obligation that DOES bite is Art. 50(1), which has shipped since Round 21 and has
            // no grace period. Gap analysis: docs/ai-act-art50-gap-analysis.md
            provenance_note: {
              ai_generated: true,
              generator: { engine: run?.engine ?? null, model: run?.model ?? null,
                           runtime: "in-browser" },
              basis: "voluntary provenance metadata, not an Article 50(2) marking layer",
              why_no_marking_layer:
                "free-form text under 200 tokens is 'very short text' under the Commission Code of "
                + "Practice (Sub-measure 1.1.2 and glossary), which excepts it from watermarking; "
                + "free-form text cannot carry metadata, so no marking layer applies",
              art_50_1_disclosure: "shown unconditionally on this panel",
            },
          }, `qsh-agentic-${usecase ?? "ai"}.json`)}>
          ⬇ Download results (JSON)
        </button>
      </div>
    </div>
  );
}

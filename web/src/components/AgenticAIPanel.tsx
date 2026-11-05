import { useEffect, useState } from "react";
import {
  type AgentRun, DEFAULT_DEVICE, type Device, type Engine, hasWebGPU, runAgent, tfjsModelLabel,
  TFJS_MODELS, WEBLLM_MODELS,
} from "../lib/clientAI";
import { checkProvenance, type ProvenanceCheck, signProvenance } from "../lib/aiProvenance";
import { clientSecureWrap, type McpResult } from "../lib/clientMcp";
import { isSoftwareAdapter, webgpuInfo } from "../lib/onnxInference";
import { cryptoCfg } from "../lib/config";
import { downloadJSON } from "../lib/export";

// Reusable in-browser agentic-AI + quantum-safe MCP panel. A small LLM runs entirely client-side
// (Transformers.js on WASM by default or WebGPU on request; WebLLM on WebGPU) and its
// request+response are sealed+signed over the client
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
  // NO QUANTUM-ADVANTAGE SCORE IN THE CONTEXT. This line used to append "quantum advantage 10/10",
  // and because the string is handed to a language model as run context, the model was invited to
  // repeat a score whose inputs measured different things on the two sides. The score is retired
  // from display (see `quantumAdvantage` in clientCompute.ts); its absence here is deliberate.
  if (!bits.length) return "";
  return `Context from this ${usecase ?? "pipeline"} run — ${bits.join(", ")}.`;
}

export default function AgenticAIPanel({
  usecase, defaultPrompt, mcpToolName = "ai_inference", runContext = null, inline = false,
}: AgenticAIPanelProps) {
  const [engine, setEngine] = useState<Engine>("transformers");
  const [model, setModel] = useState(TFJS_MODELS[0].id);
  // Transformers.js backend. WASM unless the visitor picks WebGPU: on the machine measured, WebGPU ran
  // distilGPT-2 about 80 times slower per token and no first run finished inside the timeout
  // (register, Round 59). A visitor's choice, so it survives a use-case change like engine and model.
  const [device, setDevice] = useState<Device>(DEFAULT_DEVICE);
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
  const [generatedAt, setGeneratedAt] = useState<string | null>(null);   // when the model finished
  const [saved, setSaved] = useState<{ name: string; check: ProvenanceCheck } | null>(null);
  const [combiner, setCombiner] = useState<string>(cryptoCfg().hybrid_combiner_mode ?? "hkdf");   // PQC+QKD key combiner
  const [gpu, setGpu] = useState(false);                 // genuine WebGPU (adapter obtainable), resolved async
  // "absent" and "software" are different facts and the note below used to call both of them
  // "not detected". Since hasWebGPU() began declining software rasterisers - they run this class of
  // model about 300x slower than the wasm build - that wording would have been actively wrong on a
  // machine where an adapter exists and was refused on purpose.
  const [gpuKind, setGpuKind] = useState<"usable" | "software" | "absent">("absent");
  // THE PROMPT FOLLOWS THE USE CASE. `useState(defaultPrompt)` reads its argument on the FIRST render
  // only, and the route component that mounts this panel stays mounted when a visitor moves between
  // use cases in the nav - so after Finance -> Medical the medical page still asked the AML question
  // ("why must an AML agent that scores a transaction across two banks..."), before and after a run.
  // A hard load showed the right one, which is why it survived: every check that started from a fresh
  // URL passed. A previous use case's answer and MCP envelope are cleared with it, because they
  // describe a run that is no longer on the page. Engine and model are the visitor's choices and stay.
  useEffect(() => {
    setPrompt(defaultPrompt || FALLBACK_PROMPT);
    setRun(null); setMcp(null); setProgress(""); setRanWithCtx(null);
  }, [usecase, defaultPrompt]);
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
    const effectivePrompt = ctxLine ? `${ctxLine}\n\n${prompt}` : prompt;
    const r = await runAgent(engine, effectivePrompt, { model, device, onProgress: setProgress });
    setRun(r); setGeneratedAt(new Date().toISOString());
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

  // The JSON download. A file is a data format that can carry metadata, so it carries the signed record
  // the Code of Practice's Sub-measure 1.1.1 asks of exported content (Round 86). Its Sub-measure 1.1.2
  // watermark is still excepted: both engines stop under 200 tokens, "very short text" in the Code's
  // glossary. A run that produced no text marks nothing. Gap analysis: docs/ai-act-art50-gap-analysis.md
  //
  // The obligation that binds on the page itself is Art. 50(1), shown above with no grace period.
  const download = async () => {
    const aiProvenance = run?.text
      ? await signProvenance({ text: run.text, engine: run.engine, model: run.model,
                               generatedAt: generatedAt ?? new Date().toISOString() })
      : null;
    downloadJSON({
      usecase, tool: mcpToolName, run, mcp,
      // The download outlives the page, so it has to carry its own provenance: which run's
      // numbers produced it, and whether the page had moved on by the time it was saved.
      run_context: ranWithCtx,
      stale_at_download: stale,
      ai_provenance: aiProvenance,
      ai_provenance_note: {
        covers: "the signature covers `body`, whose content_sha256 is the SHA-256 of run.text",
        key: "generated in the browser tab for this file and discarded: the signature shows the record "
          + "and the text are unchanged since signing, not who signed them; anyone can edit a copy and "
          + "sign it again under a new key, or delete the record",
        time: "generated_at and signed_at come from the visitor's clock, with no time authority",
        watermark: "none: the text is under 200 tokens, 'very short text' under the Commission Code of "
          + "Practice (Sub-measure 1.1.2 and glossary)",
        check: "the 'Check a saved file' control on this panel, or scripts/verify_ai_provenance.py in the "
          + "source repository; the format is in docs/ai-act-art50-gap-analysis.md",
      },
      art_50_1_disclosure: "shown unconditionally on this panel",
    }, `qsh-agentic-${usecase ?? "ai"}.json`);
  };

  // Read in this tab, never uploaded. Any file may arrive here, so nothing in it is trusted.
  const onCheckFile = async (file: File | undefined) => {
    if (!file) return;
    let doc: unknown = null;
    try {
      doc = JSON.parse(await file.text());
    } catch {
      setSaved({ name: file.name, check: { found: false, signature_valid: false, text_matches: false,
        copy_matches: null, ai_generated: null, record: null, problem: "This file is not JSON." } });
      return;
    }
    setSaved({ name: file.name, check: await checkProvenance(doc) });
  };

  return (
    <div className={inline ? "card" : undefined} style={inline ? { marginTop: 16 } : undefined}>
      {inline && (
        <>
          <h3 style={{ marginTop: 0 }}>Agentic AI · quantum-safe MCP</h3>
          <p className="muted" style={{ fontSize: 12, marginTop: -4 }}>
            A small LLM runs <b>in your browser</b>; the prompt and the reply are sealed and signed over a
            client-side quantum-safe MCP channel, as the arguments of a <code>tools/call</code> request named
            after this {usecase} use case&apos;s <code>{mcpToolName}</code> tool. No tool is executed, so no
            MCP response follows.
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
              {/* "will run on the slower WASM backend" until Round 60. WASM is the default backend
                  for Transformers.js everywhere now, and on the machine measured it was the faster
                  one by about 80 times per token. */}
              ⚠ WebGPU not detected — WebLLM is disabled. Transformers.js runs on WASM, which is its
              default backend anyway.
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
          {engine === "transformers" && (
            <>
              <label className="muted" htmlFor={`ai-device-${usecase ?? "std"}`}>Backend</label>
              <select id={`ai-device-${usecase ?? "std"}`} value={device} disabled={busy}
                onChange={(e) => setDevice(e.target.value as Device)}
                title={"WASM is the default. On the machine measured, WebGPU ran distilGPT-2 about 80 "
                  + "times slower per token than WASM and downloads larger fp32 weights; other GPUs and "
                  + "browsers were not measured, so WebGPU stays selectable to try on this one."}>
                <option value="wasm">WASM (default)</option>
                <option value="webgpu" disabled={!gpu}>WebGPU{gpu ? "" : " (no usable adapter)"}</option>
              </select>
            </>
          )}
          <label className="muted" htmlFor={`ai-model-${usecase ?? "std"}`}>Model</label>
          <select id={`ai-model-${usecase ?? "std"}`} value={model} onChange={(e) => setModel(e.target.value)}>
            {/* Transformers.js sizes are COMPUTED from the bytes the config records for the selected
                backend's weights file; the label used to say "tiny, ~80 MB", which matched no file
                the library loads. */}
            {engine === "transformers"
              ? TFJS_MODELS.map((m) => <option key={m.id} value={m.id}>{tfjsModelLabel(m, device)}</option>)
              : models.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
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
            <input type="checkbox" checked={mcpQkd} onChange={(e) => setMcpQkd(e.target.checked)} /> mix fixed QKD-domain constant
          </label>
          <label className="muted" htmlFor={`ai-combiner-${usecase ?? "std"}`}>Combiner</label>
          <select id={`ai-combiner-${usecase ?? "std"}`} value={combiner} disabled={!mcpQkd}
            onChange={(e) => setCombiner(e.target.value)}
            title="PQC+QKD key combiner (crypto.hybrid_combiner_mode). HKDF = dual-PRF: the session key holds if either input does, under computational assumptions. Toeplitz+QLHL = a modified-Toeplitz strong extractor. Its leakage-resilience property (if the combined output AND one input key leak, the other retains quantifiable min-entropy) needs a seed independent of the input (Giestinhas & Spiller, arXiv:2603.26907, Lemma 2). Here the seed is SHAKE-256 of the inputs themselves, so that condition fails and the combiner is computational, like HKDF. Applies when 'mix fixed QKD-domain constant' is on; the derived-key fingerprint below changes with the choice.">
            <option value="hkdf">HKDF (dual-PRF)</option>
            <option value="its_toeplitz">Toeplitz+QLHL (computationally seeded)</option>
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
                  <tr><th scope="row">engine</th><td>{run.engine}</td></tr>
                  <tr><th scope="row">model</th><td className="mono">{run.model}</td></tr>
                  <tr><th scope="row">backend</th><td>{run.backend}{run.webgpu ? " (WebGPU)" : ""}</td></tr>
                  <tr><th scope="row">elapsed</th><td>{run.elapsedMs} ms</td></tr>
                </tbody>
              </table>
            )}
          </div>
          <div className="card">
            <h3>Quantum-safe MCP envelope</h3>
            <p className="muted" style={{ fontSize: 12 }}>
              The prompt and the reply, as the arguments of a signed, sealed MCP <code>tools/call</code>{" "}
              request named after the tool. Both endpoints run in this tab; no tool is executed, so no MCP
              response follows.
            </p>
            {mcp && (
              <table style={{ marginTop: 8 }}>
                <tbody>
                  <tr><th scope="row">session</th><td className="mono">{mcp.session}</td></tr>
                  <tr><th scope="row">tool</th><td className="mono">{mcpToolName}</td></tr>
                  <tr><th scope="row">KEM / QKD-domain constant mixed</th><td>{mcp.kem} / {String(mcp.qkd_mixed)}</td></tr>
                  <tr><th scope="row">PQC+QKD combiner</th><td className="mono">{mcp.combiner}</td></tr>
                  <tr><th scope="row">derived key fingerprint</th><td className="mono">{mcp.session_key_fp}</td></tr>
                  <tr><th scope="row">method / MCP revision</th><td className="mono">{mcp.method} / {mcp.protocol_version}</td></tr>
                  <tr><th scope="row">authenticated ({mcp.signature_alg})</th>
                    <td style={{ color: mcp.authenticated ? "var(--accent-2)" : "var(--danger)" }}>{String(mcp.authenticated)}</td></tr>
                  <tr><th scope="row">tampered message rejected</th>
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
          onClick={download}>
          ⬇ Download results (JSON)
        </button>
      </div>
      {!run && (
        // Visible, not only a tooltip: a title attribute is invisible on touch screens and to most
        // keyboard users, which is what the simulator exports above already avoid.
        <p className="muted small" style={{ marginTop: 6, marginBottom: 0 }}>
          Run the in-browser agent first - there is no result to download yet.
        </p>
      )}
      <p className="muted small" style={{ marginTop: 8, marginBottom: 0 }}>
        The download carries a signed record that its text is AI-generated (ML-DSA-65, under a key made in
        this tab for that file). The record shows the text is unchanged since signing, not who signed it.
      </p>
      <div className="btn-row" style={{ marginTop: 8 }}>
        <label className="muted" htmlFor={`ai-check-${usecase ?? "std"}`}>Check a saved file (read in this tab, not uploaded)</label>
        <input id={`ai-check-${usecase ?? "std"}`} type="file" accept="application/json,.json"
          onChange={(e) => onCheckFile(e.target.files?.[0])} />
      </div>
      {saved && (
        <table style={{ marginTop: 8 }}>
          <tbody>
            <tr><th scope="row">file</th><td className="mono">{saved.name}</td></tr>
            <tr><th scope="row">AI-generated record</th>
              <td>{saved.check.found ? "found (checked by its signed metadata; no watermark or forensic detection is involved)"
                : "none, which shows nothing either way: the check does not read the text itself"}</td></tr>
            {saved.check.found && (
              <>
                <tr><th scope="row">signature verifies</th>
                  <td style={{ color: saved.check.signature_valid ? "var(--accent-2)" : "var(--danger)" }}>{String(saved.check.signature_valid)}</td></tr>
                <tr><th scope="row">text unchanged since signing</th>
                  <td style={{ color: saved.check.text_matches ? "var(--accent-2)" : "var(--danger)" }}>{String(saved.check.text_matches)}</td></tr>
              </>
            )}
            {saved.check.copy_matches === false && (
              <tr><th scope="row">MCP envelope&apos;s copy of the text</th>
                <td style={{ color: "var(--danger)" }}>differs from the checked text</td></tr>
            )}
            {saved.check.record && (
              <>
                <tr><th scope="row">marked AI-generated</th><td>{String(saved.check.ai_generated)}</td></tr>
                <tr><th scope="row">model</th><td className="mono">{String(saved.check.record.generator?.model)}</td></tr>
                <tr><th scope="row">generated at (the signer&apos;s clock)</th><td className="mono">{String(saved.check.record.generated_at)}</td></tr>
              </>
            )}
            {saved.check.problem && <tr><th scope="row">result</th><td>{saved.check.problem}</td></tr>}
          </tbody>
        </table>
      )}
    </div>
  );
}

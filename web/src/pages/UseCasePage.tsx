import { useRef } from "react";
import { useParams } from "react-router-dom";
import AgenticAIPanel from "../components/AgenticAIPanel";
import BenchmarkPanel from "../components/BenchmarkPanel";
import { ModelPanel } from "../components/ModelPanel";
import { ProvenanceBadge, type Provenance } from "../components/ProvenanceBadge";
import PipelineControls from "../components/PipelineControls";
import RunMonitor from "../components/RunMonitor";
import ShapPanel from "../components/ShapPanel";
import TwoPartyDiagram, { USECASE_SPECS } from "../components/TwoPartyDiagram";
import { DEMO_BANNER, isClientDemo } from "../lib/demoMode";
import { USECASE_META } from "../lib/useCases";
import { usePipelineRun } from "../lib/usePipelineRun";

const DEMO = isClientDemo();

export default function UseCasePage() {
  const { id = "finance_aml" } = useParams();
  const usecase = id in USECASE_SPECS ? id : "finance_aml";
  const meta = USECASE_META[usecase];
  const diagramRef = useRef<HTMLDivElement>(null);
  const run = usePipelineRun(usecase, diagramRef);
  const { events, active, done, metrics, signature, artifacts, state, logRef, gifBusy, benchmark,
    downloadGif, downloadPng, downloadEvents, downloadMetricsCsv } = run;

  return (
    <div>
      <h1 className="page-title">{meta.title}</h1>
      <p className="page-sub">{meta.blurb}</p>
      <div className="pill-row" style={{ marginBottom: 10 }}>
        <span className="tag">{meta.industry}</span>
        <span className="muted" style={{ fontSize: 12 }}>{meta.market}</span>
      </div>
      {DEMO && <div className="note" style={{ marginBottom: 12 }}>🖥️ {DEMO_BANNER}</div>}

      {/* Live two-party simulator: Start/Stop/Restart/Pause/Step drive the diagram animation */}
      <PipelineControls run={run} />
      <div ref={diagramRef} style={{ marginTop: 16, background: "var(--panel)", borderRadius: 10, padding: 8 }}>
        <TwoPartyDiagram usecase={usecase} activeStage={active} doneStages={done} metrics={metrics} />
      </div>
      <div className="btn-row" style={{ marginTop: 10 }}>
        <button className="btn ghost" onClick={() => downloadPng(`${usecase}-two-party.png`)}>⬇ Diagram (PNG)</button>
        <button className="btn ghost" onClick={() => downloadGif(`${usecase}-simulator.gif`)} disabled={gifBusy}>
          {gifBusy ? "Encoding…" : "⬇ Animation (GIF)"}
        </button>
        <button className="btn ghost" onClick={() => downloadEvents(`${usecase}-events.json`)}>⬇ Events (JSON)</button>
        <button className="btn ghost" onClick={() => downloadMetricsCsv(`${usecase}-metrics.csv`)}>⬇ Metrics (CSV)</button>
        <span className="muted" style={{ fontSize: 12, alignSelf: "center" }}>
          ▶ Start plays the protocol · Pause/Step advance it · ↻ Restart replays
        </span>
      </div>

      <RunMonitor events={events} metrics={metrics} signature={signature} logRef={logRef} />

      {/* What produced the inference on this page: the published ONNX model, the bundled real
          model, or placeholder weights. Read from the run's own metrics, not asserted. */}
      {metrics?.inference_provenance && (
        <div className="card" style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <strong style={{ fontSize: 13 }}>Inference provenance</strong>
          <ProvenanceBadge
            kind={String(metrics.inference_provenance) as Provenance}
            detail={String(metrics.inference_backend ?? "")}
            evidence={{ model: metrics.inference_model ?? null }}
          />
        </div>
      )}

      {/* The registry's models and their measured numbers, next to the published baselines. */}
      <ModelPanel usecase={usecase} />

      {/* XAI: signed SHAP attribution for the inference use cases */}
      {meta.inference && <ShapPanel explanation={artifacts?.explanation} />}

      {/* Unified benchmark: auto-filled from the single Run above (no separate "Run benchmark" button).
          The MPC-FHE row is the live run's measured FHE-linear/MPC-non-linear split. */}
      <BenchmarkPanel compact={!meta.inference} result={benchmark}
        clientComputeMs={benchmark?.client_compute_ms ?? null} />

      {/* Agentic AI · quantum-safe MCP — per-industry panel, fed by THIS run's live metrics (Round 10).
          The report frames MCP as a per-use-case cross-cutting layer; the same panel is kept standalone. */}
      <AgenticAIPanel inline usecase={usecase} defaultPrompt={meta.agenticPrompt}
        mcpToolName={meta.agenticTool} runContext={{ metrics, benchmark }} />

      <div className="note" style={{ marginTop: 16, fontSize: 12 }}>
        State: <b>{state}</b>. The diagram above is a live two-party simulator — each protocol step lights up as the
        pipeline runs, and the benchmark below fills in from that same run (one Run, one result). Demo mode computes
        entirely in the browser (no server compute); the full deployment adds federated learning. PoC runs LAN/local;
        the diagram illustrates the WAN two-party deployment.
        <div className="muted" style={{ marginTop: 6, fontSize: 11 }}>
          Hybrid split grounded in <i>EncFormer</i> (linear→FHE, non-linear→MPC) · system-level cost per
          <i> "Beyond Latency: MPC &amp; FHE for PPML"</i> (arXiv 2604.00169, 2026) · two-party QOT per
          <i> Zhang et al.</i> (npj Quantum Information 2026). See <code>references/</code>.
        </div>
      </div>
    </div>
  );
}

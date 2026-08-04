import { useRef } from "react";
import { Link, useParams } from "react-router-dom";
import AgenticAIPanel from "../components/AgenticAIPanel";
import SdnControlPlanePanel from "../components/SdnControlPlanePanel";
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

/** Route entry. Resolves the id and renders nothing else, so the branch below never sits in front
 *  of a hook: UseCaseView owns every hook and is mounted or unmounted as a whole. Putting the guard
 *  inside UseCaseView instead would call fewer hooks on the invalid path, and React tears the tree
 *  down with "rendered fewer hooks than expected" the moment a valid id is edited into a bad one. */
export default function UseCasePage() {
  const { id } = useParams();
  // An unrecognised id used to render Finance under Finance's title while the address bar still
  // said something else, so a stale or mistyped link presented one industry's data as another's.
  // The route matches, so App's catch-all NotFound never fires; this page has to say so itself.
  if (!id || !(id in USECASE_SPECS)) return <UnknownUseCase id={id} />;
  return <UseCaseView usecase={id} />;
}

function UseCaseView({ usecase }: { usecase: string }) {
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
        <TwoPartyDiagram usecase={usecase} activeStage={active} doneStages={done} metrics={metrics}
          config={{ kem: run.kem, qkd: run.qkd, sig: run.sig }} />
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
          The report frames MCP as a per-use-case cross-cutting layer; the same panel is kept standalone.
          It is an application-plane SAE: see the control-plane panel directly below, which is the
          element that actually manages key associations (ETSI GS QKD 015 puts application interfaces
          out of scope of the control interface). */}
      <AgenticAIPanel inline usecase={usecase} defaultPrompt={meta.agenticPrompt}
        mcpToolName={meta.agenticTool} runContext={{ metrics, benchmark }} />

      <SdnControlPlanePanel
        usecase={usecase}
        distanceKm={Number(run.params.qkd_distance_km ?? 0)}
        // secure_key_bits is the key material this run actually produced, which is what the
        // controller's allocation budget is denominated in. metrics.skr is per-pulse and is not.
        keyBudgetBits={Number(metrics.secure_key_bits ?? 0)}
        qkdEnabled={run.qkd !== "pqc-only"} />

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

/** Shown when /usecase/<id> names something that does not exist.
 *
 *  Deliberately not a redirect to Finance. Substituting a working page for a wrong address hides
 *  the error at exactly the moment it matters — the visitor reads real numbers and a real dataset
 *  provenance line belonging to an industry they did not ask for. */
function UnknownUseCase({ id }: { id?: string }) {
  return (
    <section className="page">
      <h1 className="page-title">No such use case</h1>
      <p className="page-sub">
        {id ? <>This app has no use case with the id <code>{id}</code>.</> : <>No use case id was given.</>}
        {" "}These are the ones it does have:
      </p>
      <ul>
        {Object.keys(USECASE_SPECS).map((key) => (
          <li key={key} style={{ marginBottom: 4 }}>
            <Link to={`/usecase/${key}`}><code>{key}</code></Link>
            {USECASE_META[key] ? <span className="muted"> — {USECASE_META[key].title}</span> : null}
          </li>
        ))}
      </ul>
      <p><Link to="/usecases">Back to the use-case index</Link></p>
    </section>
  );
}

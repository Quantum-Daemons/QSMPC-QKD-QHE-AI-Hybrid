import { useRef, useState } from "react";
import OrchestrationGraph from "../components/OrchestrationGraph";
import PipelineControls from "../components/PipelineControls";
import RunMonitor from "../components/RunMonitor";
import { DEMO_BANNER, isClientDemo } from "../lib/demoMode";
import { USECASES as USE_CASE_META } from "../lib/useCases";
import { usePipelineRun } from "../lib/usePipelineRun";

// Derived from lib/useCases.ts, not re-typed. This list used to be its own literal array whose
// labels had drifted from the canonical ones ("Genomics — Secure summation" here versus
// "Genomics — confidential analysis" there), so the same use case was named two different things
// depending on which page you were on. "demo" is appended because it is a real pipeline flow
// (clientCompute FLOWS) with no marketing page of its own.
const USECASES: [string, string][] = [
  ...USE_CASE_META.map((u) => [u.id, u.title] as [string, string]),
  ["demo", "Generic demo"],
];

const DEMO = isClientDemo();

// Generic "run any pipeline" view. The per-use-case pages (/usecase/:id) fix the use case
// and add the two-party simulator diagram + benchmark + XAI; this page keeps the selector.
export default function Orchestration() {
  const [usecase, setUsecase] = useState("finance_aml");
  const graphRef = useRef<HTMLDivElement>(null);
  const run = usePipelineRun(usecase, graphRef);
  const hasRun = run.events.length > 0;
  const { events, active, done, metrics, signature, state, logRef, gifBusy,
    downloadGif, downloadPng, downloadEvents, downloadMetricsCsv } = run;

  return (
    <div>
      <h1 className="page-title">E2E Orchestration</h1>
      <p className="page-sub">
        Run the full quantum-safe pipeline live for any use case. Crypto algorithms (KEM / QKD /
        FHE / signature) are swappable at runtime — crypto-agility. Pause, step one stage at a
        time, and export the animated run. (Per-industry pages add a two-party simulator + benchmark.)
      </p>
      {DEMO && <div className="note" style={{ marginBottom: 12 }}>🖥️ {DEMO_BANNER}</div>}

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="btn-row">
          <label className="muted" htmlFor="sel-usecase">Use case</label>
          <select id="sel-usecase" name="usecase" aria-label="Use case" value={usecase}
            onChange={(e) => setUsecase(e.target.value)}>
            {USECASES.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
          </select>
          <span style={{ flex: 1 }} />
          <span className={`badge-state ${state}`}>{state}</span>
        </div>
      </div>

      <PipelineControls run={run} />

      <div style={{ marginTop: 16 }} ref={graphRef}>
        {/* metrics, so the inference node names the published model that ran. Without it this graph
            described the same run differently from the two-party diagram on the use-case pages. */}
        <OrchestrationGraph usecase={usecase} activeStage={active} doneStages={done}
          metrics={metrics}
          config={{ kem: run.kem, qkd: run.qkd, sig: run.sig,
                    qkdProtocol: String(run.params.qkd_protocol ?? "") }} />
      </div>

      <RunMonitor events={events} metrics={metrics} signature={signature} logRef={logRef} />

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Download artifacts</h3>
        <p>Client-side exports (server also writes JSONL/CSV/JSON to <code>results/</code>). Images are high-DPI.</p>
        <div className="btn-row">
          <button className="btn ghost" onClick={() => downloadEvents(`${usecase}-events.json`)}
            disabled={!hasRun} title={hasRun ? undefined : "Run the orchestration first"}>⬇ Events (JSON)</button>
          <button className="btn ghost" onClick={() => downloadMetricsCsv(`${usecase}-metrics.csv`)}
            disabled={!hasRun} title={hasRun ? undefined : "Run the orchestration first"}>⬇ Metrics (CSV)</button>
          <button className="btn ghost" onClick={() => downloadPng(`${usecase}-dag.png`)}>⬇ DAG (PNG, hi-DPI)</button>
          <button className="btn ghost" onClick={() => downloadGif(`${usecase}-orchestration.gif`)}
            disabled={gifBusy || !hasRun} title={hasRun ? undefined : "Run the orchestration first"}>
            {gifBusy ? "Encoding…" : "⬇ Animation (GIF)"}
          </button>
        </div>
      </div>
    </div>
  );
}

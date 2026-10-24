import { useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router";
import KeyPoints from "../components/KeyPoints";
import OrchestrationGraph from "../components/OrchestrationGraph";
import PipelineControls from "../components/PipelineControls";
import RunMonitor from "../components/RunMonitor";
import { EncryptedAgreementChart, NoEncryptedComparison, RealDataScatter } from "../components/VerificationChart";
import type { EncryptedAgreementPlot } from "../components/VerificationChart";
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

const DEFAULT_USECASE = "finance_aml";

/**
 * Read `?usecase=` and say what happened, rather than silently ignoring it.
 *
 * `/orchestration?usecase=demo` rendered `finance_aml`. Nothing links to that form, so nobody had
 * hit it - but a URL that names a use case and shows a different one is a link that lies, and it was
 * carried as an open item from Round 42 to Round 44 and then re-dated to Round 46 without being
 * discharged. The register's own gate treats moving a date as not discharging it, so this resolves
 * it instead.
 *
 * A REJECTED VALUE IS ANNOUNCED, not swallowed. Falling back silently is what the parameter already
 * did; the only difference would have been that the code now meant to. Returning the reason lets the
 * page say "there is no use case called X, showing Y" - which is the behaviour a mistyped or stale
 * link deserves, and the one that makes the fallback visible to whoever sent it.
 */
function usecaseFromQuery(raw: string | null): { id: string; rejected: string | null } {
  if (raw == null || raw === "") return { id: DEFAULT_USECASE, rejected: null };
  return USECASES.some(([id]) => id === raw)
    ? { id: raw, rejected: null }
    : { id: DEFAULT_USECASE, rejected: raw };
}

// Generic "run any pipeline" view. The per-use-case pages (/usecase/:id) fix the use case
// and add the two-party simulator diagram + benchmark + XAI; this page keeps the selector.
export default function Orchestration() {
  const [searchParams] = useSearchParams();
  // Read ONCE, as the initial value. The selector below is the source of truth afterwards, so a
  // visitor who changes it does not have their choice reverted by a stale query string on re-render.
  const initial = useMemo(() => usecaseFromQuery(searchParams.get("usecase")), []);
  const [usecase, setUsecase] = useState(initial.id);
  const graphRef = useRef<HTMLDivElement>(null);
  const run = usePipelineRun(usecase, graphRef);
  const hasRun = run.events.length > 0;
  const { events, active, done, metrics, signature, state, logRef, gifBusy, artifacts, inFlight,
    downloadGif, downloadPng, downloadEvents, downloadMetricsCsv } = run;

  return (
    <div>
      <h1 className="page-title">E2E Orchestration</h1>
      {/* "Crypto algorithms (KEM / QKD / FHE / signature) are swappable at runtime" until Round 58:
          the demo's FHE backend is fixed (node-seal), so FHE is swappable only in the full deployment. */}
      <KeyPoints items={[
        "Run the full quantum-safe pipeline live for any use case; pause, step one stage at a time, and export the animated run.",
        "Swappable at run time: KEM, signature, key source and the QKD protocol (simulated). The FHE backend is selectable only in the full local deployment.",
        "The per-industry pages add a two-party simulator and the benchmark.",
      ]} />
      {DEMO && <div className="note" style={{ marginBottom: 12 }}>🖥️ {DEMO_BANNER}</div>}
      {/* The rejected-parameter notice. Silence here would be indistinguishable from the defect
          this replaced - the page would still show a use case the URL did not name, and the only
          change would be that it meant to. Naming the bad value is what lets whoever followed the
          link tell a typo from a stale bookmark. */}
      {initial.rejected != null && (
        <div className="note" style={{ marginBottom: 12 }}>
          There is no use case called <b>{initial.rejected}</b>. Showing{" "}
          <b>{USECASES.find(([id]) => id === initial.id)?.[1] ?? initial.id}</b> instead - pick
          another from the selector below.
        </div>
      )}

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="btn-row">
          <label className="muted" htmlFor="sel-usecase">Use case</label>
          {/* Disabled mid-run. It was live, and switching during a run redrew the graph for the NEW
              use case while `active`/`done` still carried the old flow's stage ids, swapped the
              Parameters panel, and - worst - relabelled the export filenames, so a downloaded
              artefact named an industry it does not describe. The published-model select two files
              over has always been gated this way; this one was not. */}
          <select id="sel-usecase" name="usecase" aria-label="Use case" value={usecase}
            disabled={inFlight}
            title={inFlight ? "Stop or finish the run before switching use case" : undefined}
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

      {/* The standing requirement is a 2D plot OF PRIVATE INFERENCE against the plaintext reference.
          It rendered on the use-case pages and not here, on the page that actually runs the
          end-to-end pipeline - so the one place a visitor watches the whole thing execute was the
          one place with no evidence of what the encrypted tier produced.

          `run.artifacts` already carried the payload and `EncryptedAgreementChart` already took it;
          nothing new is computed or drawn. It is conditional on the artifact rather than on the use
          case, so a flow that runs no private inference shows nothing instead of an empty axis - the
          same predicate `UseCasePage` uses, which keeps the two pages from disagreeing about whether
          this run produced a plot. */}
      {artifacts?.encrypted_agreement_plot ? (
        <div className="card" style={{ marginTop: 16 }}>
          <h3 style={{ margin: "0 0 4px" }}>Private inference against plaintext, this run</h3>
          <EncryptedAgreementChart
            plot={artifacts.encrypted_agreement_plot as EncryptedAgreementPlot} />
        </div>
      ) : hasRun ? (
        // A card that vanishes and a flow that performs no private inference look identical, and on
        // `satellite_comms` the second is the truth - no stage there produces the plot. Saying so is
        // the difference between "this page has nothing to show you" and "this run had nothing to
        // show", which are different facts about the demo.
        //
        // ONE COPY, shared with the use-case page. This text lived here and the use-case page
        // rendered null, so the same absence was explained on one surface and silent on the other.
        <NoEncryptedComparison usecase={usecase} />
      ) : null}

      {/* The real-data overlay the use-case pages carry. `RealDataScatter` needs only the use case -
          it fetches its own payload - so the E2E page could always have shown the corpus its model is
          scored on, its separability, its licence and its attribution.
          `card` as a PROP rather than a `<div className="card">` here. The component renders null
          where no overlay ships; the wrapper did not, so `demo` drew an empty bordered box after
          every run for a round - while these very lines claimed it rendered nothing, and `OR12`,
          written in the same commit, promised the same. The frame now belongs to the only code that
          can know whether there is anything to frame. */}
      {hasRun ? <RealDataScatter usecase={usecase} card /> : null}

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Download artifacts</h3>
        {/* The parenthetical described the FULL deployment and sat twenty-nine lines below a banner saying
            no computation is sent to a server. In the client-only build there is no server and no
            `results/`, so it contradicted the page it was on. */}
        <p>Client-side exports{isClientDemo() ? "" : " (the full deployment also writes JSONL/CSV/JSON to results/)"}. Images are high-DPI.</p>
        {/* THE REASON, VISIBLE. Three of these four buttons carry their reason only in a `title`,
            and `disabled` takes a button out of the tab order entirely - so the reason was reachable
            by hovering a mouse and by nothing else. A keyboard or screen-reader visitor got a
            control announced as unavailable with no statement of what would make it available.
            One sentence above the row says it once, to everyone, instead of three times to some. */}
        {!hasRun && (
          <p className="muted small" style={{ marginTop: -4 }}>
            Run the orchestration first - the export buttons below stay unavailable until a run has
            produced events. The DAG image can be exported at any time.
          </p>
        )}
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

      {/* Run status, announced and visible from anywhere on the page.
          The identical block has been on `UseCasePage` since Round 39 and this page never got one,
          although this is the page whose entire subject is watching a pipeline advance: the state
          badge lives in the controls card at the top and scrolls out of view, and the DAG's own
          progress was - until this round - encoded in colour alone. So a visitor reading the lower
          half of the page had no announcement of a stage change and no non-colour signal either.
          `role="status"` with `aria-live="polite"` announces without stealing focus. Rendered only
          while it says something: during a run and while paused. */}
      {(run.running || run.paused) && (
        <div role="status" aria-live="polite"
          style={{
            position: "fixed", right: 16, bottom: 16, zIndex: 50,
            background: "var(--panel)", border: "1px solid var(--accent)", borderRadius: 10,
            padding: "8px 14px", boxShadow: "0 4px 18px rgba(0,0,0,.45)",
            display: "flex", alignItems: "center", gap: 10, fontSize: 12,
          }}>
          <span className={`badge-state ${run.state}`}>{run.state}</span>
          {run.active ? <span className="muted"><code>{run.active}</code></span> : null}
        </div>
      )}
    </div>
  );
}

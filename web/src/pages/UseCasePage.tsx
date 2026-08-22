import { EncryptedAgreementChart, MeasuredAgainstPublished, NoEncryptedComparison, VerificationChart } from "../components/VerificationChart";
import type { EncryptedAgreementPlot } from "../components/VerificationChart";
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
import { FIBER, FREE_SPACE } from "../lib/quantumSim";
import { DEMO_BANNER, isClientDemo } from "../lib/demoMode";
import { hasTreeAttribution, modelsFor } from "../lib/config";
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
  // Before a run there is nothing to export: GIF silently did nothing (the recorder is only
  // built by launch()) and Metrics wrote an empty CSV. BenchmarkPanel already gates on
  // rows.length; these now match it.
  const hasRun = run.events.length > 0;
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
        {/* The protocol and the QSMPC op live in params, not in the crypto-agility family selects.
            Passing only the family is what let an E91 run draw a BB84 arrow. */}
        <TwoPartyDiagram usecase={usecase} activeStage={active} doneStages={done} metrics={metrics}
          config={{ kem: run.kem, qkd: run.qkd, sig: run.sig,
                    qkdProtocol: String(run.params.qkd_protocol ?? ""),
                    qsmpcOp: String(run.params.qsmpc_op ?? ""),
                    optiqkd: run.optiqkd, workload: run.workload,
                    inference: String(run.params.inference ?? "") }} />
      </div>
      {/* THE REASON, VISIBLE - the same fix as `Orchestration.tsx`'s export row and for the same
          reason. `disabled` removes a button from the tab order, so a reason living only in `title`
          is reachable by hovering a mouse and by nothing else. Both pages were swept together
          because a fix applied to one export row and not the other is exactly how the agreement
          scatter went three rounds without the shape channel its four siblings were given. */}
      {!hasRun && (
        <p className="muted small" style={{ marginTop: 10, marginBottom: 0 }}>
          Run the simulator first - the animation, events and metrics exports stay unavailable until
          a run has produced events. The diagram image can be exported at any time.
        </p>
      )}
      <div className="btn-row" style={{ marginTop: 10 }}>
        <button className="btn ghost" onClick={() => downloadPng(`${usecase}-two-party.png`)}>⬇ Diagram (PNG)</button>
        <button className="btn ghost" onClick={() => downloadGif(`${usecase}-simulator.gif`)}
          disabled={gifBusy || !hasRun} title={hasRun ? undefined : "Run the simulator first"}>
          {gifBusy ? "Encoding…" : "⬇ Animation (GIF)"}
        </button>
        <button className="btn ghost" onClick={() => downloadEvents(`${usecase}-events.json`)}
          disabled={!hasRun} title={hasRun ? undefined : "Run the simulator first"}>⬇ Events (JSON)</button>
        <button className="btn ghost" onClick={() => downloadMetricsCsv(`${usecase}-metrics.csv`)}
          disabled={!hasRun} title={hasRun ? undefined : "Run the simulator first"}>⬇ Metrics (CSV)</button>
        <span className="muted" style={{ fontSize: 12, alignSelf: "center" }}>
          ▶ Start plays the protocol · Pause/Step advance it · ↻ Restart replays
        </span>
      </div>

      <RunMonitor events={events} metrics={metrics} signature={signature} logRef={logRef} />

      {/* What produced the inference on this page: the published ONNX model, the bundled real
          model, or placeholder weights. Read from the run's own metrics, not asserted. */}
      {/* BOTH inferences, because the run performs two and this panel used to name only one - the
          smaller one. It read "MEASURED (plain JS) - js-linear (bundled real model)" on a run that
          had also just executed a pinned ResNet-18, so the panel headed "Inference provenance"
          described the least of what the page had done and the published model appeared nowhere
          above the Models panel. They are different corpora and neither substitutes for the other,
          so both are labelled rather than merged into one badge. */}
      {(metrics?.inference_provenance || metrics?.published_model) && (
        <div className="card" style={{ display: "grid", gap: 8 }}>
          <strong style={{ fontSize: 13 }}>Inference provenance</strong>
          {metrics?.inference_provenance && (
            <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
              <span className="muted" style={{ fontSize: 12, minWidth: 190 }}>
                Pipeline model, on this page's own data
              </span>
              <ProvenanceBadge
                kind={String(metrics.inference_provenance) as Provenance}
                detail={String(metrics.inference_backend ?? "")}
                evidence={{ model: metrics.inference_model ?? null }}
              />
            </div>
          )}
          {metrics?.published_model && (
            <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
              <span className="muted" style={{ fontSize: 12, minWidth: 190 }}>
                Published model, on its own corpus
              </span>
              {metrics.published_model_status === "failed" ? (
                <span className="small" style={{ color: "var(--warn)" }}>
                  {String(metrics.published_model)} did not run — no substitute is shown
                </span>
              ) : (
                <ProvenanceBadge
                  kind={(metrics.published_model_backend === "onnx-webgpu"
                    ? "measured-webgpu" : "measured-wasm") as Provenance}
                  detail={String(metrics.published_model_backend ?? "")}
                  evidence={{
                    model: String(metrics.published_model),
                    revision: metrics.published_model_revision ?? null,
                    input: metrics.published_input_shape ?? null,
                    latency_ms: metrics.published_inference_ms ?? null,
                  }}
                />
              )}
            </div>
          )}
        </div>
      )}

      {/* The registry's models and their measured numbers, next to the published baselines - and a
          Run button per model, or the reason it has none. This absorbed PublishedModelPanel, which
          used to sit forty lines below and offer the same run again with different chrome. */}
      <ModelPanel usecase={usecase} />

      {/* XAI: signed SHAP attribution.
          Gated on an explanation EXISTING, not on `meta.inference`. The flag meant "has a
          classical-vs-quantum private-inference path", which is a different question: satellite runs
          a genuinely linear anomaly model and its attribution was computed and then dropped on the
          floor because the flag said no. Ask whether there is something to show. */}
      {/* What the in-run verification measured, drawn. Gated on a plot EXISTING rather than on the
          task type: a model whose sample ships no labels produces no plot and should render nothing
          rather than an empty axis. */}
      {/* The VERDICT belongs here, next to the chart. `artifacts.verification` is written whether or
          not the model matched its recorded figure, and the only component that rendered
          `matchesExpected` was `ModelEvalReport` - defined, never referenced, and tree-shaken out of
          the bundle. So a verification FAIL drew this card identically to a PASS, captioned
          "Re-scored in this tab", with the failing verdict reachable only as a raw `verification |
          FAIL` row in the metrics table. A card that looks the same whether the check passed or
          failed is worse than no card. */}
      {/* What PRIVATE inference produced. Its own card, gated only on its own artifact: nesting it
          inside the verification card above made a figure about the ENCRYPTED path conditional on
          the PLAINTEXT published model having produced a plot, which is a different question. */}
      {artifacts?.encrypted_agreement_plot ? (
        <div className="card">
          <h3 style={{ margin: "0 0 4px" }}>Private inference against plaintext, this run</h3>
          <EncryptedAgreementChart
            plot={artifacts.encrypted_agreement_plot as EncryptedAgreementPlot} />
        </div>
      ) : hasRun ? (
        <NoEncryptedComparison usecase={usecase} />
      ) : null}

      {artifacts?.verification?.plot ? (
        <div className="card">
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <h3 style={{ margin: 0 }}>Verification, this run</h3>
            {artifacts.verification.matchesExpected != null && (
              <span className={`badge-state ${artifacts.verification.matchesExpected ? "done" : "error"}`}>
                {artifacts.verification.matchesExpected ? "PASS" : "FAIL"}
              </span>
            )}
          </div>
          <p className="muted small" style={{ marginTop: -4 }}>
            Re-scored in this tab on the model's own published held-out sample. Charts are drawn from
            the same arrays the metric was computed on.
          </p>
          {artifacts.verification.matchesExpected === false && (
            <div className="note" style={{ marginBottom: 10, borderColor: "var(--danger)", color: "var(--danger)" }}>
              ⚠ This tab measured <b>{Number(artifacts.verification.score).toFixed(4)}</b> where{" "}
              <b>{Number(artifacts.verification.expected).toFixed(4)}</b> was recorded, over{" "}
              {String(artifacts.verification.n)} published rows. Something differs - preprocessing,
              quantisation, or the execution provider - so treat the number on this page as
              <b> unverified rather than as a result</b>, and read the chart below as what was actually
              computed here rather than as a reproduction of the published figure.
            </div>
          )}
          <VerificationChart
            res={artifacts.verification as never}
            modelKey={String(artifacts.verification.modelKey ?? "")}
            usecase={usecase}
          />
          {/* Measured against published, on ONE metric. The verification chart's own axes are
              precision/recall or per-class F1, neither of which a published aggregate belongs on -
              so the comparison the standing request asks for gets its own axis rather than a line
              drawn across a chart that means something else. */}
          {(() => {
            const spec = modelsFor(usecase).find(
              (m) => m.key === String(artifacts.verification.modelKey ?? ""));
            const rec = (spec?.achieved ?? {}) as Record<string, unknown>;
            const measured = typeof rec[spec?.metric?.name ?? ""] === "number"
              ? (rec[spec?.metric?.name ?? ""] as number) : null;
            return spec ? (
              <MeasuredAgainstPublished
                measured={measured}
                published={spec.metric?.baseline ?? null}
                ceiling={spec.metric?.ceiling ?? null}
                metric={spec.metric?.name ?? ""}
                modelKey={spec.key}
                source={spec.metric?.baseline_source}
              />
            ) : null;
          })()}
          {/* The sample's own note on why the figure is what it is. Unrendered, the finance student
              once showed 0.3337 here beside a recorded 0.0178 for the same metric, with nothing on
              screen to reconcile them. */}
          {artifacts.verification.note ? (
            <p className="muted" style={{ fontSize: 11, marginTop: 10 }}>
              <b>About this figure.</b> {String(artifacts.verification.note)}
            </p>
          ) : null}
          {artifacts.verification.provenance ? (
            <p className="muted" style={{ fontSize: 11, marginTop: 6 }}>
              {String(artifacts.verification.provenance)}
            </p>
          ) : null}
        </div>
      ) : null}

      {/* `surfacedBackend` is the backend that RAN, not the control that was set. Passing
          `run.params.inference` here made the panel assert "the class comes from the quantum circuit"
          on a page where no trained weights ship, so the class had actually come from CKKS. The
          selector states an intention; only the artifact states what happened. */}
      {artifacts?.explanation?.top_features?.length ? (
        <ShapPanel
          explanation={artifacts?.explanation}
          provenance={artifacts?.explanation_provenance ?? null}
          baseline={artifacts?.explanation_baseline ?? null}
          publishedModel={metrics?.published_model ? String(metrics.published_model) : null}
          surfacedBackend={artifacts?.primary_inference?.backend ?? null}
          isOfReportedClass={artifacts?.explanation_is_of_reported_class ?? null}
          hasTreeAttribution={hasTreeAttribution(usecase)}
        />
      ) : null}

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

      {usecase === "medical_fl" && (
        <div className="note" style={{ marginTop: 12, fontSize: 12 }}>
          Federated learning runs on every run here - there is no toggle. A real client-side FedAvg
          over the page's own bundled Wisconsin sample, not a synthetic stand-in, so the accuracy it
          reports describes the data shown above it.
          <div style={{ marginTop: 6 }}>
            <b>The encrypted path evaluates the published student itself</b> - all 3,226 parameters
            of it, fetched from its own pinned Hugging Face revision and run end to end under CKKS:
            conv, average pool, and both dense layers, with im2col patches packed into a single
            ciphertext and reduced by rotate-and-add. The run reports the total rotations it
            performed, because key-switching - one per rotation - is what dominates CKKS inference
            cost. Nothing here is multiplied by a cost model, and if the weights cannot be fetched
            the run says so rather than substituting invented ones.
          </div>
          <div style={{ marginTop: 6 }}>
            There is no Workload selector any more. It offered Linear against CNN, and once CNN
            began running the real student, Linear was the only fabricated option left - an
            8-feature dot product with a hand-written weight pattern matching no trained model.
          </div>
        </div>
      )}

      <SdnControlPlanePanel
        usecase={usecase}
        // The satellite pipeline runs its QKD stage over FREE_SPACE, not the fibre channel the
        // qkd_distance_km parameter drives, so feeding that parameter here drew a 50 km terrestrial
        // fibre span underneath an 800 km free-space downlink. Take the distance from whichever
        // channel this use case actually models.
        //
        // And take it from the RUN, not from the config default. Reading FREE_SPACE.distanceKm here
        // fixed the wrong-CHANNEL bug and left a wrong-VALUE one: setting Downlink to 2000 km moved
        // the QKD stage and the diagram while this panel kept printing a single 800 km span, so two
        // panels described different geometries for the same run.
        distanceKm={usecase === "satellite_comms"
          ? Number(run.params.sat_distance_km ?? FREE_SPACE.distanceKm)
          : Number(run.params.qkd_distance_km ?? FIBER.distanceKm)}
        // A downlink is one line of sight: there is no mid-span trusted relay to insert, however
        // long it is. Terrestrial fibre is the opposite, which is why the relay caveat exists.
        maxSpanKm={usecase === "satellite_comms" ? Infinity : undefined}
        // secure_key_bits is the key material this run actually produced, which is what the
        // controller's allocation budget is denominated in. `skr_per_pulse` is per-pulse and is not.
        keyBudgetBits={Number(metrics.secure_key_bits ?? 0)}
        // From the RUN's own metrics, not from the params. The panel must render what ran, and a
        // visitor can edit the params after a run without re-running.
        secretKeyRatePerPulse={Number(metrics.skr_per_pulse ?? 0)}
        sourceRateHz={Number(metrics.source_rate_hz ?? 0)}
        pulses={Number(metrics.qkd_pulses ?? 0)}
        qkdEnabled={run.qkd !== "pqc-only"} />

      <div className="note" style={{ marginTop: 16, fontSize: 12 }}>
        State: <b>{state}</b>. The diagram above is a live two-party simulator — each protocol step lights up as the
        pipeline runs, and the benchmark below fills in from that same run (one Run, one result). Demo mode computes
        entirely in the browser (no server compute); the full deployment adds the server-side backends and the real
        MCP server. PoC runs LAN/local; the diagram illustrates the WAN two-party deployment.
        <div className="muted" style={{ marginTop: 6, fontSize: 11 }}>
          Hybrid split grounded in <i>EncFormer</i> (linear→FHE, non-linear→MPC) · system-level cost per
          <i> "Beyond Latency: MPC &amp; FHE for PPML"</i> (arXiv 2604.00169, 2026) · two-party QOT per
          <i> Zhang et al.</i> (npj Quantum Information 2026). See <code>references/</code>.
        </div>
      </div>

      {/* Run status, visible from anywhere on the page.
          This page is about 7.3 viewport-heights tall and its whole point is watching a live run,
          but the status badge lives in the controls card at the top and scrolls out of view almost
          immediately. A visitor reading the benchmark or the posture table - which is where the
          interesting output is - had no way to tell whether the run was still going, had finished,
          or had stopped. Only the sidebar was sticky.

          Rendered only while it carries information: during a run, and while paused. It disappears
          on idle and once a completed run has been read, so it adds no permanent chrome to a page
          that is already long. */}
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

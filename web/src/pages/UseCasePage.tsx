import { EncryptedAgreementChart, MeasuredAgainstPublished, NoEncryptedComparison, VerificationChart } from "../components/VerificationChart";
import type { EncryptedAgreementPlot } from "../components/VerificationChart";
import { useRef } from "react";
import { Link, useParams } from "react-router";
import AgenticAIPanel from "../components/AgenticAIPanel";
import SdnControlPlanePanel from "../components/SdnControlPlanePanel";
import BenchmarkPanel from "../components/BenchmarkPanel";
import { ModelPanel } from "../components/ModelPanel";
import { ProvenanceBadge, type Provenance } from "../components/ProvenanceBadge";
import PipelineControls from "../components/PipelineControls";
import RunMonitor from "../components/RunMonitor";
import ShapPanel from "../components/ShapPanel";
import UseCaseCanvas from "../components/UseCaseCanvas";
import { USECASE_SPECS } from "../lib/useCaseFlows";
import { Qcnn3D, Qotp3D, Sat3D } from "../components/Lazy3D";
import { FibreDistanceCharts, QsdcRangeChart } from "../components/QkdCharts";
import SatNetworkPanel from "../components/SatNetworkPanel";
import Tabs from "../components/Tabs";
import { satChannel } from "../lib/satLink";
import Details from "../components/Details";
import NotFound from "../components/NotFound";
import KeyPoints from "../components/KeyPoints";
import StageBadge from "../components/StageBadge";
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
  // A page needs a canvas spec AND a page entry: the generic demo has the first since Round 81 (the E2E
  // page draws it) and no page of its own, so `/usecase/demo` stays unknown rather than half-drawn.
  if (!id || !(id in USECASE_SPECS) || !(id in USECASE_META)) return <UnknownUseCase id={id} />;
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
      </div>
      {DEMO && <div className="note" style={{ marginBottom: 12 }}>🖥️ {DEMO_BANNER}</div>}

      {/* SECTION HEADINGS, for a screen reader's heading list (SC 1.3.1). Every card title on this
          page is an h3, in this file and in the components it renders, so the outline went from
          the page title straight to h3. Converting the card titles would change visual weight
          everywhere; these four h2s group them instead, hidden because the cards already read as
          sections to the eye. Round 74. */}
      <h2 className="sr-only">Architecture simulator</h2>
      {/* Live architecture simulator: Start/Stop/Restart/Pause/Step drive the canvas animation */}
      <PipelineControls run={run} />
      {/* `overflowX: "auto"` because this div hand-rolls the LOOK of `.card` and not its behaviour.
          `.card` carries `overflow-x: auto`, and its comment records the mobile audit that put it
          there - at 390px the document was 1326px wide, 936px of horizontal page scroll, "from two
          diagram SVGs (990px, 930px)" among others. The two-party diagram then here set `minWidth: 680`
          DELIBERATELY, so it scrolls inside its container rather than shrinking to ~4.5px labels,
          and its comment says that is "what `.card { overflow-x: auto }` already provides". It was
          right about the rule and wrong about this element: copying the panel background, radius and
          padding by hand reproduced the appearance and dropped the one property that mattered.
          Measured on the deploy at a 500px viewport before this: 204px of document overflow, and
          exactly ONE genuine cause once nav-internal scrollers are discounted - this SVG. */}
      <div ref={diagramRef} style={{ marginTop: 16, background: "var(--panel)", borderRadius: 10,
                                     padding: 8, overflowX: "auto" }}>
        {/* The run's own config object (usePipelineRun), the one the E2E page's canvas reads too. */}
        <UseCaseCanvas usecase={usecase} activeStage={active} doneStages={done} metrics={metrics}
          config={run.diagramConfig} />
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
        <button className="btn ghost" onClick={() => downloadPng(`${usecase}-architecture.png`)}>⬇ Diagram (PNG)</button>
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

      {/* The simulated channel across its whole range, beside the single run above. Computed from the
          same model functions the run calls, at this page's own parameter values. Round 58. */}
      {(usecase === "telecom_ids" || usecase === "satellite_comms") && (
        <div className="card">
          <h3>Simulated quantum channel across its range <StageBadge kind="simulated" /></h3>
          {usecase === "satellite_comms"
            ? (() => {
                // The run's own link - profile, minimum elevation and the rest - for every view below.
                const sat = satChannel(run.params as Record<string, any>);
                // Two tabs since Round 77: one downlink across its range, and what carries a key past
                // the horizon that downlink cannot see over.
                return (
                  <Tabs label="Satellite channel views" tabs={[
                    { id: "downlink", title: "Downlink", content: () => (
                      <>
                        <QsdcRangeChart channel={sat} />
                        <Details summary="Downlink geometry in 3D">
                          <Sat3D slantKm={sat.distanceKm} orbitKm={Number(sat.freeSpace!.orbitAltitudeKm)}
                            minElevationDeg={Number(sat.freeSpace!.minElevationDeg)}
                            stationKm={Number(sat.freeSpace!.stationAltitudeKm ?? 0)} />
                        </Details>
                      </>
                    ) },
                    { id: "network", title: "Network", content: () => (
                      <SatNetworkPanel params={run.params as Record<string, any>} />
                    ) },
                  ]} />
                );
              })()
            : <FibreDistanceCharts misalignment={Number(run.params.qkd_misalignment ?? FIBER.misalignment)}
                eve={Number(run.params.qkd_eve ?? 0)} />}
        </div>
      )}

      <h2 className="sr-only">Models and verification</h2>
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
            Re-scored in this tab on the model's own published evaluation sample
            ({artifacts.verification.split ?? "split not declared"}). Charts are drawn from the same
            arrays the metric was computed on.
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
                caveat={spec.metric?.comparison_caveat || null}
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
          usecase={usecase}
        />
      ) : null}

      <h2 className="sr-only">Benchmark</h2>
      {/* Unified benchmark: auto-filled from the single Run above (no separate "Run benchmark" button).
          The MPC-FHE row is the live run's measured FHE-linear/MPC-non-linear split. */}
      <BenchmarkPanel compact={!meta.inference} result={benchmark}
        clientComputeMs={benchmark?.client_compute_ms ?? null} />

      {/* What the QSMPC-FHE row's circuit does, in 3D: the per-qubit reduced Bloch vectors of the real
          state-vector simulation (content), beside the one-time pad on a single qubit (illustration).
          Loaded on demand. Round 58. */}
      {meta.inference && (
        <div className="card" style={{ marginTop: 14 }}>
          <h3>Inside the QSMPC-FHE comparison <StageBadge kind="seeded" /></h3>
          <Qcnn3D features={[0.3, -0.5, 0.8, 0.1, -0.9, 0.6, -0.2, 0.4]} seed={1} />
          <Details summary="The quantum one-time pad on one qubit, in 3D (an illustration)">
            <Qotp3D x={0.3} />
          </Details>
        </div>
      )}

      <h2 className="sr-only">Agentic AI and the QKD control plane</h2>
      {/* Agentic AI · quantum-safe MCP — per-industry panel, fed by THIS run's live metrics (Round 10).
          The report frames MCP as a per-use-case cross-cutting layer; the same panel is kept standalone.
          It is an application-plane SAE: see the control-plane panel directly below, which is the
          element that actually manages key associations (ETSI GS QKD 015 puts application interfaces
          out of scope of the control interface). */}
      <AgenticAIPanel inline usecase={usecase} defaultPrompt={meta.agenticPrompt}
        mcpToolName={meta.agenticTool} runContext={{ metrics, benchmark }} />

      {usecase === "medical_fl" && (
        <div className="note" style={{ marginTop: 12, fontSize: 12 }}>
          {/* 167 words became four points; the history of the removed selector moved into a fold. */}
          <KeyPoints items={[
            "Federated learning runs on every run here - a real client-side FedAvg over the page's own bundled Wisconsin sample, with no toggle.",
            <><b>The encrypted path evaluates the published student itself</b>: all 3,226 parameters, from its pinned Hugging Face revision. CKKS evaluates the convolution and both dense layers; the client decrypts after each layer, applies ReLU and average pooling in plaintext, and takes the CNN&apos;s argmax in the clear. The garbled-circuit argmax runs on the bundled Wisconsin logistic regression, not on the CNN.</>,
            "The run reports the total rotations it performed, because key-switching - one per rotation - is what dominates CKKS inference cost.",
            "Nothing is multiplied by a cost model; if the weights cannot be fetched, the run says so rather than substituting invented ones.",
          ]} />
          <Details summary="Why there is no Workload selector">
            It offered Linear against CNN, and once CNN began running the real student, Linear was the
            only fabricated option left - an 8-feature dot product with a hand-written weight pattern
            matching no trained model. The im2col patches are packed into a single ciphertext and
            reduced by rotate-and-add.
          </Details>
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
        // THE BLOCK'S PULSES on the satellite (Round 79): the key budget above is the finite key of
        // a block of link time, a pass by default, so the channel time the panel divides consumed
        // bits by must be that block's. The sampled run's own 2e7 pulses would put a pass's budget
        // over a fifth of a second and drive eskr to zero.
        pulses={Number(metrics.qkd_block_pulses ?? metrics.qkd_pulses ?? 0)}
        decoyStates={metrics.qkd_decoy_states as string | undefined}
        qkdEnabled={run.qkd !== "pqc-only"} />

      <div className="note" style={{ marginTop: 16, fontSize: 12 }}>
        {/* "the benchmark below fills in" sat BELOW the benchmark (fixed in Round 58). */}
        State: <b>{state}</b>. The diagram at the top is a live two-party simulator — each protocol step lights up as
        the pipeline runs, and every panel on this page fills in from that same run (one Run, one result). Demo mode
        computes entirely in the browser (no server compute); the full deployment adds the server-side backends and
        the real MCP server. PoC runs LAN/local; the diagram illustrates the WAN two-party deployment.
        <div className="muted" style={{ marginTop: 6, fontSize: 11 }}>
          Hybrid split, on the pages that run one (finance, medical, genomics), grounded in{" "}
          <i>EncFormer</i> (linear layers to FHE, non-linear to MPC; in this tab only the argmax runs as
          MPC) - system-level cost per
          <i> "Beyond Latency: MPC &amp; FHE for PPML"</i> (arXiv 2604.00169, 2026) · two-party QOT per
          <i> Zhang et al.</i> (npj Quantum Information 2026). See <code>references/</code>.
        </div>
      </div>

      {/* The run's status is shown in the sidebar (lib/runStatus.ts) since Round 80. */}
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
    <NotFound title="No such use case"
      lead={<>{id ? <>This app has no use case with the id <code>{id}</code>.</> : <>No use case id was given.</>}
        {" "}These are the pages and use cases it does have:</>} />
  );
}

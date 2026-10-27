// Shared pipeline run-engine hook, used by both the generic E2E Orchestration page and
// the per-use-case pages. Encapsulates: crypto-agility options, advanced params, the
// Start/Stepwise/Pause/Resume/Step/Stop/Restart controls, the demo (client-side) vs backend
// (WS) transport branch, the live event stream, metrics, signature, and GIF/PNG frame capture.
import { useEffect, useRef, useState, type RefObject } from "react";
import { api, type PipelineEvent, streamRun } from "./api";
import { benchCfg, dataCfg, idsCfg, qkdCfg, qrngCfg, webCfg } from "./config";
import { FREE_SPACE } from "./quantumSim";
import { assembleBenchmark, clientFlow, type PipelineControl, runClientPipeline } from "./clientCompute";
import { isClientDemo } from "./demoMode";

import { downloadCSV, downloadJSON, downloadNodePNG, FrameRecorder } from "./export";

/**
 * Whether a use case runs the stage that reads a given control.
 *
 * Derived from the FLOW, not from a hand-kept list of use-case ids, because the two disagreed. The
 * lists said QSMPC and inference belong to finance, medical and genomics; `FLOWS.demo` contains
 * `qsmpc_compute` AND `private_inference` and was in neither list. So the Generic demo hid three
 * controls that its own pipeline then read - and since `params` survives a use-case change on the
 * Orchestration page, whatever the previous use case had set is what the demo silently ran with.
 *
 * Asking the flow makes the panel and the run agree by construction: a control is shown exactly when
 * a stage exists to read it, and `buildConfig` ships it exactly when it is shown.
 */
export const runsQsmpc = (usecase: string) => clientFlow(usecase).includes("qsmpc_compute");
export const runsInference = (usecase: string) => clientFlow(usecase).includes("private_inference");
export const runsFibreQkd = (usecase: string) => clientFlow(usecase).includes("qkd_key_agreement");

/** Stages whose implementation reads `config.inference` to choose a backend.
 *
 * `runsInference` above answers a different question - whether the use case runs the stage NAMED
 * `private_inference` - and using it to gate the Inference control conflated the two. Telecom reads
 * `config.inference` from `qml_ids`, so the control it depends on was never rendered: the page
 * offered no way to pick a backend at all, while the QCNN branch behind it sat ready. That is the
 * same inert-control failure as before with the halves swapped, and it survived the fix for the
 * first half because the reachability test asks about stages and artefacts, not about whether a
 * control exists to reach them.
 *
 * `tests/test_qcnn_reachable.py` now checks this list against the stage bodies that actually read
 * the key, so a stage gaining or losing the backend switch cannot leave this stale.
 */
export const INFERENCE_BACKEND_STAGES = ["private_inference", "qml_ids"];
export const readsInferenceBackend = (usecase: string) =>
  clientFlow(usecase).some((s) => INFERENCE_BACKEND_STAGES.includes(s));

/** Use cases that ship trained QCNN weights under `public/data/<dir>/qcnn_weights.json`.
 *
 * The same reasoning as the three predicates above, one level down. A backend option is honest only
 * when BOTH halves hold:
 *
 *   1. the artefact implementing it exists, and
 *   2. a stage the use case actually RUNS reads that artefact.
 *
 * Genomics failed the first half: it reaches `private_inference`, so `runsInference` is true and the
 * selector belongs on the page, but it ships no weights - its QCNN option fell back to CKKS while two
 * panels announced that a quantum circuit had decided the class.
 *
 * Telecom then failed the SECOND half, because this comment only stated the first. Adding
 * `telecom_ids` here and committing `qcnn_weights.json` satisfied condition 1, and the option duly
 * appeared - but `FLOWS.telecom_ids` runs `qml_ids`, not `private_inference`, and the QCNN branch
 * lived in the `private_inference` case. Same visible failure as genomics, opposite cause. tsc was
 * clean, every gate was green, and the weights were served with the right content-type; what surfaced
 * it was the deployed page never fetching them at all.
 *
 * `tests/test_qcnn_usecases.py` asserts this set against the files on disk (half 1) and
 * `tests/test_qcnn_reachable.py` asserts it against the stages in `FLOWS` (half 2), so training a
 * third circuit cannot leave the list stale, deleting one cannot leave it lying, and registering one
 * whose stage cannot reach it fails in CI rather than on the site.
 */
export const QCNN_USECASES = new Set(["finance_aml", "medical_fl", "telecom_ids"]);
export const hasTrainedQcnn = (usecase: string) => QCNN_USECASES.has(usecase);

/** The widths a trained circuit actually SHIPS for, per use case.
 *
 * The "QCNN qubits" control offered 2..14 on every page while exactly two artifacts existed for
 * telecom and one each for finance and medical. Every other width fell back to the 8-wire file in
 * silence: the control moved, the run reported `qcnn_qubits` as the width that actually ran, and the
 * two disagreed with no way for a visitor to tell. Round 40 made the control REACH the circuit;
 * it did not make the range honest.
 *
 * WHY THE SETS DIFFER, and it is a property of the data rather than a choice. A wire must carry a
 * feature or be padded, so the floor is the feature count: telecom has 7 features, finance and
 * medical have 8, and `qcnn_variational.py` refuses a narrower run outright ("8 features exceeds 7
 * wires"). Telecom can therefore be narrowed and the others can only be widened.
 *
 * Kept beside `QCNN_USECASES` and gated the same way - `tests/test_qcnn_usecases.py` compares this
 * against `web/public/data/<dir>/qcnn_weights*.json` in BOTH directions, so adding a width without
 * declaring it, or declaring one without shipping it, fails in CI rather than on the site.
 */
export const QCNN_WIDTHS: Record<string, number[]> = {
  finance_aml: [8, 9],
  medical_fl: [8, 9],
  telecom_ids: [7, 8],
};
export const qcnnWidths = (usecase: string) => QCNN_WIDTHS[usecase] ?? [];

/** Use cases where a federated-training stage has both real data and somewhere to be rendered.
 *
 * The same rule as the predicates above, applied to an injected stage. `qfl` is `useState(true)` with
 * no writer, so the stage was injected into finance, genomics, telecom, satellite and the generic
 * demo as well - and in three of those it had neither half:
 *
 *   - No render target. `OrchestrationGraph`'s default flow has no `federated_training` node and
 *     `TwoPartyDiagram`'s federated step is medical-only, so the animation advanced to a stage no
 *     figure contained and every diagram row dimmed with nothing highlighted.
 *   - No real data. `realFederatedSet` maps only medical and finance to a bundled sample, so genomics,
 *     telecom and satellite trained on a synthetic two-cluster toy - whose accuracy was then handed to
 *     the agent panel as "federated accuracy" with no qualifier, on pages that never mention
 *     federation. The demo banner meanwhile still tells visitors federated training is the medical
 *     page's feature.
 *
 * Finance qualifies on both counts: `realFederatedSet` has its sample, and its diagram renders the
 * step. `tests/test_federated_scope.py` checks this set against the data map AND against both
 * figures, because for one round it checked only the data map. Finance was admitted on the sample
 * alone while `USECASE_SPECS.finance_aml` had no federated step and `OrchestrationGraph.FLOWS` had
 * no finance key - so the render half of the rule this comment states was false about the very use
 * case the comment cites, and CI was green. Measured on the deployed site: the stage started,
 * reported `federated_accuracy`, and no svg on the page contained the string "federat".
 */
export const FEDERATED_USECASES = new Set(["medical_fl", "finance_aml"]);
export const runsFederated = (usecase: string) => FEDERATED_USECASES.has(usecase);

export const MODEL_SIZES = [
  { id: "lightweight", label: "Lightweight" },
  { id: "medium", label: "Medium" },
  { id: "large", label: "Large (projected)" },
];

const DEMO = isClientDemo();

// crypto-agility options for the pure static demo (mirror of /api/crypto-options)
export const DEMO_CRYPTO = {
  families: {
    kem: ["ml-kem-768+x25519", "ml-kem-1024+x25519"],
    qkd: ["bb84-decoy", "pqc-only"],
    fhe: ["tenseal-cpu", "openfhe-cpu"],
    mpc: ["mpyc-shamir", "mp-spdz", "hpmpc-gpu"],
    sig: ["ml-dsa-65", "ml-dsa-87"],
  },
  defaults: { kem: "ml-kem-768+x25519", qkd: "bb84-decoy", fhe: "tenseal-cpu",
    mpc: "mpyc-shamir", sig: "ml-dsa-65" },
};

// Editable run parameters — defaults sourced from config (config.defaults.json / /api/config), so the
// WebUI Parameters panel starts from the backend defaults and any edit flows into the run config (no
// hardcoded literals here). Every field is overridable per run from the panel.
export const makeDefaultParams = () => ({
  // 1, not 0. Round 21 raised this field's `min` to 1 because quantumSim's rng() does
  // `(seed >>> 0) || 1` and maps both to the same stream - and left the DEFAULT at 0, so every
  // page shipped a field whose value sat below its own minimum, and "Reset to defaults"
  // restored an out-of-range value. Caught by capturing every input's value against its own
  // min/max rather than by reading either one alone.
  seed: 1,
  qkd_pulses: qkdCfg().pulses,
  qkd_distance_km: qkdCfg().distance_km,
  qkd_eve: qkdCfg().eve_intercept_fraction,
  qkd_misalignment: qkdCfg().misalignment,
  qkd_opt_samples: qkdCfg().optimize_n_samples,
  qrng_bits: qrngCfg().bits,
  n_per_class: idsCfg().n_per_class,
  aml_transactions: dataCfg().aml_transactions,
  qubits: benchCfg().quantum_qubits,
  qrng_eps_log2: qrngCfg().security_epsilon_log2,   // LHL extractor security 2^this (toeplitz/seedless)
  qsmpc_op: "",          // "" = use the use case's default
  prime: 0,              // 0 = use config default
  inference: "",         // "" = use the use case's default
  qrng_post: qrngCfg().post_processing,             // "von_neumann" | "toeplitz" | "seedless"
  qkd_protocol: qkdCfg().protocol,                  // "bb84" (decoy) | "e91" (CHSH) — OptiQKD model
  // The satellite channel. `satellite_qkd` and `qsdc_telemetry` have read these four keys since the
  // page was written, and no default or control ever supplied them -- so Eve sat at 0, free-space
  // QBER never moved off ~0.033, and the inline eavesdrop check the page advertises could not fire
  // from any user action. Defaults come from the free-space channel model and config, not literals.
  sat_distance_km: FREE_SPACE.distanceKm,
  sat_eve: FREE_SPACE.eveFraction,
  sat_pulses: webCfg().demo_sat_pulses,
  // The two fixed terms of the link budget, and the reason the eavesdrop check could never fire.
  // `channelLossDb` caps the diffraction term at the near field, so these floor total loss at 8 dB
  // no matter how short the range - with the detector's 4.0 dB, g >= 15.8 - and at the default
  // budget C_s is negative at every range (-3.70 at 0.1 km with no eavesdropper). Every abort was
  // therefore attributable to the budget rather than to Eve. (Corrected 2026-10-24: this comment
  // quoted g >= 6.31 and the pooled tolerance 1/(1+g), both from before detector efficiency entered
  // g and the per-basis entropy replaced the pooled one.) Exposing them lets a
  // visitor select a LINK (the Qi et al. 2019 demonstration was 4.1 dB on fibre) rather than edit
  // physics; the defaults are Liao et al. 2017's Micius downlink.
  sat_atmospheric_db: FREE_SPACE.freeSpace!.atmosphericDb,
  sat_pointing_db: FREE_SPACE.freeSpace!.pointingDb,
  // The two assumed source clocks, in MHz, one per channel. From config rather than a literal, so
  // the control opens on exactly the value `clientCompute.ts` would have used had nobody touched it
  // - a control whose default disagrees with the fallback behind it is a silent state change on
  // first render. Two keys because the two defaults have different provenance: the free-space
  // figure is the anchor experiment's own configuration, the fibre one is that clock carried over
  // to a channel it was not measured on.
  qkd_source_rate_mhz: qkdCfg().source_rate_hz / 1e6,
  sat_source_rate_mhz: qkdCfg().free_space_source_rate_hz / 1e6,
});

export const DEFAULT_PARAMS = makeDefaultParams();

export type PipelineParams = ReturnType<typeof makeDefaultParams>;

// `recordRef` is the DOM node captured for the animated-GIF export (the DAG on the
// Orchestration page, the two-party diagram on a use-case page).
export function usePipelineRun(usecase: string, recordRef: RefObject<HTMLElement | null>) {
  const [opts, setOpts] = useState<{ families: Record<string, string[]>; defaults: Record<string, string> }>(
    { families: { kem: [], qkd: [], fhe: [], mpc: [], sig: [] }, defaults: {} },
  );
  const [kem, setKem] = useState("");
  const [qkd, setQkd] = useState("");
  const [fhe, setFhe] = useState("");
  const [mpc, setMpc] = useState("");
  const [sig, setSig] = useState("");
  const [workload, setWorkload] = useState("linear");
  // On by default where it is real. It runs a client-side FedAvg over the page's own bundled sample,
  // which is worth showing without making a visitor find a checkbox - but only two use cases HAVE such
  // a sample, so `buildConfig` gates the flag on `runsFederated`. The state is kept so the flag can
  // still be turned off programmatically for a test or a comparison.
  const [qfl, setQfl] = useState(true);
  const [optiqkd, setOptiqkd] = useState(false);         // "Optimize QKD (OptiQKD)" toggle (WS1 AI-QKD)
  // Which published model the orchestration's inference stage runs. "" = the registry's first
  // fetchable one for this use case, which is what every run used before this existed. Three use
  // cases publish more than one and could only ever exercise the first: medical has ResNet-18,
  // ResNet-50 and the CKKS student; satellite has four across two incomparable tasks.
  const [publishedModel, setPublishedModel] = useState("");
  // The backend the published ONNX models run on. "" = the config default (WASM); "webgpu" only when
  // the visitor selects it. A visitor's choice about their machine rather than the use case, so it
  // is NOT reset when the use case changes.
  const [onnxBackend, setOnnxBackend] = useState<"" | "webgpu">("");
  const [params, setParams] = useState({ ...DEFAULT_PARAMS });
  const resetParams = () => setParams({ ...makeDefaultParams() });

  const [runId, setRunId] = useState<string | null>(null);
  const [state, setState] = useState("idle");
  const [events, setEvents] = useState<PipelineEvent[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [done, setDone] = useState<Set<string>>(new Set());
  const [metrics, setMetrics] = useState<Record<string, any>>({});
  const [signature, setSignature] = useState<any>(null);
  const [artifacts, setArtifacts] = useState<Record<string, any>>({});
  const [benchmark, setBenchmark] = useState<any>(null);   // unified: assembled from the live run
  const [gifBusy, setGifBusy] = useState(false);

  const stopRef = useRef<(() => void) | null>(null);
  const controlRef = useRef<PipelineControl | null>(null);
  // Which run owns the UI. Incremented by every `launch`, captured by that launch's handlers.
  //
  // WHY THIS EXISTS. `stop()` only sets a flag the pipeline reads BETWEEN stages, so a run stopped
  // mid-stage keeps executing to the end of that stage and then calls its `onEnd` - which used to
  // write its metrics, artifacts, signature and benchmark into state that a NEWER run already owned.
  // Pressing Restart during `private_inference` (seconds long: CKKS convolution, a garbled-circuit
  // argmax, three or four ONNX evaluations) therefore replaced the new run's results with the
  // previous run's, and flipped the badge to "stopped" while the new run was visibly still going.
  // The caption on the use-case page invites exactly that click. Comparing tokens makes a superseded
  // run inert rather than merely flagged.
  const runTokenRef = useRef(0);
  const logRef = useRef<HTMLDivElement>(null);
  const recorderRef = useRef<FrameRecorder | null>(null);
  const stageStartRef = useRef<Record<string, number>>({});
  const stageMsRef = useRef<Record<string, number>>({});

  useEffect(() => {
    const apply = (o: { families: Record<string, string[]>; defaults: Record<string, string> }) => {
      setOpts(o);
      setKem(o.defaults.kem); setQkd(o.defaults.qkd); setFhe(o.defaults.fhe);
      setMpc(o.defaults.mpc); setSig(o.defaults.sig);
    };
    if (DEMO) { apply(DEMO_CRYPTO); return; } // no backend in the public demo
    api.cryptoOptions().then(apply).catch(() => apply(DEMO_CRYPTO));
  }, []);

  useEffect(() => { logRef.current?.scrollTo(0, logRef.current.scrollHeight); }, [events]);

  // Clear the model choice when the use case changes.
  //
  // `/usecase/:id` matches the same route for every use case, so React Router does not remount this
  // hook - and the Orchestration page changes `usecase` without navigating at all. The chosen key
  // therefore survived into a use case that does not publish it, `browserModels(next).find(...)`
  // returned nothing, and the run reported "no published model registered for this use case" on a
  // page that has one. Observed: select the finance GBM, move to telecom, and telecom_ids.qml -
  // published, pinned and perfectly runnable - vanished.
  //
  // This is the same leak the Parameters controls had, one selector later. Anything keyed to the use
  // case has to be reset when the use case changes, or gated at the point of use.
  //
  // ROUND 43: that sentence states a rule about a SET and the code below it implemented one member.
  // `publishedModel` was cleared; the nine fields a run writes were not, because `reset()` is only
  // called from `launch()`. Measured on the deployed site: run `finance_aml` on `/orchestration`,
  // switch the selector to `demo`, and the QHE/FHE node still reads "(finance_aml.gbm on wasm)".
  // The caption was only the visible corner of it - `Orchestration.tsx` also offered the previous
  // run's events for export under the NEW use case's filename, and drew the old encrypted-agreement
  // chart under the new selection.
  //
  // It calls `reset()`, which is declared further down: legal because the effect body runs after
  // render, not during it, and `tsc` confirms. It cancels the in-flight run FIRST, because
  // `/usecase/:id` navigation is not blocked by `inFlight` the way the Orchestration selector is, so
  // a stage still running could otherwise write into the use case the visitor just switched to.
  useEffect(() => {
    stopRef.current?.();          // stop the previous run before its next stage lands
    runTokenRef.current += 1;     // and drop anything already in flight from it
    setPublishedModel("");
    reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [usecase]);

  const buildConfig = () => {
    // spread every numeric/editable param into the run config; the backend + demo read the keys they
    // know (ctx.config.get(key, <config default>)), so a Parameters-panel edit overrides the default.
    const { qsmpc_op, inference, prime, ...nums } = params;
    // `usecase` is READ by the run - `clientCompute`'s security facts pick the free-space channel
    // model with `config.usecase === "satellite_comms"` - and was never emitted here, so the key was
    // permanently `undefined` and the ternary permanently took the fibre branch. The satellite page
    // therefore printed a decoy-state tagged fraction of 7.26 computed over 50 km of FIBRE, using two
    // parameters that page does not even render, where the free-space downlink gives roughly 125.
    // A number labelled `measured` that describes a different channel is the defect class this
    // project treats as most serious, and it was caused by a missing key rather than by wrong maths.
    const c: Record<string, unknown> = { usecase, kem, qkd, fhe, mpc, sig, workload, ...nums };
    // Gated on the SAME predicates that decide whether the control is rendered, because `params`
    // survives a use-case change and the Orchestration page changes it without remounting. Set QSMPC
    // op = sum, Inference = qcnn and prime = 97 on Finance, then switch to Generic demo: the three
    // selects disappear and, before this gate, all three still reached the run. The demo flow reads
    // every one of them, so it performed a secure summation over Z97 with a QCNN readout, driven by
    // controls that were no longer on screen.
    if (runsQsmpc(usecase) && qsmpc_op) c.qsmpc_op = qsmpc_op;
    // `readsInferenceBackend`, not `runsInference`: the key is read by `qml_ids` too, and gating the
    // send on the stage NAME meant telecom's selection was dropped on the way to the run even once
    // the control was on screen and the branch existed. Three gates in series, each invisible alone.
    if (readsInferenceBackend(usecase) && inference) c.inference = inference;
    if (runsQsmpc(usecase) && prime > 0) c.prime = prime;
    // Only when a choice was made. Absent, runPublishedModel falls to browserModel(usecase), so the
    // key never appears in a signed payload claiming a selection the visitor did not make.
    if (publishedModel) c.published_model = publishedModel;
    // Same rule: sent only when chosen, so a run's payload never claims a selection nobody made.
    if (onnxBackend === "webgpu") c.onnx_backend = "webgpu";
    // Gated on the same predicate style as the controls above: the flag reaches the run only where a
    // federated stage has real data and a figure to appear in.
    if (qfl && runsFederated(usecase)) c.qfl = true;
    if (optiqkd) c.qkd_optimize = true;                  // run the AI-QKD parameter optimization this run
    return c;
  };

  /** Clear everything a run writes. ALL NINE fields, not the seven this used to clear.
   *
   *  `state` and `runId` were missing. From `launch()` that was invisible, because the next two
   *  statements set both - but the `[usecase]` effect above has no such follow-up, so resetting
   *  seven of nine would have left the badge reading `done` over an empty event log and a stale
   *  `runId` addressing a run belonging to another use case. Seven of nine is the same defect one
   *  layer in, which is why the set is derived from the writers rather than listed by hand:
   *  `handleEvent` writes active/done/events/metrics/state, the two `onEnd` blocks add
   *  artifacts/signature/benchmark, and `launch` adds runId.
   */
  const reset = () => {
    setEvents([]); setActive(null); setDone(new Set()); setMetrics({}); setSignature(null);
    setArtifacts({}); setBenchmark(null); setState("idle"); setRunId(null);
    stageStartRef.current = {}; stageMsRef.current = {};
    recorderRef.current = recordRef.current ? new FrameRecorder(recordRef.current) : null;
  };

  // assemble the unified benchmark from the finished run (prefer the server-assembled
  // `benchmark_result`; fall back to a client-side assembly for the pure demo / older runs)
  const finishBenchmark = (st: { metrics: Record<string, any>; artifacts: Record<string, any> }) => {
    const server = (st.artifacts as any)?.benchmark_result;
    setBenchmark(server ?? assembleBenchmark(usecase, buildConfig(),
      { metrics: st.metrics, artifacts: st.artifacts, stageMs: stageMsRef.current }));
  };
  const captureFrame = () => { recorderRef.current?.capture().catch(() => {}); };

  // shared event handler for both transports (backend WS + client-side pipeline)
  const handleEvent = (e: PipelineEvent) => {
    setEvents((prev) => [...prev, e]);
    if (e.type === "stage_start") { setActive(e.stage); stageStartRef.current[e.stage] = e.t; }
    if (e.type === "stage_end") {
      setActive(null); setDone((p) => new Set(p).add(e.stage)); captureFrame();
      const s = stageStartRef.current[e.stage];          // recover per-stage duration (both transports)
      if (s != null) stageMsRef.current[e.stage] = Math.round((e.t - s) * 1000);
    }
    // "pausing"/"stopping" are honest intermediate states: the button has been pressed, the flag is
    // set, and the stage in flight runs to its end before the pipeline reads it. Only
    // `pipeline_pause_effective` means the run is actually sitting at the gate.
    if (e.type === "pipeline_pause") setState("pausing");
    if (e.type === "pipeline_pause_effective") setState("paused");
    if (e.type === "pipeline_stop_requested") setState("stopping");
    // The EFFECT. The client pipeline announces its halt through `onEnd`, which the backend
    // transport has no equivalent of - so over the API a stopped run told nobody it had stopped and
    // the badge stayed on "stopping". Reading it costs nothing on the client path, where `onEnd`
    // arrives immediately after and sets the same state.
    if (e.type === "pipeline_stop") setState("stopped");
    if (e.type === "pipeline_resume") setState("running");
    if (e.type === "metric") setMetrics((m) => ({ ...m, [String(e.data.name)]: e.data.value }));
  };

  const launch = async (startPaused: boolean) => {
    stopRef.current?.();
    reset();
    // Claim the UI for this run. Anything arriving from an earlier token is dropped, so the previous
    // run finishing its in-flight stage cannot write over these results.
    const token = ++runTokenRef.current;
    const mine = () => token === runTokenRef.current;
    const onEventOwned = (e: PipelineEvent) => { if (mine()) handleEvent(e); };
    setState(startPaused ? "paused" : "running");
    captureFrame();   // initial frame

    if (DEMO) {
      setRunId("client");
      controlRef.current = runClientPipeline(usecase, buildConfig(), {
        onEvent: onEventOwned,
        onEnd: (st) => {
          if (!mine()) return;
          setState(st.state); setMetrics(st.metrics); setArtifacts(st.artifacts ?? {});
          setSignature((st.artifacts as any)?.result_signature ?? null);
          finishBenchmark({ metrics: st.metrics, artifacts: st.artifacts ?? {} });
        },
      }, startPaused);
      stopRef.current = () => controlRef.current?.stop();
      return;
    }

    const { run_id } = await api.start(usecase, buildConfig(), startPaused);
    if (!mine()) return;                       // a newer run started while this awaited the server
    setRunId(run_id);
    stopRef.current = streamRun(run_id, onEventOwned, async () => {
      const st = await api.status(run_id).catch(() => null);
      if (st && mine()) {
        setState(st.state); setMetrics(st.metrics); setArtifacts((st.artifacts as any) ?? {});
        setSignature((st.artifacts as any)?.result_signature ?? null);
        finishBenchmark({ metrics: st.metrics, artifacts: (st.artifacts as any) ?? {} });
      }
    });
  };

  const start = () => launch(false);
  const startStepwise = () => launch(true);
  const pause = async () => { if (DEMO) { controlRef.current?.pause(); return; } if (runId) await api.pause(runId); };
  const resume = async () => { if (DEMO) { controlRef.current?.resume(); return; } if (runId) await api.resume(runId); };
  const step = async () => { if (DEMO) { controlRef.current?.step(); return; } if (runId) await api.step(runId); };
  const stop = async () => {
    // Do NOT set "stopped" here. Both pipelines emit `pipeline_stop_requested` (badge: stopping) and
    // reports "stopped" from `onEnd` once it has actually stopped, which is at the next stage
    // boundary. Setting it here is what made the badge say stopped while the run was still visibly
    // producing output.
    if (DEMO) { controlRef.current?.stop(); return; }
    if (runId) { setState("stopping"); await api.stop(runId); }
  };

  const downloadGif = async (filename: string) => {
    if (!recorderRef.current || recorderRef.current.count === 0) return;
    setGifBusy(true);
    try { await recorderRef.current.encodeGIF(filename); }
    finally { setGifBusy(false); }
  };
  const downloadPng = async (filename: string) => {
    if (recordRef.current) await downloadNodePNG(recordRef.current as HTMLElement, filename);
  };
  const downloadEvents = (filename: string) => downloadJSON(events, filename);
  const downloadMetricsCsv = (filename: string) =>
    downloadCSV(Object.entries(metrics).map(([metric, value]) => ({ metric, value })), filename);

  return {
    usecase, opts, kem, setKem, qkd, setQkd, fhe, setFhe, mpc, setMpc, sig, setSig,
    workload, setWorkload, qfl, setQfl, optiqkd, setOptiqkd,
    publishedModel, setPublishedModel, onnxBackend, setOnnxBackend,
    params, setParams, resetParams,
    runId, state, events, active, done, metrics, signature, artifacts, benchmark,
    gifBusy, logRef, captureFrame,
    start, startStepwise, pause, resume, step, stop,
    downloadGif, downloadPng, downloadEvents, downloadMetricsCsv,
    running: state === "running", paused: state === "paused",
    // `running` and `paused` are exact-match, so the transitional "pausing"/"stopping" states are
    // neither - which would have re-enabled Start and the published-model select mid-run. Button
    // enablement asks "is a run in flight", which is a different question from "is it producing
    // output right now", and conflating the two is what made the two new states unsafe to add.
    inFlight: !["idle", "done", "stopped", "error"].includes(state),
  };
}

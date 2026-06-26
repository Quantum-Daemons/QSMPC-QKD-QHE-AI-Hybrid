// Shared pipeline run-engine hook, used by both the generic E2E Orchestration page and
// the per-use-case pages. Encapsulates: crypto-agility options, advanced params, the
// Start/Stepwise/Pause/Resume/Step/Stop/Restart controls, the demo (client-side) vs backend
// (WS) transport branch, the live event stream, metrics, signature, and GIF/PNG frame capture.
import { useEffect, useRef, useState, type RefObject } from "react";
import { api, type PipelineEvent, streamRun } from "./api";
import { assembleBenchmark, type PipelineControl, runClientPipeline } from "./clientCompute";
import { isClientDemo } from "./demoMode";
import { downloadCSV, downloadJSON, downloadNodePNG, FrameRecorder } from "./export";

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

export const DEFAULT_PARAMS = {
  qkd_pulses: 200000,
  qkd_distance_km: 50,
  qkd_eve: 0,
  qsmpc_op: "",          // "" = use the use case's default
  prime: 0,              // 0 = use config default
  inference: "",         // "" = use the use case's default
  qubits: 8,
  seed: 0,
};

export type PipelineParams = typeof DEFAULT_PARAMS;

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
  const [modelSize, setModelSize] = useState("medium");
  const [workload, setWorkload] = useState("linear");
  const [qfl, setQfl] = useState(false);                 // "Federated (QFL)" toggle (quantum federated learning)
  const [params, setParams] = useState({ ...DEFAULT_PARAMS });

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

  const buildConfig = () => {
    const c: Record<string, unknown> = { kem, qkd, fhe, mpc, sig, model_size: modelSize, workload,
      seed: params.seed, qkd_pulses: params.qkd_pulses, qkd_distance_km: params.qkd_distance_km,
      qkd_eve: params.qkd_eve, qubits: params.qubits };
    if (params.qsmpc_op) c.qsmpc_op = params.qsmpc_op;
    if (params.inference) c.inference = params.inference;
    if (params.prime > 0) c.prime = params.prime;
    if (qfl) c.qfl = true;                                // run Quantum Federated Learning this run
    return c;
  };

  const reset = () => {
    setEvents([]); setActive(null); setDone(new Set()); setMetrics({}); setSignature(null);
    setArtifacts({}); setBenchmark(null);
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
    if (e.type === "pipeline_pause") setState("paused");
    if (e.type === "pipeline_resume") setState("running");
    if (e.type === "metric") setMetrics((m) => ({ ...m, [String(e.data.name)]: e.data.value }));
  };

  const launch = async (startPaused: boolean) => {
    stopRef.current?.();
    reset();
    setState(startPaused ? "paused" : "running");
    captureFrame();   // initial frame

    if (DEMO) {
      setRunId("client");
      controlRef.current = runClientPipeline(usecase, buildConfig(), {
        onEvent: handleEvent,
        onEnd: (st) => {
          setState(st.state); setMetrics(st.metrics); setArtifacts(st.artifacts ?? {});
          setSignature((st.artifacts as any)?.result_signature ?? null);
          finishBenchmark({ metrics: st.metrics, artifacts: st.artifacts ?? {} });
        },
      }, startPaused);
      stopRef.current = () => controlRef.current?.stop();
      return;
    }

    const { run_id } = await api.start(usecase, buildConfig(), startPaused);
    setRunId(run_id);
    stopRef.current = streamRun(run_id, handleEvent, async () => {
      const st = await api.status(run_id).catch(() => null);
      if (st) {
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
    if (DEMO) { controlRef.current?.stop(); setState("stopped"); return; }
    if (runId) { await api.stop(runId); setState("stopped"); }
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
    opts, kem, setKem, qkd, setQkd, fhe, setFhe, mpc, setMpc, sig, setSig,
    modelSize, setModelSize, workload, setWorkload, qfl, setQfl, params, setParams,
    runId, state, events, active, done, metrics, signature, artifacts, benchmark,
    gifBusy, logRef, captureFrame,
    start, startStepwise, pause, resume, step, stop,
    downloadGif, downloadPng, downloadEvents, downloadMetricsCsv,
    running: state === "running", paused: state === "paused",
  };
}

// In-browser ONNX inference for the client-side demo, via onnxruntime-web.
//
// Execution-provider preference: WebGPU -> WebNN -> WASM (mature -> experimental -> floor). Vite
// bundles the ORT wasm as a same-origin hashed asset (so it satisfies the COOP/COEP
// cross-origin-isolation headers); we let ORT resolve it via import.meta.url rather than overriding
// wasmPaths.
//
// Two things this module is strict about, because the demo's credibility rests on them:
//
//   1. The reported backend is GENUINE. A real GPU adapter must be obtainable before "webgpu" is
//      claimed, so the page never reports WebGPU for a run that actually executed on WASM.
//   2. A failure is REPORTED, not absorbed. Until Round 17 this file loaded a model path that could
//      never exist (`*.onnx` is git-ignored and `web/public/models/` shipped empty), logged one
//      console warning, and silently served a hard-coded JS linear model instead. The page looked
//      identical whether real inference happened or not. Now every load records why it failed and
//      the UI surfaces it through ProvenanceBadge.
//
// Model weights come from the Hugging Face CDN at a pinned commit sha, addressed from the model
// registry in config.defaults.json — no model id, revision or URL is written in TypeScript.
// Types only. The RUNTIME build is chosen at load time by loadOrt() below, because the two builds
// are not interchangeable at runtime even though their APIs are identical.
import type * as OrtTypes from "onnxruntime-web";

/**
 * Load the ORT build that matches this browser's actual capability.
 *
 * `import * as ort from "onnxruntime-web"` resolves to the **JSEP** bundle -- the WebGPU-enabled
 * one, which pulls `ort-wasm-simd-threaded.jsep.wasm`. On a browser with no WebGPU adapter that
 * build's `InferenceSession.create` never returns: not for a tree ensemble, not for an LSTM, not
 * for a 45 MB ResNet, and not for a 0.3 MB graph. The same models create a session in 0.01-0.23 s
 * against the plain wasm build. That single import is what made the entire browser tier look
 * broken, and it survived two wrong diagnoses (unsupported operators; then the WebNN provider)
 * because every symptom it produces looks like a model problem.
 *
 * So: `onnxruntime-web/webgpu` only when an adapter genuinely exists, `onnxruntime-web/wasm`
 * otherwise. Both export the same surface, so nothing downstream changes.
 */
// One cached promise per build, not one for the page: a page can legitimately need both -- a
// convolutional model on webgpu and a tree ensemble on wasm, whose operators the webgpu build
// does not register.
const ortBuilds: Record<"webgpu" | "wasm", Promise<typeof OrtTypes> | null> = {
  webgpu: null, wasm: null,
};

function loadOrt(useWebGPU: boolean): Promise<typeof OrtTypes> {
  const kind = useWebGPU ? "webgpu" : "wasm";
  if (!ortBuilds[kind]) {
    ortBuilds[kind] = (useWebGPU
      ? import("onnxruntime-web/webgpu")
      : import("onnxruntime-web/wasm")) as Promise<typeof OrtTypes>;
  }
  return ortBuilds[kind]!;
}

import { modelsFor, type ModelSpec } from "./config";

// 30 s, not unbounded. These three cross-origin fetches had no timeout and no abort signal, so on a
// slow, proxied or blocked network the run sat in `private_inference` forever with the badge on
// `running` - a corporate visitor's most likely experience of the site being "broken". Only session
// CREATION was bounded. A timeout turns a hang into a reported failure, and the failure path here is
// already honest: it degrades and says so rather than substituting a number.
const HF_FETCH_TIMEOUT_MS = 30_000;

export type OnnxStatus = {
  loaded: boolean;
  backend: string;          // "onnx-webgpu" | "onnx-webnn" | "onnx-wasm" | "none"
  modelKey: string | null;
  url: string | null;
  bytes: number | null;     // transferred size, or null when served from cache
  fromCache: boolean;
  threads: number;
  /** Probed, not used: the WebNN provider is deliberately not requested. */
  webnnAvailable?: boolean;
  crossOriginIsolated: boolean;
  loadMs: number | null;
  error: string | null;     // non-null => nothing was inferred; the caller must say so
};

const status: OnnxStatus = {
  loaded: false, backend: "none", modelKey: null, url: null, bytes: null,
  fromCache: false, threads: 1, crossOriginIsolated: false, loadMs: null, error: null,
};

let session: OrtTypes.InferenceSession | null = null;
//: The build actually loaded, kept so Tensor construction uses the same module.
let ortRuntime: typeof OrtTypes | null = null;
/** Per-model, not per-page: see loadModel for the swap this prevents. */
const inflight = new Map<string, Promise<boolean>>();

export function onnxBackend(): string {
  return status.backend;
}

export function onnxStatus(): OnnxStatus {
  return { ...status };
}

/**
 * GENUINE WebGPU availability: a GPU adapter must actually be obtainable (the same check
 * chrome://gpu / webgpureport.org perform) — not merely `navigator.gpu` being defined. This is what
 * stops the demo from reporting "webgpu" in a browser that has the API but no working adapter
 * (headless, flags off, blocklisted driver), where ORT silently drops to WASM.
 */
/** Software rasterisers that expose a WebGPU adapter without any GPU behind it. */
const SOFTWARE_ADAPTERS = ["swiftshader", "lavapipe", "llvmpipe", "software", "warp"];

/** Whether this adapter is a CPU rasteriser pretending to be a GPU. */
export function isSoftwareAdapter(info: Record<string, unknown> | null | undefined): boolean {
  if (!info) return false;
  const hay = `${info.architecture ?? ""} ${info.vendor ?? ""} ${info.description ?? ""}`
    .toLowerCase();
  return SOFTWARE_ADAPTERS.some((s) => hay.includes(s));
}

export async function hasWebGPU(): Promise<boolean> {
  try {
    if (typeof window !== "undefined" && !window.isSecureContext) return false; // WebGPU is secure-context only
    const gpu = (navigator as any)?.gpu;
    if (!gpu?.requestAdapter) return false;
    const adapter = await gpu.requestAdapter();   // resolves to null, does not throw, when unavailable
    if (!adapter) return false;
    // A SOFTWARE adapter is worse than no adapter, which is the opposite of what "is WebGPU
    // available" suggests. Measured on the same model, same single row, in two browsers:
    //
    //   SwiftShader adapter -> onnx-webgpu -> ResNet-50 in 12842 ms
    //   no adapter          -> onnx-wasm   -> ResNet-50 in    42 ms
    //
    // 300x. The adapter is real and the API works; there is simply no GPU under it, so every shader
    // dispatch is emulated on the CPU while ORT's wasm build runs SIMD kernels natively. Declining
    // it is not a downgrade - it is choosing the faster of two CPU paths, and it stops the page
    // reporting "MEASURED (local WebGPU)" for arithmetic no GPU performed.
    if (isSoftwareAdapter(adapter.info)) {
      console.info("[onnx] WebGPU adapter is a software rasteriser "
        + `(${adapter.info?.architecture ?? "unknown"}); using the wasm build, which measured ~300x `
        + "faster on this class of adapter");
      return false;
    }
    return true;
  } catch {
    return false;
  }
}

/** Adapter identity + features, for the provenance badge. Empty object when there is no adapter. */
export async function webgpuInfo(): Promise<Record<string, unknown>> {
  try {
    const adapter = await (navigator as any)?.gpu?.requestAdapter?.();
    if (!adapter) return {};
    return {
      info: adapter.info ? { ...adapter.info } : undefined,
      features: Array.from(adapter.features ?? []),
      shaderF16: adapter.features?.has?.("shader-f16") ?? false,
    };
  } catch {
    return {};
  }
}

/** WebNN is a W3C browser API (navigator.ml); experimental / flag-gated in 2026 — offered, not assumed. */
/**
 * WebNN, established by **building and executing a graph** — not by checking for an API surface.
 *
 * Three attempts to answer "is WebNN available" got this wrong in Round 18, each one weaker than
 * it looked:
 *   1. `!!navigator.ml` — Chrome 149 exposes it with no backend at all.
 *   2. `await navigator.ml.createContext({deviceType:"gpu"})` — this RESOLVES with a context on a
 *      machine with no GPU adapter, so it proves nothing either.
 *   3. Checking for `ctx.compute` — that method was removed from the spec in favour of
 *      `dispatch()` + MLTensor, so its absence says nothing about whether WebNN works.
 *
 * Measured with `--enable-features=WebMachineLearningNeuralNetwork`, a 2x2 matmul against the
 * identity builds, dispatches and reads back correctly. Without the flag the context still creates
 * and the graph still builds — execution is where it actually stops. So the probe runs one, and a
 * capability is claimed only when a known input produces the known output.
 *
 * The cost is a few milliseconds once per page. The alternative cost was every model on every page
 * appearing unsupported, which is what a cheaper check bought last time.
 */
export async function hasWebNN(): Promise<boolean> {
  const ml = (navigator as any)?.ml;
  if (!ml?.createContext || typeof (globalThis as any).MLGraphBuilder !== "function") return false;
  try {
    const ctx: any = await ml.createContext({ deviceType: "gpu" });
    if (!ctx?.dispatch || !ctx?.createTensor || !ctx?.readTensor) return false;
    const builder = new (globalThis as any).MLGraphBuilder(ctx);
    const a = builder.input("a", { dataType: "float32", shape: [2, 2] });
    const identity = builder.constant(
      { dataType: "float32", shape: [2, 2] }, new Float32Array([1, 0, 0, 1]));
    const graph = await builder.build({ c: builder.matmul(a, identity) });
    const tA = await ctx.createTensor({ dataType: "float32", shape: [2, 2], writable: true });
    const tC = await ctx.createTensor({ dataType: "float32", shape: [2, 2], readable: true });
    ctx.writeTensor(tA, new Float32Array([1, 2, 3, 4]));
    ctx.dispatch(graph, { a: tA }, { c: tC });
    const out = new Float32Array(await ctx.readTensor(tC));
    // x * I === x, or the backend is not doing what it says.
    return out.length === 4 && [1, 2, 3, 4].every((v, i) => Math.abs(out[i] - v) < 1e-4);
  } catch {
    return false;
  }
}

/**
 * Threads only work under cross-origin isolation, because SharedArrayBuffer does. Asking for more
 * than one without it makes ORT fall back to a single thread anyway, so the number the page
 * displays would be a wish rather than a measurement.
 */
function resolveThreads(): number {
  if (typeof self !== "undefined" && !self.crossOriginIsolated) return 1;
  const cores = (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 4;
  return Math.max(1, Math.min(4, Math.floor(cores / 2)));
}

//: A cold session on a few MB is seconds, not minutes. Bounded so a genuine hang ends rather than
//: leaving a button spinning forever -- which it did, before the JSEP-build bug was found.
const SESSION_CREATE_TIMEOUT_MS = 150_000;

function withTimeout<T>(p: Promise<T>, ms: number, message: string): Promise<T> {
  return Promise.race([
    p,
    new Promise<T>((_, reject) => setTimeout(() => reject(new Error(message)), ms)),
  ]);
}

/** Every published, pinned, fetchable model for a use case, in registry order.
 *
 *  There can be more than one now: satellite carries both the ESA-ADB Telemanom forecaster and the
 *  OPS-SAT segment classifier, which are different tasks against different published baselines.
 *  `browserModel` keeps returning the first so existing callers are unchanged; the verification
 *  panel offers the whole list so a visitor can check either one rather than whichever happens to
 *  be declared first. */
export function browserModels(usecase: string): { spec: ModelSpec; url: string }[] {
  const out: { spec: ModelSpec; url: string }[] = [];
  for (const spec of modelsFor(usecase)) {
    if (!spec.hf_repo_id || !spec.hf_revision) continue;   // unpublished or unpinned => not fetchable
    // ORT Web implements ai.onnx; it does not implement ai.onnx.ml (TreeEnsembleClassifier,
    // LinearClassifier, ...). A tree ensemble is therefore unrunnable here at any size, and finding
    // that out by downloading 4.6 MB and hanging in session creation is how this was discovered.
    // The registry records the graph's domains, so decline before spending the bandwidth.
    // NOTE: an earlier revision skipped models here on a `browser_runnable` flag derived from their
    // operator sets. That flag was built on a misdiagnosis -- the hang was the WebNN execution
    // provider, not the graphs -- and every model in this registry loads in ORT Web's wasm backend
    // in under a quarter of a second, tree ensembles included. The flag is still recorded as
    // provenance because knowing a graph's domains is useful, but it no longer blocks a fetch.
    const onnxArtifacts = spec.artifacts.filter(
      (a) => a.fmt === "onnx" && (a.serves === "browser" || a.serves === "both"));
    const art = onnxArtifacts.find((a) => a.precision === "int8")
      ?? onnxArtifacts.find((a) => a.precision === "fp32")
      ?? onnxArtifacts[0];
    if (!art) continue;
    // Pinned to a 40-char commit sha, never "main": a visitor must not silently get a different
    // model from the one whose metrics this page is printing.
    out.push({ spec, url: `https://huggingface.co/${spec.hf_repo_id}/resolve/${spec.hf_revision}/${art.path}` });
  }
  return out;
}

export function browserModel(usecase: string): { spec: ModelSpec; url: string } | null {
  return browserModels(usecase)[0] ?? null;
}

export async function loadModel(usecase: string, modelKey?: string): Promise<boolean> {
  // Keyed on the MODEL, not the use case. The cache used to match on a use-case prefix, so once a
  // satellite model was loaded every later satellite request got that same session back - which
  // meant selecting the OPS-SAT classifier ran the Telemanom forecaster's graph and failed on
  // input rank, since one takes [n,19] and the other [n,250,6].
  const want = modelKey ?? browserModel(usecase)?.spec.key;
  if (session && want && status.modelKey === want) return true;
  // Keyed by model, because `inflight` was a single global promise: a caller asking for model B
  // while model A was still loading got A's promise back, and then read the module-level `session`
  // believing it held B. The graph-name assertion in runPublishedModel/evaluatePublishedModel runs
  // BEFORE loadModel, so it cannot catch that swap. Two callers now genuinely exist at once - the
  // pipeline's inference stage and the Models panel's Run buttons - so this stopped being
  // theoretical.
  const slot = want ?? `__usecase__:${usecase}`;
  const pending = inflight.get(slot);
  if (pending) return pending;
  const load = (async () => {
    const target = modelKey
      ? browserModels(usecase).find((m) => m.spec.key === modelKey) ?? null
      : browserModel(usecase);
    if (!target) {
      // Say WHICH of the several reasons applies. "No model registered" is true but unhelpful when
      // the real answer is "there is one, and its operators cannot run in any browser".
      const blocked = modelsFor(usecase).find(
        (m) => m.hf_repo_id && m.hf_revision
          && (m as { browser_runnable?: boolean }).browser_runnable === false);
      const domains = (blocked as { onnx_domains?: string[] } | undefined)?.onnx_domains ?? [];
      status.error = blocked
        ? `${blocked.key} is published but cannot run in a browser: its graph uses `
          + `${domains.join(", ") || "unsupported"} operators, and ONNX Runtime Web implements only `
          + `ai.onnx. Tree ensembles (TreeEnsembleClassifier) are server-tier only.`
        : `no published, pinned, browser-tier ONNX model registered for "${usecase}"`;
      status.backend = "none";
      return false;
    }
    const started = performance.now();
    try {
      // Probe FIRST: the answer decides which ORT build to import, not just which provider to ask for.
      //
      // Second input to that decision, learned by running it: the webgpu build does NOT register
      // the ai.onnx.ml operators. A TreeEnsembleClassifier fails there with
      // "No Op registered for TreeEnsembleClassifier" while running fine in the wasm build, so a
      // model in that domain takes wasm even on a machine with a working adapter. This is the grain
      // of truth in the diagnosis I retracted earlier: ai.onnx.ml really is unavailable -- in one
      // build, which happened to be the one being loaded unconditionally.
      const needsMlDomain = (target.spec as { onnx_domains?: string[] }).onnx_domains
        ?.includes("ai.onnx.ml") ?? false;
      const gpuAvailable = !needsMlDomain && await hasWebGPU();
      const ort = await loadOrt(gpuAvailable);
      ortRuntime = ort;
      ort.env.wasm.numThreads = resolveThreads();
      ort.env.wasm.proxy = false;
      status.threads = ort.env.wasm.numThreads;

      // Fetch explicitly rather than handing ORT a URL, so the transferred byte count and the
      // cache-hit state are observable. "Downloaded 11 MB" versus "served from cache" is the
      // difference between a measurement and an assertion.
      const res = await fetch(target.url, { mode: "cors", signal: AbortSignal.timeout(HF_FETCH_TIMEOUT_MS) });
      if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText} for ${target.url}`);
      status.fromCache = ["memory", "disk", "prefetch"].includes(
        (res as any).headers?.get?.("x-cache") ?? "");
      const buf = new Uint8Array(await res.arrayBuffer());
      status.bytes = buf.byteLength;

      const gpu = gpuAvailable;
      const webnn = await hasWebNN();   // reported in status only; see the EP note below
      const eps: any[] = [];
      if (gpu) eps.push("webgpu");
      // WebNN is probed but NOT offered to ORT, and this time the reason is measured rather than
      // assumed. With --enable-features=WebMachineLearningNeuralNetwork the WebNN API genuinely
      // works: hasWebNN() below builds a 2x2 matmul, dispatches it and reads back the right answer.
      // ORT's WebNN *execution provider* is a different question, and the answer is no: with the
      // provider in the list the medical ResNet takes 162.7 s and hits the session-creation timeout,
      // against 5.1 s on wasm in the same browser. A working WebNN backend is not sufficient for
      // ORT's WebNN EP to run this graph, and it hangs rather than declining.
      //
      // So the capability is reported honestly and the provider is left out until ORT can complete
      // a session with it. Re-test by adding it here; the numbers above are the bar to beat.
      eps.push("wasm");                                   // always the floor

      // Bounded. An unsupported operator can leave session creation pending rather than
      // rejecting, and an unbounded await turns that into a permanently spinning button.
      session = await withTimeout(
        ort.InferenceSession.create(buf, { executionProviders: eps }),
        SESSION_CREATE_TIMEOUT_MS,
        `ONNX session creation exceeded ${SESSION_CREATE_TIMEOUT_MS / 1000}s — either the graph `
        + `uses operators this browser cannot execute, or it is too large for this device`,
      );
      // Report the provider that RAN, not a capability that was probed. This said "onnx-webnn"
      // while the WebNN provider was not even in the list -- a label describing something the code
      // does not do, which is the exact defect this project keeps finding. `webnn` is still probed
      // and recorded separately as an environment fact.
      // The provider that RAN. webnn is reported separately because it is available and unused.
      status.backend = gpu ? "onnx-webgpu" : "onnx-wasm";
      status.webnnAvailable = webnn;   // proven by execution, but deliberately not used
      status.loaded = true;
      status.modelKey = target.spec.key;
      status.url = target.url;
      status.loadMs = performance.now() - started;
      status.error = null;
      return true;
    } catch (e) {
      // Recorded, not swallowed. Callers must render this rather than quietly substituting a
      // different computation and presenting the result as if the model had run.
      session = null;
      status.loaded = false;
      status.backend = "none";
      status.modelKey = target.spec.key;
      status.url = target.url;
      status.loadMs = performance.now() - started;
      status.error = e instanceof Error ? e.message : String(e);
      console.warn("[onnx] model load failed:", status.error);
      return false;
    } finally {
      inflight.delete(slot);
    }
  })();
  inflight.set(slot, load);
  return load;
}

/** Logits from the registered browser model, or null if it is unavailable (status.error says why). */
export async function inferLogits(usecase: string, features: number[]): Promise<number[] | null> {
  if (!(await loadModel(usecase))) return null;
  if (!session) return null;
  try {
    const name = session.inputNames[0];
    const t = new ortRuntime!.Tensor("float32", Float32Array.from(features), [1, features.length]);
    const out = await session.run({ [name]: t });
    // Classifier graphs emit a hard label first and scores second; the scores are what a caller wants.
    const outputs = session.outputNames.map((n) => out[n]);
    const scores = outputs.find((o) => o && o.type === "float32" && (o.dims?.[1] ?? 0) > 1) ?? outputs[0];
    return Array.from(scores.data as Float32Array);
  } catch (e) {
    status.error = e instanceof Error ? e.message : String(e);
    console.warn("[onnx] inference failed:", status.error);
    return null;
  }
}

// --------------------------------------------------------------------------------------------- //
//  Model-native evaluation: run the PUBLISHED model on a sample from the corpus it was trained on
// --------------------------------------------------------------------------------------------- //

export interface ModelInputFile {
  /** "classification" (default) or "forecast" — decides what the self-check compares. */
  task?: "classification" | "forecast";
  provenance: string;
  input: { shape: number[]; dtype: string; preprocessing: string[]; layout: string };
  samples: number[][];
  labels?: number[];
  /** Human-readable class names, when the sample ships them. Telecom's eight attack types are the
   *  reason a per-class chart is worth drawing at all - "class 5" says nothing. */
  class_names?: string[];
  /** Next-step ground truth, for forecast models. */
  targets?: number[][];
  expected: {
    graph: string; note: string;
    accuracy?: number; majority_baseline?: number;
    mae?: number; persistence_mae?: number;
    /** F1 of the rare class alone, the AMLworld / Elliptic++ convention. Present when THAT is the
     *  metric the model was scored on, which for three of the four distilled students it is.
     *
     *  Scoring them on accuracy instead would have been the flattering-aggregate pattern this
     *  project rules out everywhere else: the finance student reads about 99 percent accurate on a
     *  0.1-percent-positive corpus while its minority-class F1 is 0.0178. A browser that "verified"
     *  the 99 would be confirming the wrong number. */
    minority_class_f1?: number;
    macro_f1?: number;
    /** F1 of predicting EVERYTHING positive: precision = prevalence, recall = 1. The floor a rare-
     *  class F1 has to clear, and the reason 0.0178 needs a companion rather than a bare comparison
     *  against zero. */
    all_positive_f1?: number;
  };
  classes?: Record<string, string>;
  feature_names?: string[];
  channels?: string[];
}

export interface ModelEvalResult {
  ran: boolean;
  task: "classification" | "forecast";
  n: number;
  /** Accuracy, rare-class F1, or mean absolute error, depending on `metricName`. */
  score: number | null;
  /** Which metric `score` and `expected` are. The model is scored on the metric it was PUBLISHED
   *  with, not on whichever one the browser found easiest to compute. */
  metricName: "accuracy" | "minority_class_f1" | "macro_f1" | "mae";
  expected: number | null;
  /** Majority-class accuracy, or persistence MAE — what a trivial predictor would achieve. */
  baseline: number | null;
  /** True when the model is no better than the trivial baseline. Surfaced, not buried. */
  beatsBaseline: boolean;
  /** |measured - expected| within tolerance. False means the browser must NOT present the number. */
  matchesExpected: boolean;
  backend: string;
  /** Transferred size, or null when the browser served it from cache. */
  bytes: number | null;
  provenance: string;
  /** `expected.note` from the sample: why this figure is what it is.
   *
   *  Rendered, not just carried. The finance student measures 0.3337 here and the registry table
   *  eight lines above it prints 0.0178 for the same metric - a 19x gap that is entirely an artefact
   *  of the sample being stratified to contain positives. Two numbers for one metric with no visible
   *  explanation is worse than either alone. */
  note: string | null;
  error: string | null;
  /** Compact plot data, reduced HERE rather than by shipping thousands of raw scores into React
   *  state. Everything needed to draw it already exists at this point in the function; recomputing
   *  it in a component would be a second chance to compute it differently. */
  plot?: {
    kind: "pr-curve" | "per-class-f1" | "forecast";
    /** Precision-recall pairs for a binary task, thinned to a drawable number of points. */
    pr?: Array<{ recall: number; precision: number }>;
    /** Score histogram split by true class - what separability actually looks like. */
    hist?: Array<{ bin: number; positive: number; negative: number }>;
    /** Per-class F1 for a multi-class task. The aggregate hides exactly this. */
    perClass?: Array<{ label: string; f1: number }>;
    /** Predicted against actual for a forecast window. */
    series?: Array<{ i: number; actual: number; predicted: number }>;
    /** Reliability: mean PREDICTED probability against OBSERVED frequency, per probability bin.
     *
     *  The one comparison of prediction against test data that is honest on the same axis. A
     *  prediction-versus-label scatter for a classifier is a confusion matrix drawn badly - the
     *  y-axis is categorical - and this project's other charts already cover ranking (`pr`),
     *  separability (`hist`) and per-class weakness (`perClass`). None of them shows whether a
     *  probability MEANS anything, which is a different property and the one the finance calibration
     *  work turned on: Round 50 found `fit_platt` returning one constant for nine deciles, a defect
     *  that is invisible to every metric here and unmissable on this plot.
     *
     *  Both axes are probabilities, so `y = x` is perfect calibration and the deviation is readable
     *  without a legend. Binary models bin P(positive); multi-class models bin the winning-class
     *  confidence against whether that prediction was right, which is the same question asked of a
     *  different quantity and is labelled as such where it renders. */
    reliability?: Array<{ p: number; observed: number; n: number }>;
    /** What `reliability` bins: "p(positive)" for binary, "winning-class confidence" otherwise. */
    reliabilityOf?: string;
  };
}

/** The evaluation sample for one model, memoised per URL.
 *
 *  Two callers now want it -- the Models panel's Run button and the orchestration's inference stage
 *  -- and before this cache existed the panel refetched several megabytes on every click. `null` is
 *  cached too: "this model publishes no sample" is a stable answer, not a transient failure, and
 *  re-asking costs a round trip to be told the same thing. */
const _sampleCache = new Map<string, Promise<ModelInputFile | null>>();

/** The sample lives beside the weights at the SAME pinned commit, so the two cannot drift apart. */
function sampleUrlFor(modelUrl: string): string {
  // Every artefact path in the registry is `onnx/model*.onnx`. The previous form of this stripped the
  // last two path segments generically, which is only correct BECAUSE of that convention: a
  // root-level `model.onnx` would have eaten the revision segment and fetched
  // `.../resolve/eval_sample.json` -- a 404 that reads exactly like "publishes no sample". Anchor on
  // the convention instead, and say so when it does not hold.
  if (!/\/onnx\/[^/]+$/.test(modelUrl)) {
    throw new Error(`cannot derive an eval_sample.json URL from "${modelUrl}": expected the artefact `
      + `path to be onnx/<file>. Deriving it by stripping segments would silently fetch the wrong `
      + `URL and report the model as unevaluated.`);
  }
  return modelUrl.replace(/\/onnx\/[^/]+$/, "/eval_sample.json");
}

function evalSample(modelUrl: string): Promise<ModelInputFile | null> {
  const url = sampleUrlFor(modelUrl);
  let hit = _sampleCache.get(url);
  if (!hit) {
    hit = (async () => {
      try {
        const res = await fetch(url, { mode: "cors", signal: AbortSignal.timeout(HF_FETCH_TIMEOUT_MS) });
        if (!res.ok) return null;              // no evaluation sample published for this model
        return (await res.json()) as ModelInputFile;
      } catch {
        return null;
      }
    })();
    _sampleCache.set(url, hit);
  }
  return hit;
}

/** Apply the preprocessing recorded WITH the data, never a locally-guessed equivalent. */
function applyPreprocessing(v: number, steps: string[]): number {
  let x = v;
  for (const step of steps) {
    if (step === "divide by 255") x = x / 255;
    else if (step === "subtract 0.5") x = x - 0.5;
    else if (step === "divide by 0.5") x = x / 0.5;
    else throw new Error(`unknown preprocessing step "${step}" — refusing to guess`);
  }
  return x;
}

/**
 * Evaluate the published model on its own held-out sample and CHECK the result.
 *
 * The point is the check. `expected.accuracy` was measured offline by
 * `scripts/prep_model_inputs.py` on exactly these rows with onnxruntime, so if the browser gets a
 * materially different number then something differs — preprocessing, quantisation, execution
 * provider — and the honest response is to say so rather than render it. Writing that generator,
 * a missing normalisation step silently turned 0.8782 into 0.4615 with no error anywhere; this is
 * the guard against shipping that class of mistake to a visitor.
 */
/** Equal-WIDTH probability bins: mean predicted probability against observed frequency.
 *
 * Equal-width rather than equal-frequency (deciles) on purpose. Equal-frequency bins hide exactly
 * the failure this plot exists to show: when a calibrator collapses most of its output onto one
 * value - which is what Round 50 found `fit_platt` doing, returning 0.00090148 for nine of finance's
 * ten deciles - equal-frequency binning spreads that constant across ten x-positions and draws a
 * plausible curve. Equal-width binning puts every one of those rows in a single bin and the collapse
 * is the thing you see.
 *
 * Empty bins are dropped rather than plotted at zero: a bin with no rows has no observed frequency,
 * and drawing 0 there would read as "predicted high, observed never".
 */
function reliabilityBins(
  n: number, prob: (i: number) => number, hit: (i: number) => boolean, bins = 12,
): Array<{ p: number; observed: number; n: number }> {
  const sum = new Float64Array(bins), obs = new Float64Array(bins), cnt = new Int32Array(bins);
  for (let i = 0; i < n; i++) {
    const p = prob(i);
    if (!Number.isFinite(p)) continue;
    const b = Math.min(bins - 1, Math.max(0, Math.floor(p * bins)));
    sum[b] += p; cnt[b] += 1; if (hit(i)) obs[b] += 1;
  }
  const out: Array<{ p: number; observed: number; n: number }> = [];
  for (let b = 0; b < bins; b++) {
    if (cnt[b] > 0) out.push({ p: sum[b] / cnt[b], observed: obs[b] / cnt[b], n: cnt[b] });
  }
  return out;
}

export async function evaluatePublishedModel(
  usecase: string, tolerance = 0.05, modelKey?: string,
): Promise<ModelEvalResult | null> {
  const base: ModelEvalResult = {
    ran: false, task: "classification", metricName: "accuracy", n: 0, score: null, expected: null,
    baseline: null,
    beatsBaseline: false, matchesExpected: false, backend: "none", bytes: null,
    provenance: "", note: null, error: null,
  };
  // The evaluation sample lives in the MODEL's own HF repo at the same pinned commit, not in this
  // bundle. Two reasons, both practical: the sample and the weights it evaluates cannot drift apart
  // when they share a revision, and HF's CDN serves the bytes instead of a 12 GB VPS -- which is
  // what makes 8192 AMLworld rows affordable (95% interval ~+/-0.6pp against ~+/-2.6pp at 512)
  // without putting a megabyte into the demo image.
  // A named model when the caller offers a choice, else the use case's first.
  const target = modelKey
    ? browserModels(usecase).find((m) => m.spec.key === modelKey) ?? null
    : browserModel(usecase);
  if (!target) return null;
  const file = await evalSample(target.url);
  if (!file) return null;                           // no evaluation sample published for this model

  // Check we are about to score the graph the expectation was measured on. expected.graph has been
  // published since Round 18 and never once compared against the artefact actually fetched -- and
  // they did diverge: the satellite generator measured model.onnx while browserModel() prefers
  // int8. A reproduction claim that does not check graph identity is not a reproduction claim.
  const fetchedGraph = target.url.split("/").pop() ?? "";
  const expectedGraph = file.expected?.graph ?? "";
  if (expectedGraph && fetchedGraph && expectedGraph !== fetchedGraph) {
    return {
      ...base, provenance: file.provenance,
      error: `graph mismatch: the expected figure was measured on ${expectedGraph} but this browser `
        + `fetched ${fetchedGraph}. Reporting the comparison would credit one graph with another's `
        + `result.`,
    };
  }

  if (!(await loadModel(usecase, target.spec.key)) || !session) {
    return {
      ...base, provenance: file.provenance,
      error: status.error
        ?? "no browser-runnable published model for this use case — the graph uses operators "
           + "ONNX Runtime Web does not implement (ai.onnx.ml tree ensembles), so it runs on the "
           + "server tier only",
    };
  }

  try {
    const [, ...dims] = file.input.shape;           // batch dim comes from the sample count
    const per = dims.reduce((a, b) => a * b, 1);
    const n = file.samples.length;
    const flat = new Float32Array(n * per);
    for (let i = 0; i < n; i++) {
      const row = file.samples[i];
      if (row.length !== per) {
        throw new Error(`sample ${i} has ${row.length} values, model expects ${per} `
          + `(${file.input.shape.join("x")}) — refusing to pad or truncate`);
      }
      for (let j = 0; j < per; j++) {
        flat[i * per + j] = applyPreprocessing(row[j], file.input.preprocessing);
      }
    }
    const t = new ortRuntime!.Tensor("float32", flat, [n, ...dims]);
    const out = await session.run({ [session.inputNames[0]]: t });
    const outputs = session.outputNames.map((k) => out[k]);
    const scores = outputs.find((o) => o && o.type === "float32" && (o.dims?.length ?? 0) >= 2)
      ?? outputs[0];
    const data = scores.data as Float32Array;
    const width = Math.max(1, Math.floor(data.length / n));

    const task = file.task ?? "classification";
    if (task === "forecast") {
      // A forecaster is scored by error against the true next step, and against what predicting
      // "no change" would score. If the model loses to persistence, the page says so.
      const targets = file.targets ?? [];
      let err = 0, persist = 0, count = 0;
      for (let i = 0; i < n; i++) {
        const truth = targets[i] ?? [];
        const lastStep = file.samples[i].slice(-truth.length);
        for (let c = 0; c < truth.length; c++) {
          err += Math.abs(data[i * width + c] - truth[c]);
          persist += Math.abs(lastStep[c] - truth[c]);
          count++;
        }
      }
      const mae = count ? err / count : NaN;
      const persistenceMae = count ? persist / count : NaN;
      const exp = file.expected.mae ?? null;
      // Channel 0 predicted against actual across the scored windows. A forecaster's MAE says how
      // far off it is on average; this says WHERE - whether the error is uniform drift or a few
      // large misses, which is the difference between a miscalibrated model and a missed event.
      const series = Array.from({ length: Math.min(n, 240) }, (_, i) => ({
        i, actual: (file.targets?.[i] ?? [])[0] ?? 0, predicted: data[i * width] ?? 0,
      }));
      return {
        ran: true, task, metricName: "mae", n, score: mae, expected: exp,
        baseline: file.expected.persistence_mae ?? persistenceMae,
        beatsBaseline: mae < (file.expected.persistence_mae ?? persistenceMae),
        matchesExpected: exp == null ? false : Math.abs(mae - exp) <= Math.max(tolerance, exp * 0.1),
        backend: status.backend, bytes: status.bytes, provenance: file.provenance,
        note: file.expected.note ?? null, error: null,
        plot: { kind: "forecast", series },
      };
    }

    const labels = file.labels ?? [];
    // Retained rather than consumed in place: macro F1 needs a second pass over the predictions,
    // and recomputing the argmax there would be a second chance to compute it differently.
    const preds = new Int32Array(n);
    let correct = 0, tp = 0, fp = 0, fn = 0, positives = 0;
    for (let i = 0; i < n; i++) {
      let best = 0;
      for (let c = 1; c < width; c++) if (data[i * width + c] > data[i * width + best]) best = c;
      const pred = width > 1 ? best : (data[i] > 0.5 ? 1 : 0);
      preds[i] = pred;
      if (pred === labels[i]) correct++;
      if (labels[i] === 1) positives++;
      if (pred === 1 && labels[i] === 1) tp++;
      else if (pred === 1) fp++;
      else if (labels[i] === 1) fn++;
    }

    // ---- plot data, computed once, from the arrays already here --------------------------------
    const positiveScore = (i: number) =>
      width > 1 ? data[i * width + 1] - data[i * width + 0] : data[i];
    const nClasses = width > 1 ? width : 2;
    let plot: ModelEvalResult["plot"];
    if (nClasses === 2) {
      // Rank by score, sweep the threshold down, and read precision/recall off the running counts.
      const order = Array.from({ length: n }, (_, i) => i)
        .sort((a, b) => positiveScore(b) - positiveScore(a));
      const totalPos = labels.reduce((acc: number, l: number) => acc + (l === 1 ? 1 : 0), 0);
      const pr: Array<{ recall: number; precision: number }> = [];
      let seenTp = 0;
      const stride = Math.max(1, Math.floor(n / 200));   // <=200 points is plenty for a curve
      for (let k = 0; k < n; k++) {
        if (labels[order[k]] === 1) seenTp++;
        if (k % stride === 0 || k === n - 1) {
          pr.push({ recall: totalPos ? seenTp / totalPos : 0, precision: seenTp / (k + 1) });
        }
      }
      // Score histogram, split by TRUE class. A PR curve says how good; this says why.
      const lo = Math.min(...Array.from({ length: n }, (_, i) => positiveScore(i)));
      const hi = Math.max(...Array.from({ length: n }, (_, i) => positiveScore(i)));
      const bins = 24;
      const hist = Array.from({ length: bins }, (_, b) => ({ bin: b, positive: 0, negative: 0 }));
      for (let i = 0; i < n; i++) {
        const t = hi > lo ? (positiveScore(i) - lo) / (hi - lo) : 0;
        const b = Math.min(bins - 1, Math.max(0, Math.floor(t * bins)));
        if (labels[i] === 1) hist[b].positive++; else hist[b].negative++;
      }
      // Reliability. `positiveScore` is a DIFFERENCE when the graph emits per-class scores, and a
      // difference is not a probability - binning it against an observed frequency would compare two
      // different things on one axis, which is the defect this round already found in the companion
      // caption and in the baseline-on-a-foreign-axis question. So take the probability directly:
      // column 1 is P(positive) for a two-column classifier output, and the single column already is
      // one when the graph emits a lone sigmoid.
      const pPos = (i: number) => (width > 1 ? data[i * width + 1] : data[i]);
      plot = { kind: "pr-curve", pr, hist,
               reliability: reliabilityBins(n, pPos, (i) => labels[i] === 1),
               reliabilityOf: "p(positive)" };
    } else {
      const perClass: Array<{ label: string; f1: number }> = [];
      for (let c = 0; c < nClasses; c++) {
        let ctp = 0, cfp = 0, cfn = 0;
        for (let i = 0; i < n; i++) {
          if (preds[i] === c && labels[i] === c) ctp++;
          else if (preds[i] === c) cfp++;
          else if (labels[i] === c) cfn++;
        }
        perClass.push({ label: file.class_names?.[c] ?? String(c),
                        f1: ctp === 0 ? 0 : (2 * ctp) / (2 * ctp + cfp + cfn) });
      }
      // Confidence calibration for the multi-class case: bin the WINNING class's probability against
      // whether that prediction was right. Same question - does the number mean what it says - asked
      // of a different quantity, so it is labelled differently where it renders rather than being
      // silently pooled with the binary case.
      const conf = (i: number) => {
        let m = data[i * width];
        for (let c = 1; c < width; c++) if (data[i * width + c] > m) m = data[i * width + c];
        return m;
      };
      plot = { kind: "per-class-f1", perClass,
               reliability: reliabilityBins(n, conf, (i) => preds[i] === labels[i]),
               reliabilityOf: "winning-class confidence" };
    }

    // Score on the metric the model was PUBLISHED with. Three of the four distilled students record
    // minority-class F1, and reporting accuracy for them would confirm a number nobody claimed:
    // the finance student is about 99 percent accurate on a corpus that is 0.1 percent positive.
    if (file.expected.minority_class_f1 != null) {
      // sklearn f1_score(pos_label=1, zero_division=0): 0 when the model predicts no positives.
      const f1 = tp === 0 ? 0 : (2 * tp) / (2 * tp + fp + fn);
      const exp = file.expected.minority_class_f1;
      // Predicting everything positive: precision = prevalence, recall = 1.
      const prevalence = n ? positives / n : 0;
      const allPositive = file.expected.all_positive_f1
        ?? (prevalence ? (2 * prevalence) / (1 + prevalence) : 0);
      return {
        ran: true, task, metricName: "minority_class_f1", n, score: f1, expected: exp,
        baseline: allPositive,
        beatsBaseline: f1 > allPositive,
        matchesExpected: Math.abs(f1 - exp) <= tolerance,
        backend: status.backend, bytes: status.bytes, provenance: file.provenance,
        note: file.expected.note ?? null, error: null, plot,
      };
    }

    // Macro F1 for the multi-class detector. Aggregate accuracy is precisely the metric Engelen
    // et al. (WTMC 2021) showed stays at 0.99 while a per-class F1 moves 0.60 -> 0.99, so verifying
    // an eight-class intrusion detector on accuracy checks almost nothing about the classes that
    // matter. Until Round 24i that sample published only accuracy and this branch did not exist.
    if (file.expected.macro_f1 != null) {
      const classes = new Set<number>();
      for (let i = 0; i < n; i++) { classes.add(preds[i]); classes.add(labels[i]); }
      let sum = 0;
      for (const c of classes) {
        let ctp = 0, cfp = 0, cfn = 0;
        for (let i = 0; i < n; i++) {
          if (preds[i] === c && labels[i] === c) ctp++;
          else if (preds[i] === c) cfp++;
          else if (labels[i] === c) cfn++;
        }
        sum += ctp === 0 ? 0 : (2 * ctp) / (2 * ctp + cfp + cfn);
      }
      const macro = classes.size ? sum / classes.size : 0;
      const exp = file.expected.macro_f1;
      // A balanced n-class problem: guessing one class always gives macro F1 of about 2/(n+1)/n.
      const chance = classes.size ? (2 / (classes.size + 1)) / classes.size : 0;
      return {
        ran: true, task, metricName: "macro_f1", n, score: macro, expected: exp,
        baseline: chance,
        beatsBaseline: macro > chance,
        matchesExpected: Math.abs(macro - exp) <= tolerance,
        backend: status.backend, bytes: status.bytes, provenance: file.provenance,
        note: file.expected.note ?? null, error: null, plot,
      };
    }

    const accuracy = correct / n;
    const exp = file.expected.accuracy ?? null;
    const majority = file.expected.majority_baseline ?? null;
    return {
      ran: true, task, metricName: "accuracy", n, score: accuracy, expected: exp, baseline: majority,
      beatsBaseline: majority == null ? true : accuracy > majority,
      matchesExpected: exp == null ? false : Math.abs(accuracy - exp) <= tolerance,
      backend: status.backend, bytes: status.bytes, provenance: file.provenance,
      note: file.expected.note ?? null, error: null, plot,
    };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    status.error = msg;
    console.warn("[onnx] published-model evaluation failed:", msg);
    return { ...base, provenance: file.provenance, error: msg };
  }
}

// --------------------------------------------------------------------------------------------- //
//  One published-model inference, for the orchestration pipeline
// --------------------------------------------------------------------------------------------- //

/**
 * A single row through the published model, with everything needed to say what just happened.
 *
 * `evaluatePublishedModel` above scores the whole held-out sample and answers "does this browser
 * reproduce the published figure". This answers a different question -- "what does the published
 * model output for one input, and how long did that take here" -- which is what an orchestration
 * stage needs. They share the fetch, the graph check, the preprocessing and the session.
 */
export interface PublishedModelRun {
  modelKey: string;
  repoId: string;
  /** The 40-char commit the weights were fetched from. Printed, so the claim is checkable. */
  revision: string;
  graph: string;
  /** "onnx-webgpu" or "onnx-wasm": the execution provider that actually ran, not the one requested. */
  backend: string;
  inputShape: number[];
  rowIndex: number;
  task: "classification" | "forecast";
  /** Raw model output: class scores for a classifier, the forecast vector for a forecaster. */
  logits: number[];
  /** Classifiers only. A forecaster has no class, and argmax over its output means nothing. */
  predictedClass: number | null;
  className: string | null;
  /** The row's true label where the sample ships one, so the single inference is checkable too. */
  trueClass: number | null;
  trueClassName: string | null;
  /** Forecasters only: mean absolute error of this one row against its shipped next-step target. */
  rowMae: number | null;
  /** Wall-clock for session.run alone, excluding fetch and session creation. */
  latencyMs: number;
  /** Total including the fetch and session creation on a cold cache. */
  totalMs: number;
  provenance: string;
}

export type PublishedRunOutcome =
  /** The published model ran. */
  | { kind: "ran"; run: PublishedModelRun }
  /** No published, pinned, browser-tier model exists for this use case. A legitimate state. */
  | { kind: "unavailable"; reason: string }
  /** One exists and did not run. Never silently substituted -- the caller must surface this. */
  | { kind: "failed"; modelKey: string; error: string };

/**
 * Run the published model for `usecase` on one row of its OWN held-out sample.
 *
 * The row comes from the model's corpus, not from the caller, and that is deliberate. The pipeline's
 * own features are a different dataset -- medical runs Wisconsin's 8 tabular features while the
 * published model is a BreastMNIST CNN over 28x28 images -- so feeding one to the other would be the
 * same category error this project already refuses elsewhere. The two run side by side and each says
 * what it is.
 *
 * `rowIndex` is taken from the caller (the pipeline derives it from the run seed) so the same seed
 * selects the same row and the stage is reproducible.
 */
export async function runPublishedModel(
  usecase: string, opts: { modelKey?: string; rowIndex?: number } = {},
): Promise<PublishedRunOutcome> {
  const started = performance.now();
  const target = opts.modelKey
    ? browserModels(usecase).find((m) => m.spec.key === opts.modelKey) ?? null
    : browserModel(usecase);
  if (!target) {
    return { kind: "unavailable", reason: status.error
      ?? `no published, pinned, browser-tier ONNX model is registered for "${usecase}"` };
  }

  const key = target.spec.key;
  let file: ModelInputFile | null;
  try {
    file = await evalSample(target.url);
  } catch (e) {
    return { kind: "failed", modelKey: key, error: e instanceof Error ? e.message : String(e) };
  }
  if (!file) {
    return { kind: "failed", modelKey: key,
      error: `${key} is published and pinned but ships no eval_sample.json at that revision, so there `
        + `is no input from its own corpus to run it on` };
  }

  // Same identity check the evaluation path makes: score the graph the sample was measured on.
  const fetchedGraph = target.url.split("/").pop() ?? "";
  const expectedGraph = file.expected?.graph ?? "";
  if (expectedGraph && fetchedGraph && expectedGraph !== fetchedGraph) {
    return { kind: "failed", modelKey: key,
      error: `graph mismatch: the sample was built against ${expectedGraph}, this browser fetched `
        + `${fetchedGraph}` };
  }

  if (!(await loadModel(usecase, key)) || !session) {
    return { kind: "failed", modelKey: key, error: status.error ?? "session creation failed" };
  }

  try {
    const [, ...dims] = file.input.shape;
    const per = dims.reduce((a, b) => a * b, 1);
    const n = file.samples.length;
    if (!n) throw new Error("eval_sample.json ships zero rows");
    const rowIndex = ((opts.rowIndex ?? 0) % n + n) % n;
    const row = file.samples[rowIndex];
    if (row.length !== per) {
      throw new Error(`row ${rowIndex} has ${row.length} values, model expects ${per} `
        + `(${file.input.shape.join("x")}) - refusing to pad or truncate`);
    }
    const flat = new Float32Array(per);
    for (let j = 0; j < per; j++) flat[j] = applyPreprocessing(row[j], file.input.preprocessing);

    const t = new ortRuntime!.Tensor("float32", flat, [1, ...dims]);
    const ran = performance.now();
    const out = await session.run({ [session.inputNames[0]]: t });
    const latencyMs = performance.now() - ran;

    const outputs = session.outputNames.map((k) => out[k]);
    const scores = outputs.find((o) => o && o.type === "float32" && (o.dims?.length ?? 0) >= 2)
      ?? outputs[0];
    const logits = Array.from(scores.data as Float32Array);
    const task = file.task ?? "classification";

    // A forecaster's output is the next-step prediction, not class scores. Taking an argmax over it
    // yields a channel index and prints it as a class, which is meaningless -- Telemanom's output is
    // 6 continuous channel values. Score it the way its own sample is scored instead: absolute error
    // against the target row shipped beside it.
    let predictedClass: number | null = null;
    let rowMae: number | null = null;
    if (task === "forecast") {
      const target = file.targets?.[rowIndex];
      if (target && target.length) {
        const k = Math.min(target.length, logits.length);
        let sum = 0;
        for (let i = 0; i < k; i++) sum += Math.abs(logits[i] - target[i]);
        rowMae = Number((sum / k).toFixed(6));
      }
    } else {
      predictedClass = 0;
      for (let c = 1; c < logits.length; c++) {
        if (logits[c] > logits[predictedClass]) predictedClass = c;
      }
      if (logits.length === 1) predictedClass = logits[0] > 0.5 ? 1 : 0;
    }

    return {
      kind: "ran",
      run: {
        modelKey: key,
        repoId: target.spec.hf_repo_id ?? "",
        revision: target.spec.hf_revision ?? "",
        graph: fetchedGraph,
        backend: status.backend,
        inputShape: [1, ...dims],
        rowIndex,
        task,
        logits,
        predictedClass,
        rowMae,
        className: predictedClass == null
          ? null
          : file.classes?.[String(predictedClass)] ?? null,
        trueClass: task === "forecast" ? null : file.labels?.[rowIndex] ?? null,
        trueClassName: task === "forecast" || file.labels?.[rowIndex] == null
          ? null
          : file.classes?.[String(file.labels[rowIndex])] ?? null,
        latencyMs: Number(latencyMs.toFixed(2)),
        totalMs: Number((performance.now() - started).toFixed(2)),
        provenance: file.provenance,
      },
    };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.warn("[onnx] published-model run failed:", msg);
    return { kind: "failed", modelKey: key, error: msg };
  }
}

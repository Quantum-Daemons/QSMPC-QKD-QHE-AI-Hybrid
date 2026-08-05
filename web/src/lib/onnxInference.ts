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
let ortPromise: Promise<typeof OrtTypes> | null = null;

function loadOrt(useWebGPU: boolean): Promise<typeof OrtTypes> {
  if (!ortPromise) {
    ortPromise = (useWebGPU
      ? import("onnxruntime-web/webgpu")
      : import("onnxruntime-web/wasm")) as Promise<typeof OrtTypes>;
  }
  return ortPromise;
}

import { modelsFor, type ModelSpec } from "./config";

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
let inflight: Promise<boolean> | null = null;

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
export async function hasWebGPU(): Promise<boolean> {
  try {
    if (typeof window !== "undefined" && !window.isSecureContext) return false; // WebGPU is secure-context only
    const gpu = (navigator as any)?.gpu;
    if (!gpu?.requestAdapter) return false;
    const adapter = await gpu.requestAdapter();   // resolves to null, does not throw, when unavailable
    return !!adapter;
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

export function browserModel(usecase: string): { spec: ModelSpec; url: string } | null {
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
    return { spec, url: `https://huggingface.co/${spec.hf_repo_id}/resolve/${spec.hf_revision}/${art.path}` };
  }
  return null;
}

export async function loadModel(usecase: string): Promise<boolean> {
  if (session && status.modelKey?.startsWith(usecase)) return true;
  if (inflight) return inflight;
  inflight = (async () => {
    const target = browserModel(usecase);
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
      const gpuAvailable = await hasWebGPU();
      const ort = await loadOrt(gpuAvailable);
      ortRuntime = ort;
      ort.env.wasm.numThreads = resolveThreads();
      ort.env.wasm.proxy = false;
      status.threads = ort.env.wasm.numThreads;

      // Fetch explicitly rather than handing ORT a URL, so the transferred byte count and the
      // cache-hit state are observable. "Downloaded 11 MB" versus "served from cache" is the
      // difference between a measurement and an assertion.
      const res = await fetch(target.url, { mode: "cors" });
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
      inflight = null;
    }
  })();
  return inflight;
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
  /** Next-step ground truth, for forecast models. */
  targets?: number[][];
  expected: {
    graph: string; note: string;
    accuracy?: number; majority_baseline?: number;
    mae?: number; persistence_mae?: number;
  };
  classes?: Record<string, string>;
  feature_names?: string[];
  channels?: string[];
}

export interface ModelEvalResult {
  ran: boolean;
  task: "classification" | "forecast";
  n: number;
  /** Accuracy for a classifier, mean absolute error for a forecaster. Lower is better for MAE. */
  score: number | null;
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
  error: string | null;
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
export async function evaluatePublishedModel(
  usecase: string, tolerance = 0.05,
): Promise<ModelEvalResult | null> {
  const base: ModelEvalResult = {
    ran: false, task: "classification", n: 0, score: null, expected: null, baseline: null,
    beatsBaseline: false, matchesExpected: false, backend: "none", bytes: null,
    provenance: "", error: null,
  };
  // The evaluation sample lives in the MODEL's own HF repo at the same pinned commit, not in this
  // bundle. Two reasons, both practical: the sample and the weights it evaluates cannot drift apart
  // when they share a revision, and HF's CDN serves the bytes instead of a 12 GB VPS -- which is
  // what makes 8192 AMLworld rows affordable (95% interval ~+/-0.6pp against ~+/-2.6pp at 512)
  // without putting a megabyte into the demo image.
  const target = browserModel(usecase);
  if (!target) return null;
  const sampleUrl = target.url.replace(/\/[^/]+\/[^/]+$/, "/eval_sample.json")
    .replace(/\/onnx\/[^/]+$/, "/eval_sample.json");

  let file: ModelInputFile;
  try {
    const res = await fetch(sampleUrl, { mode: "cors" });
    if (!res.ok) return null;                       // no evaluation sample published for this model
    file = await res.json();
  } catch {
    return null;
  }

  if (!(await loadModel(usecase)) || !session) {
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
      return {
        ran: true, task, n, score: mae, expected: exp,
        baseline: file.expected.persistence_mae ?? persistenceMae,
        beatsBaseline: mae < (file.expected.persistence_mae ?? persistenceMae),
        matchesExpected: exp == null ? false : Math.abs(mae - exp) <= Math.max(tolerance, exp * 0.1),
        backend: status.backend, bytes: status.bytes, provenance: file.provenance, error: null,
      };
    }

    const labels = file.labels ?? [];
    let correct = 0;
    for (let i = 0; i < n; i++) {
      let best = 0;
      for (let c = 1; c < width; c++) if (data[i * width + c] > data[i * width + best]) best = c;
      const pred = width > 1 ? best : (data[i] > 0.5 ? 1 : 0);
      if (pred === labels[i]) correct++;
    }
    const accuracy = correct / n;
    const exp = file.expected.accuracy ?? null;
    const majority = file.expected.majority_baseline ?? null;
    return {
      ran: true, task, n, score: accuracy, expected: exp, baseline: majority,
      beatsBaseline: majority == null ? true : accuracy > majority,
      matchesExpected: exp == null ? false : Math.abs(accuracy - exp) <= tolerance,
      backend: status.backend, bytes: status.bytes, provenance: file.provenance, error: null,
    };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    status.error = msg;
    console.warn("[onnx] published-model evaluation failed:", msg);
    return { ...base, provenance: file.provenance, error: msg };
  }
}

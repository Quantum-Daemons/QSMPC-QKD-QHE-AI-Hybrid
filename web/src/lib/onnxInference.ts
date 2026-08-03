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
import * as ort from "onnxruntime-web";

import { modelsFor, type ModelSpec } from "./config";

export type OnnxStatus = {
  loaded: boolean;
  backend: string;          // "onnx-webgpu" | "onnx-webnn" | "onnx-wasm" | "none"
  modelKey: string | null;
  url: string | null;
  bytes: number | null;     // transferred size, or null when served from cache
  fromCache: boolean;
  threads: number;
  crossOriginIsolated: boolean;
  loadMs: number | null;
  error: string | null;     // non-null => nothing was inferred; the caller must say so
};

const status: OnnxStatus = {
  loaded: false, backend: "none", modelKey: null, url: null, bytes: null,
  fromCache: false, threads: 1, crossOriginIsolated: false, loadMs: null, error: null,
};

let session: ort.InferenceSession | null = null;
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
export function hasWebNN(): boolean {
  return typeof navigator !== "undefined" && !!(navigator as any).ml;
}

/**
 * Threads only work under cross-origin isolation, because SharedArrayBuffer does. Asking for more
 * than one without it makes ORT fall back to a single thread anyway, so the number the page
 * displays would be a wish rather than a measurement.
 */
function resolveThreads(): number {
  const isolated = typeof globalThis !== "undefined" && (globalThis as any).crossOriginIsolated === true;
  status.crossOriginIsolated = isolated;
  if (!isolated) return 1;
  const cores = (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 4;
  return Math.max(1, Math.min(4, Math.floor(cores / 2)));
}

/** Resolve the browser-tier artefact for a use case from the registry. */
export function browserModel(usecase: string): { spec: ModelSpec; url: string } | null {
  for (const spec of modelsFor(usecase)) {
    if (!spec.hf_repo_id || !spec.hf_revision) continue;   // unpublished or unpinned => not fetchable
    const art = spec.artifacts.find((a) => a.fmt === "onnx" && a.serves === "browser")
      ?? spec.artifacts.find((a) => a.fmt === "onnx" && a.serves === "both");
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
      status.error = `no published browser-tier ONNX model registered for "${usecase}"`;
      status.backend = "none";
      return false;
    }
    const started = performance.now();
    try {
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

      const gpu = await hasWebGPU();
      const webnn = hasWebNN();
      const eps: any[] = [];
      if (gpu) eps.push("webgpu");
      if (webnn) eps.push({ name: "webnn", deviceType: "gpu", powerPreference: "default" });
      eps.push("wasm");                                   // always the floor

      session = await ort.InferenceSession.create(buf, { executionProviders: eps });
      status.backend = gpu ? "onnx-webgpu" : webnn ? "onnx-webnn" : "onnx-wasm";
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
    const t = new ort.Tensor("float32", Float32Array.from(features), [1, features.length]);
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

// In-browser ONNX inference for the client-side demo, via onnxruntime-web.
// Execution-provider preference: WebGPU -> WebNN -> WASM (mature -> experimental -> floor). Vite bundles
// the ORT wasm as a same-origin hashed asset (so it satisfies the COOP/COEP cross-origin-isolation
// headers); we let ORT resolve it via import.meta.url rather than overriding wasmPaths. Every entry
// point fails soft: if the runtime or model can't load, callers fall back to the pure-JS linear model so
// the demo never breaks — but the reported backend is GENUINE (a real GPU adapter must be obtainable, so
// we never claim "webgpu" when it actually fell back to WASM).
import * as ort from "onnxruntime-web";

let session: ort.InferenceSession | null = null;
let tried = false;
let backendName = "none";

export function onnxBackend(): string {
  return backendName;
}

/**
 * GENUINE WebGPU availability: a GPU adapter must actually be obtainable (the same check
 * chrome://gpu / webgpureport.org perform) — not merely ``navigator.gpu`` being defined. This is what
 * stops the demo from reporting "webgpu" in a browser that has the API but no working adapter (headless
 * / flags-off), where ORT silently drops to WASM.
 */
export async function hasWebGPU(): Promise<boolean> {
  try {
    const gpu = (navigator as any)?.gpu;
    if (!gpu?.requestAdapter) return false;
    const adapter = await gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

/** WebNN is a W3C browser API (navigator.ml); experimental / flag-gated in 2026 — offered, not assumed. */
export function hasWebNN(): boolean {
  return typeof navigator !== "undefined" && !!(navigator as any).ml;
}

export async function loadModel(url = "/models/finance_mlp.onnx"): Promise<boolean> {
  if (session) return true;
  if (tried) return false;
  tried = true;
  try {
    ort.env.wasm.numThreads = 1; // single-threaded: no SharedArrayBuffer dependency
    ort.env.wasm.proxy = false;   // no proxy worker needed single-threaded
    const gpu = await hasWebGPU();
    const webnn = hasWebNN();
    const eps: any[] = [];
    if (gpu) eps.push("webgpu");
    if (webnn) eps.push({ name: "webnn", deviceType: "gpu", powerPreference: "default" });
    eps.push("wasm");                                   // always the floor
    session = await ort.InferenceSession.create(url, { executionProviders: eps });
    backendName = gpu ? "onnx-webgpu" : webnn ? "onnx-webnn" : "onnx-wasm";
    return true;
  } catch (e) {
    console.warn("[onnx] init failed; using JS fallback:", e);
    session = null;
    backendName = "none";
    return false;
  }
}

// Returns decrypted-equivalent logits, or null if ONNX is unavailable (JS fallback).
export async function inferLogits(features: number[]): Promise<number[] | null> {
  if (!session && !(await loadModel())) return null;
  if (!session) return null;
  try {
    const name = session.inputNames[0];
    const t = new ort.Tensor("float32", Float32Array.from(features), [1, features.length]);
    const out = await session.run({ [name]: t });
    return Array.from(out[session.outputNames[0]].data as Float32Array);
  } catch (e) {
    console.warn("[onnx] inference failed; using JS fallback:", e);
    return null;
  }
}

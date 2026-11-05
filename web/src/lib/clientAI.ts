// Dual in-browser AI engines for the public demo (zero server). BOTH are loaded ONLY via dynamic
// import() so neither heavy library lands in the initial bundle — they become async chunks fetched
// on first use. The AI agent's output is then wrapped in the client quantum-safe MCP channel
// (clientMcp.ts): agentic AI with a quantum-safe MCP layer, entirely client-side.
//   - Transformers.js (@huggingface/transformers, Apache-2.0): runs ONNX models via onnxruntime-web,
//     on WASM by default and on WebGPU when the visitor selects it - the same default the private
//     inference stage uses. Until Round 60 it took WebGPU whenever an adapter existed; on the machine
//     measured that path ran distilGPT-2 about 80 times slower per token than WASM, and no first run
//     finished inside the timeout (register, Round 59).
//   - WebLLM (@mlc-ai/web-llm, Apache-2.0): a WebGPU chat-LLM engine (small models); WebGPU-required.
// Everything fails soft: if WebGPU / model-fetch is unavailable, runAgent returns a notice so the
// quantum-safe MCP demo still proceeds (headless / CI-safe).
//
// What a run fetches, and from where (checklist CS12): this site serves the engines' runtime files, ONNX
// Runtime's wasm and the WebLLM model libraries; the weights come from Hugging Face at a pinned revision
// and are browser-cached after the first run. Nothing else is contacted.

import { webCfg } from "./config";
// Same-origin copies of the two ONNX Runtime builds Transformers.js chooses between. `?url` emits the file
// and imports no code, and the wasm files are ones the build already ships for the inference stage.
import ortAsyncifyMjs from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs?url";
import ortAsyncifyWasm from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm?url";
import ortPlainMjs from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url";
import ortPlainWasm from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url";

export type Engine = "transformers" | "webllm";
/** The Transformers.js backend. WASM is the default (`web.tfjs_default_device`); WebGPU is opt-in. */
export type Device = "wasm" | "webgpu";

export interface AgentRun {
  engine: Engine; text: string; elapsedMs: number;
  backend: string; model: string; webgpu: boolean; note?: string;
}

/** A Transformers.js model as the config records it: pinned, with the weights each backend fetches. */
export interface TfjsModel {
  id: string; label: string; revision: string;
  dtype: Record<Device, string>; bytes: Record<Device, number>;
}

/** A WebLLM model as the config records it: its weights' pinned revision and the library this site serves. */
export interface WebLlmModel {
  id: string; label: string; revision: string; bytes: number; requires: string;
  lib: string; lib_sri: string; context_window_size: number;
}

/** What a picker shows: the name, the bytes a first run downloads, and the adapter feature it needs. */
export function webLlmModelLabel(m: WebLlmModel): string {
  return `${m.label} (${(m.bytes / 1e6).toFixed(0)} MB${m.requires ? `, needs ${m.requires}` : `, no shader-f16`})`;
}

/** What a picker shows: the name, the bytes the chosen backend downloads, and the weights' dtype. */
export function tfjsModelLabel(m: TfjsModel, device: Device): string {
  return `${m.label} (${(m.bytes[device] / 1e6).toFixed(0)} MB ${m.dtype[device]} weights)`;
}

// model lists + generation limits + timeout come from the central config (config.ts ← /api/config ←
// config/defaults.py: WebConfig) — single source of truth, no inline literals here.
export const TFJS_MODELS = webCfg().tfjs_models as TfjsModel[];
export const DEFAULT_DEVICE = webCfg().tfjs_default_device as Device;
export const WEBLLM_MODELS = webCfg().webllm_models as WebLlmModel[];

// GENUINE WebGPU check: a GPU adapter must actually be obtainable (matches chrome://gpu /
// webgpureport.org) — not merely navigator.gpu being defined — so the demo never claims WebGPU when it
// would silently run on WASM (no-easy-fallback: report the real backend).
//
// Re-exported, not reimplemented. This was a second copy, and a weaker one: it omitted the
// secure-context guard, and once onnxInference learned to decline software rasterisers it would have
// gone on answering "yes" to the question the ONNX tier now answers "no" to. The two tiers would
// then disagree about whether this machine has a GPU — the pipeline running on wasm while this panel
// offered WebLLM, whose shaders are exactly what a SwiftShader adapter is 300x too slow for.
import { hasWebGPU } from "./onnxInference";

export { hasWebGPU };

const SYSTEM = "You are a concise assistant for a quantum-safe AI orchestration demo.";

/** "downloading onnx/model_quantized.onnx 45% of 237 MB" from a Transformers.js progress event. */
function describeProgress(p: any): string | null {
  if (p?.status !== "progress" || !p.file) return null;
  const total = typeof p.total === "number" && p.total > 0 ? ` of ${(p.total / 1e6).toFixed(0)} MB` : "";
  return `downloading ${p.file} ${Math.round(Number(p.progress) || 0)}%${total}`;
}

/**
 * Point ONNX Runtime at this site's copies of its wasm. Transformers.js sets `wasmPaths` to cdn.jsdelivr.net
 * when it loads (`backends/onnx.js`) and chooses between two builds: asyncify, or plain for Safari below 26
 * with no WebGPU. The choice is kept and only the host changes.
 */
function useSameOriginOrtWasm(lib: any): void {
  const wasm = lib.env?.backends?.onnx?.wasm;
  if (!wasm) throw new Error("Transformers.js exposed no ONNX Runtime settings");
  const asyncify = String(wasm.wasmPaths?.wasm ?? "").includes("asyncify");
  const [mjs, file] = asyncify ? [ortAsyncifyMjs, ortAsyncifyWasm] : [ortPlainMjs, ortPlainWasm];
  wasm.wasmPaths = { mjs: new URL(mjs, location.href).href, wasm: new URL(file, location.href).href };
}

// The library's own fetch, kept once: a cancelled run can still be unwinding when the next one starts, so each run
// wraps this rather than whatever the last run left.
let libraryFetch: ((url: string, init?: RequestInit) => Promise<Response>) | null = null;

async function runTransformersJs(
  prompt: string, model: string, requested: Device, onProgress?: (t: string) => void, signal?: AbortSignal,
): Promise<AgentRun> {
  const t0 = performance.now();
  const lib: any = await import("@huggingface/transformers");
  signal?.throwIfAborted();
  if (lib.env) lib.env.allowLocalModels = false;          // weights from Hugging Face; browser-cached after
  useSameOriginOrtWasm(lib);
  // Cancel reaches a download through the library's fetch hook, and a generation through its stopping
  // criterion. Neither stops a graph already inside session creation; `runAgent` stops waiting for it.
  const stop = new lib.InterruptableStoppingCriteria();
  signal?.addEventListener("abort", () => stop.interrupt(), { once: true });
  libraryFetch ??= lib.env.fetch as (url: string, init?: RequestInit) => Promise<Response>;
  const baseFetch = libraryFetch;
  lib.env.fetch = (url: string, init?: RequestInit) => baseFetch(url, { ...init, signal });
  // The REQUESTED backend, then the genuine adapter check: WebGPU is used only when the visitor asked
  // for it AND a usable adapter exists. Asking for it on a machine without one runs WASM and says so.
  const device: Device = requested === "webgpu" && (await hasWebGPU()) ? "webgpu" : "wasm";
  const spec = TFJS_MODELS.find((m) => m.id === model);
  // `dtype` and `revision` from the config, not the library's defaults: the label's byte count is
  // computed from the config, so it describes the fetched file only if the fetch is pinned to it.
  const gen: any = await lib.pipeline("text-generation", model, {
    device, dtype: spec?.dtype[device], revision: spec?.revision,
    // Reported, so a first run shows its download moving - and each event also counts as progress
    // for the stall timeout in runAgent, which must not cut off a download that is still arriving.
    progress_callback: (p: any) => { const t = describeProgress(p); if (t) onProgress?.(t); },
  });
  try {
    onProgress?.(`generating on ${device}`);
    const out: any = await gen(prompt, { max_new_tokens: webCfg().tfjs_max_new_tokens, temperature: 0.7, do_sample: true,
      return_full_text: false, stopping_criteria: stop });
    const raw = Array.isArray(out) ? out[0]?.generated_text : out;
    const text = (typeof raw === "string" ? raw : JSON.stringify(raw ?? "")).trim();
    const notes = [
      requested === "webgpu" && device !== "webgpu"
        ? "WebGPU was selected, but this browser has no usable WebGPU adapter, so this ran on WASM." : "",
      text ? "" : "The model produced only whitespace on this run; it samples, so another run will differ.",
    ].filter(Boolean);
    return {
      engine: "transformers", text,
      elapsedMs: Math.round(performance.now() - t0), backend: `transformers-${device}`, model, webgpu: device === "webgpu",
      note: notes.length ? notes.join(" ") : undefined,
    };
  } finally {
    // A session holds its graph in the wasm heap until it is released, and each run opens one.
    await gen.dispose?.();
  }
}

/** Adapter feature probe. Every q4f16_1 MLC build needs `shader-f16`, which some Qualcomm adapters do not expose. */
export async function hasShaderF16(): Promise<boolean> {
  try {
    const a = await (navigator as any)?.gpu?.requestAdapter?.();
    return !!a?.features?.has?.("shader-f16");
  } catch {
    return false;
  }
}

/**
 * Ask the browser to make this origin's storage persistent before a large model download.
 *
 * A browser may evict non-persisted origin storage, which silently turns a one-time download into a
 * recurring one for a returning visitor. WebKit also deletes it after seven days of Safari use without
 * interaction with the site, and does not say a grant avoids that. Returns whether the grant was given so
 * the UI can say so rather than assume it.
 */
export async function requestPersistentStorage(): Promise<{ persisted: boolean; quotaMB: number | null }> {
  try {
    const persisted = (await navigator.storage?.persisted?.()) || (await navigator.storage?.persist?.()) || false;
    const est = await navigator.storage?.estimate?.();
    return { persisted, quotaMB: est?.quota ? Math.round(est.quota / 1e6) : null };
  } catch {
    return { persisted: false, quotaMB: null };
  }
}

let loadedWebLlm: { model: string; engine: any } | null = null;

/**
 * Load a WebLLM model, resuming after a lost request. The engine stops at the first shard that fails, and a long
 * download can lose one to nothing but the network (3 of 6 downloads on the machine measured, and the one captured was
 * Chrome's ERR_NETWORK_CHANGED, reaching the page as "Failed to fetch" or Cache.add's "network error"). Finished shards stay in the cache,
 * so another load fetches only what is missing. Anything else is reported at once.
 */
async function loadWebLlmResuming(
  engine: any, model: string, onProgress?: (t: string) => void, signal?: AbortSignal,
): Promise<void> {
  const attempts = webCfg().webllm_load_attempts;
  for (let attempt = 1; ; attempt++) {
    try {
      await engine.reload(model);
      return;
    } catch (e: any) {
      const message = String(e?.message ?? e);
      if (signal?.aborted || attempt >= attempts || !/failed to fetch|network error/i.test(message)) throw e;
      onProgress?.(`the download lost a request (${message.slice(0, 60)}); resuming from the cached files, `
        + `attempt ${attempt + 1} of ${attempts}`);
    }
  }
}

async function runWebLLM(
  prompt: string, model: string, onProgress?: (t: string) => void, signal?: AbortSignal,
): Promise<AgentRun> {
  const t0 = performance.now();
  if (!(await hasWebGPU())) {
    return { engine: "webllm", text: "", elapsedMs: 0, backend: "unavailable", model, webgpu: false,
      note: "WebLLM requires WebGPU (unavailable in this browser) — switch to Transformers.js (WASM)." };
  }
  // A q4f16_1 build on an adapter without shader-f16 fails deep inside the runtime with an opaque
  // shader-compilation error. Checking the feature first turns that into a sentence a reader can act on.
  const spec = WEBLLM_MODELS.find((m) => m.id === model);
  if (!spec) throw new Error(`${model} is not in the WebLLM list`);
  if (spec.requires === "shader-f16" && !(await hasShaderF16())) {
    return { engine: "webllm", text: "", elapsedMs: 0, backend: "unavailable", model, webgpu: true,
      note: `${model} is a q4f16_1 build and this GPU adapter does not expose the WebGPU shader-f16 `
        + `feature (absent on some Qualcomm adapters). Pick the q4f32_1 model instead.` };
  }
  {
    const { persisted, quotaMB } = await requestPersistentStorage();
    if (!persisted) {
      onProgress?.(`storage not persisted${quotaMB ? ` (quota ~${quotaMB} MB)` : ""} — `
        + "the browser may evict these weights, so a later visit re-downloads them");
    }
  }
  const webllm: any = await import("@mlc-ai/web-llm");
  signal?.throwIfAborted();
  // The whole record comes from the config, not the library's prebuilt list: the weights at a pinned revision
  // (the library would read `resolve/main`), and the model library from this site, built from the Apache-2.0
  // MLC-LLM source (`web/public/webllm/`), since the prebuilt libraries' repository has no licence file.
  const record = {
    model: `https://huggingface.co/mlc-ai/${spec.id}/resolve/${spec.revision}/`,
    model_id: spec.id,
    model_lib: new URL(`${import.meta.env.BASE_URL}${webCfg().webllm_lib_dir}/${spec.lib}`, location.href).href,
    required_features: spec.requires ? [spec.requires] : undefined,
    overrides: { context_window_size: spec.context_window_size },
    integrity: { model_lib: spec.lib_sri },
  };
  // One engine at a time. A loaded model holds its weights on the GPU until it is unloaded, and each run built a
  // new engine, so a second run held two. The same model is reused; another one replaces it.
  if (loadedWebLlm && loadedWebLlm.model !== model) {
    await loadedWebLlm.engine.unload();
    loadedWebLlm = null;
  }
  const engine = loadedWebLlm?.engine ?? new webllm.MLCEngine({
    appConfig: { model_list: [record] },
    initProgressCallback: (p: any) => onProgress?.(p?.text ?? ""),
  });
  // `interruptGenerate` stops an answer; `unload` aborts a load in progress, which is what stops a download.
  signal?.addEventListener("abort", () => {
    void engine.interruptGenerate?.();
    if (!loadedWebLlm) void engine.unload?.();
  }, { once: true });
  if (!loadedWebLlm) {
    await loadWebLlmResuming(engine, model, onProgress, signal);
    loadedWebLlm = { model, engine };
  }
  const reply: any = await engine.chat.completions.create({
    messages: [{ role: "system", content: SYSTEM }, { role: "user", content: prompt }],
    max_tokens: webCfg().webllm_max_tokens, temperature: 0.7,
  });
  return {
    engine: "webllm", text: reply?.choices?.[0]?.message?.content ?? "",
    elapsedMs: Math.round(performance.now() - t0), backend: "webllm-webgpu", model, webgpu: true,
  };
}

// Safety net (config): the run is abandoned after this long WITHOUT PROGRESS - no download event and no
// answer. Until Round 60 it was a limit on TOTAL time, which a first run's weights download could
// exhaust on its own: 313 MB took 277 s on one connection measured, so a run that was working was
// reported as timed out. Per the no-easy-fallback policy this is a last resort for a stuck load or a
// generation that cannot finish, not a mask for a tunable slowness.
const AGENT_TIMEOUT_MS = webCfg().agent_timeout_ms;

export async function runAgent(
  engine: Engine, prompt: string,
  opts?: { model?: string; device?: Device; onProgress?: (t: string) => void; signal?: AbortSignal },
): Promise<AgentRun> {
  const model = opts?.model ?? (engine === "webllm" ? WEBLLM_MODELS[0].id : TFJS_MODELS[0].id);
  const gpu = await hasWebGPU();                           // genuine adapter check, surfaced in the result
  const t0 = performance.now();
  let lastProgress = t0;
  const onProgress = (t: string) => { lastProgress = performance.now(); opts?.onProgress?.(t); };
  // A cancel and the stall timeout stop the engine the same way: an abandoned run must not go on downloading, or
  // finish loading after the next one has started.
  const stop = new AbortController();
  opts?.signal?.addEventListener("abort", () => stop.abort(), { once: true });
  const work = (async (): Promise<AgentRun> => {
    try {
      return engine === "webllm"
        ? await runWebLLM(prompt, model, onProgress, stop.signal)
        : await runTransformersJs(prompt, model, opts?.device ?? DEFAULT_DEVICE, onProgress, stop.signal);
    } catch (e: any) {
      // fail soft so the quantum-safe MCP demo always proceeds (no WebGPU / model fetch blocked / headless)
      return { engine, text: "", elapsedMs: 0, backend: "error", model, webgpu: gpu,
        note: `In-browser AI unavailable here (${String(e?.message || e).slice(0, 120)}); the quantum-safe MCP channel still runs.` };
    }
  })();
  // Race the run against a STALL timeout, so a stuck load or a generation that cannot finish never
  // hangs the page and the quantum-safe MCP envelope always renders.
  const timeout = new Promise<AgentRun>((resolve) => {
    const iv = setInterval(() => {
      if (performance.now() - lastProgress < AGENT_TIMEOUT_MS) return;
      clearInterval(iv);
      stop.abort();
      resolve({
        engine, text: "", elapsedMs: Math.round(performance.now() - t0), backend: "timeout", model, webgpu: gpu,
        // "try ... a WebGPU-capable browser" until Round 60: on the machine measured, WebGPU was the
        // backend that could not finish, so the advice pointed the wrong way. WebLLM has no WASM to
        // compare with, and the one stall measured on it was a download that stopped arriving.
        note: `The in-browser model made no progress for ${AGENT_TIMEOUT_MS / 1000} s and was stopped. `
          + (engine === "webllm"
            ? "Files already downloaded stay in the browser's cache, so another run continues from them. "
            : "On some machines the WebGPU backend is far slower than WASM. ")
          + "The quantum-safe MCP channel still runs.",
      });
    }, 1000);
    void work.finally(() => clearInterval(iv));
  });
  // Cancel answers at once, whatever the engine is doing, and the engine is told to stop in its own way.
  const cancelled = new Promise<AgentRun>((resolve) => {
    opts?.signal?.addEventListener("abort", () => resolve({
      engine, text: "", elapsedMs: Math.round(performance.now() - t0), backend: "cancelled", model, webgpu: gpu,
      note: "You cancelled this run. Files already downloaded stay in the browser's cache, so the next run "
        + "continues from them. The quantum-safe MCP channel still runs.",
    }), { once: true });
  });
  return Promise.race([work, timeout, cancelled]);
}

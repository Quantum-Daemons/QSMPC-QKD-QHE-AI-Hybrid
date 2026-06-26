// Dual in-browser, WebGPU-accelerated AI engines for the public demo (zero server). BOTH are loaded
// ONLY via dynamic import() so neither heavy library lands in the initial bundle — they become async
// chunks fetched on first use. The AI agent's output is then wrapped in the client quantum-safe MCP
// channel (clientMcp.ts), realising the report's "MCP + agentic AI with a quantum-safe layer" vision
// entirely client-side.
//   - Transformers.js (@huggingface/transformers, Apache-2.0): runs ONNX models via onnxruntime-web
//     (WebGPU, WASM fallback) — the same stack the demo already uses for private inference.
//   - WebLLM (@mlc-ai/web-llm, Apache-2.0): a WebGPU chat-LLM engine (small models); WebGPU-required.
// Everything fails soft: if WebGPU / model-fetch is unavailable, runAgent returns a notice so the
// quantum-safe MCP demo still proceeds (headless / CI-safe). Model weights stream from the HF/MLC CDN
// on first run and are browser-cached thereafter (zero VPS bandwidth/compute).

import { webCfg } from "./config";

export type Engine = "transformers" | "webllm";

export interface AgentRun {
  engine: Engine; text: string; elapsedMs: number;
  backend: string; model: string; webgpu: boolean; note?: string;
}

// model lists + generation limits + timeout come from the central config (config.ts ← /api/config ←
// config/defaults.py: WebConfig) — single source of truth, no inline literals here.
export const TFJS_MODELS = webCfg().tfjs_models;
export const WEBLLM_MODELS = webCfg().webllm_models;

// GENUINE WebGPU check: a GPU adapter must actually be obtainable (matches chrome://gpu /
// webgpureport.org) — not merely navigator.gpu being defined — so the demo never claims WebGPU when it
// would silently run on WASM (no-easy-fallback: report the real backend).
export async function hasWebGPU(): Promise<boolean> {
  try {
    const gpu = (navigator as any)?.gpu;
    if (!gpu?.requestAdapter) return false;
    return !!(await gpu.requestAdapter());
  } catch {
    return false;
  }
}

const SYSTEM = "You are a concise assistant for a quantum-safe AI orchestration demo.";

async function runTransformersJs(prompt: string, model: string): Promise<AgentRun> {
  const t0 = performance.now();
  const lib: any = await import("@huggingface/transformers");
  if (lib.env) lib.env.allowLocalModels = false;          // fetch from the HF CDN; browser-cached after
  const device = (await hasWebGPU()) ? "webgpu" : "wasm"; // prefer WebGPU; genuine adapter check
  const gen: any = await lib.pipeline("text-generation", model, { device });
  const out: any = await gen(prompt, { max_new_tokens: webCfg().tfjs_max_new_tokens, temperature: 0.7, do_sample: true, return_full_text: false });
  const raw = Array.isArray(out) ? out[0]?.generated_text : out;
  return {
    engine: "transformers", text: typeof raw === "string" ? raw : JSON.stringify(raw ?? ""),
    elapsedMs: Math.round(performance.now() - t0), backend: `transformers-${device}`, model, webgpu: device === "webgpu",
  };
}

async function runWebLLM(prompt: string, model: string, onProgress?: (t: string) => void): Promise<AgentRun> {
  const t0 = performance.now();
  if (!(await hasWebGPU())) {
    return { engine: "webllm", text: "", elapsedMs: 0, backend: "unavailable", model, webgpu: false,
      note: "WebLLM requires WebGPU (unavailable in this browser) — switch to Transformers.js (WASM)." };
  }
  const webllm: any = await import("@mlc-ai/web-llm");
  const engine = await webllm.CreateMLCEngine(model, { initProgressCallback: (p: any) => onProgress?.(p?.text ?? "") });
  const reply: any = await engine.chat.completions.create({
    messages: [{ role: "system", content: SYSTEM }, { role: "user", content: prompt }],
    max_tokens: webCfg().webllm_max_tokens, temperature: 0.7,
  });
  return {
    engine: "webllm", text: reply?.choices?.[0]?.message?.content ?? "",
    elapsedMs: Math.round(performance.now() - t0), backend: "webllm-webgpu", model, webgpu: true,
  };
}

// safety-net timeout (config): bounds a stuck WebGPU/model load while leaving room for a slow first-run
// (weights download + shader compile ~60-70s; browser-cached after). Per the no-easy-fallback policy this
// is a true last-resort for an unfixable-here hang, not a mask for a tunable slowness.
const AGENT_TIMEOUT_MS = webCfg().agent_timeout_ms;

export async function runAgent(
  engine: Engine, prompt: string, opts?: { model?: string; onProgress?: (t: string) => void },
): Promise<AgentRun> {
  const model = opts?.model ?? (engine === "webllm" ? WEBLLM_MODELS[0].id : TFJS_MODELS[0].id);
  const gpu = await hasWebGPU();                           // genuine adapter check, surfaced in the result
  const work = (async (): Promise<AgentRun> => {
    try {
      return engine === "webllm"
        ? await runWebLLM(prompt, model, opts?.onProgress)
        : await runTransformersJs(prompt, model);
    } catch (e: any) {
      // fail soft so the quantum-safe MCP demo always proceeds (no WebGPU / model fetch blocked / headless)
      return { engine, text: "", elapsedMs: 0, backend: "error", model, webgpu: gpu,
        note: `In-browser AI unavailable here (${String(e?.message || e).slice(0, 120)}); the quantum-safe MCP channel still runs.` };
    }
  })();
  // race the model run against a timeout so a slow first-load / stuck WebGPU shader never hangs the page;
  // the quantum-safe MCP envelope then always renders (perf-conscious demo).
  const timeout = new Promise<AgentRun>((resolve) => setTimeout(() => resolve({
    engine, text: "", elapsedMs: AGENT_TIMEOUT_MS, backend: "timeout", model, webgpu: gpu,
    note: "In-browser model is still loading / too slow in this browser — try a smaller model or a WebGPU-capable browser. The quantum-safe MCP channel still runs.",
  }), AGENT_TIMEOUT_MS));
  return Promise.race([work, timeout]);
}

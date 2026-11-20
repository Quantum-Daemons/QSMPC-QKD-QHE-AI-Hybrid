// This site's copies of ONNX Runtime's two wasm builds, for every caller that loads ONNX Runtime (Round 86).
//
// Transformers.js sets ONNX Runtime's `wasmPaths` to cdn.jsdelivr.net when its module runs (`backends/onnx.js`), and
// Vite puts onnxruntime-web in the same chunk as Transformers.js, so loading `onnxruntime-web/webgpu` for the private
// inference stage ran that code too. On 2026-11-20 a pipeline run with WebGPU selected fetched the asyncify build's
// .mjs and .wasm from jsDelivr: third-party code, where this site says it serves the engines' runtime files itself.
// The agent panel had been pointed at this site's copies earlier in the round; the inference stage, its twin, had not.
// The Content-Security-Policy added the same day refuses that host, so the run then fell back to WASM. Both callers now
// point ONNX Runtime here, through this one module.
//
// `?url` emits the file and imports no code; the wasm files are ones the build already ships. The asyncify build's
// .mjs is not named: ONNX Runtime's bundle holds it (see useSameOriginOrtBuild).
import ortAsyncifyWasm from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm?url";
import ortPlainMjs from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url";
import ortPlainWasm from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url";

/** ONNX Runtime's `env.wasm`, as much of it as this module touches. */
export interface OrtWasmEnv { wasmPaths?: unknown }

/** The build a setting names: asyncify (WebGPU, and Transformers.js's usual choice), or plain. */
export function ortBuildOf(wasm: OrtWasmEnv | undefined): "asyncify" | "plain" {
  return String((wasm?.wasmPaths as { wasm?: unknown } | undefined)?.wasm ?? "").includes("asyncify") ? "asyncify" : "plain";
}

/**
 * Point ONNX Runtime's wasm settings at this site's copy of `build`.
 *
 * For the asyncify build only the wasm file is named, and ONNX Runtime uses the module its bundle already holds
 * (`ort.webgpu.bundle.min.mjs` embeds that build's .mjs). Naming the .mjs too worked for the inference stage, but
 * Transformers.js fetches a named .mjs and imports it from a `blob:` URL (its `loadWasmFactory`), which the site's
 * Content-Security-Policy refuses: on 2026-11-20, the day the policy went live, the agent panel's Transformers.js
 * engine failed with "Failed to fetch dynamically imported module: blob:...". The plain build (Transformers.js's choice
 * for Safari below 26 with no WebGPU) is not the one the bundle holds, so its .mjs is still named; under the policy
 * that engine then reports the refusal there instead of running.
 */
export function useSameOriginOrtBuild(wasm: OrtWasmEnv | undefined, build: "asyncify" | "plain"): void {
  if (!wasm) throw new Error("no ONNX Runtime wasm settings to point at this site");
  wasm.wasmPaths = build === "asyncify"
    ? { wasm: new URL(ortAsyncifyWasm, location.href).href }
    : { mjs: new URL(ortPlainMjs, location.href).href, wasm: new URL(ortPlainWasm, location.href).href };
}

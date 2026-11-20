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
// `?url` emits the file and imports no code; the wasm files are ones the build already ships.
import ortAsyncifyMjs from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs?url";
import ortAsyncifyWasm from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm?url";
import ortPlainMjs from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url";
import ortPlainWasm from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url";

/** ONNX Runtime's `env.wasm`, as much of it as this module touches. */
export interface OrtWasmEnv { wasmPaths?: unknown }

/** The build a setting names: asyncify (WebGPU, and Transformers.js's usual choice), or plain. */
export function ortBuildOf(wasm: OrtWasmEnv | undefined): "asyncify" | "plain" {
  return String((wasm?.wasmPaths as { wasm?: unknown } | undefined)?.wasm ?? "").includes("asyncify") ? "asyncify" : "plain";
}

/** Point ONNX Runtime's wasm settings at this site's copy of `build`. */
export function useSameOriginOrtBuild(wasm: OrtWasmEnv | undefined, build: "asyncify" | "plain"): void {
  if (!wasm) throw new Error("no ONNX Runtime wasm settings to point at this site");
  const [mjs, file] = build === "asyncify" ? [ortAsyncifyMjs, ortAsyncifyWasm] : [ortPlainMjs, ortPlainWasm];
  wasm.wasmPaths = { mjs: new URL(mjs, location.href).href, wasm: new URL(file, location.href).href };
}

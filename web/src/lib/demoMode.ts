// Demo-mode detection. The public demo runs ALL compute client-side (browser JS/WASM/WebGPU) so the
// VPS is not a bottleneck; the full local deployment uses the FastAPI backend.
//
// A CLIENT_ONLY build is client-only, and `?demo=` cannot override that.
//
// It used to. An earlier revision honoured the query string BEFORE the build flag so that
// `?demo=full` could reach the slim API shipped alongside the public demo. Running it against the
// deployed site showed why that was wrong: the slim image is built without FHE/torch on purpose, so
// the pipeline died at `stage_error private_inference No module named 'tenseal'` and the run ended
// in State: error — while the page still displayed DEMO_BANNER's "No computation is sent to a
// server", which by then was false. It also let any anonymous visitor drive compute and write
// result files inside the container on a small shared VPS.
//
// So the escape hatch is gone from client-only bundles. `?demo=client` remains useful in a FULL
// build, where it lets a developer exercise the browser path without rebuilding.

const KEY = "qsh_demo_client";

export function isClientDemo(): boolean {
  const clientOnlyBuild = (import.meta as any).env?.VITE_CLIENT_ONLY === "1";
  // No override in a client-only bundle: there is no backend behind it that can serve a run.
  if (clientOnlyBuild) return true;
  if (typeof window === "undefined") return false;
  try {
    const q = new URLSearchParams(window.location.search).get("demo");
    if (q === "client") {
      sessionStorage.setItem(KEY, "1"); // persist across client-side route changes
      return true;
    }
    if (q === "full") {
      sessionStorage.setItem(KEY, "0"); // sticky, so the choice survives in-app navigation
      return false;
    }
    const stored = sessionStorage.getItem(KEY);
    if (stored !== null) return stored === "1";
    return false;
  } catch {
    return false;
  }
}

export const DEMO_BANNER =
  "Client-side compute — this page runs in your browser (WASM / WebGPU + @noble PQC). " +
  "No computation is sent to a server. Demo = inference only; the full local deployment " +
  "adds federated learning.";

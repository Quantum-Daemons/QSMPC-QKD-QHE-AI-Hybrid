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

// "Demo = inference only" stopped being true when Round 18 made federated learning run client-side
// on every medical run, and Round 19 removed the toggle that used to gate it. The banner said the
// demo could not do the thing the page below it was doing.
//
// The quantum-channel clause is here because this banner is the one text every page renders. The
// "simulated" label used to live at the bottom of Overview and Value, in a use-case page's last
// card and in hover text, while /orchestration and the security-posture table showed QKD figures
// with no label at all - and the posture table called them "measured".
export const DEMO_BANNER =
  "Client-side compute — this page runs in your browser (WASM / WebGPU + @noble PQC). " +
  "No computation is sent to a server. Inference, CKKS, garbled-circuit MPC and — on the medical " +
  "and finance pages — federated training all run here. The quantum channel (QKD, QSDC) is " +
  "simulated: no browser API for QKD exists. The full local deployment adds the server-side " +
  "backends and the real MCP server.";

// Demo-mode detection. The public demo runs ALL compute client-side (browser
// JS/WASM/WebGPU) so the VPS is not a bottleneck; the full local deployment uses the
// FastAPI backend. Enabled by `?demo=client` (persisted for the session) or the
// VITE_CLIENT_ONLY=1 build flag (baked into the static demo bundle).

const KEY = "qsh_demo_client";

export function isClientDemo(): boolean {
  // build-time flag for the published static bundle
  if ((import.meta as any).env?.VITE_CLIENT_ONLY === "1") return true;
  if (typeof window === "undefined") return false;
  try {
    const q = new URLSearchParams(window.location.search).get("demo");
    if (q === "client") {
      sessionStorage.setItem(KEY, "1"); // persist across client-side route changes
      return true;
    }
    if (q === "full") {
      sessionStorage.removeItem(KEY);
      return false;
    }
    return sessionStorage.getItem(KEY) === "1";
  } catch {
    return false;
  }
}

export const DEMO_BANNER =
  "Client-side compute — this page runs in your browser (WASM / WebGPU + @noble PQC). " +
  "No computation is sent to a server. Demo = inference only; the full local deployment " +
  "adds federated learning.";

// Demo-mode detection. The public demo runs ALL compute client-side (browser
// JS/WASM/WebGPU) so the VPS is not a bottleneck; the full local deployment uses the
// FastAPI backend. Enabled by `?demo=client` (persisted for the session) or the
// VITE_CLIENT_ONLY=1 build flag (baked into the static demo bundle).

const KEY = "qsh_demo_client";

export function isClientDemo(): boolean {
  const clientOnlyBuild = (import.meta as any).env?.VITE_CLIENT_ONLY === "1";
  if (typeof window === "undefined") return clientOnlyBuild;
  try {
    // An explicit ?demo= is honoured BEFORE the build flag. Testing the flag first meant
    // ?demo=full could never escape a CLIENT_ONLY bundle, contradicting both the
    // Dockerfile.api.demo comment and the README's "?demo=full on the VPS" fallback: the slim
    // backend shipped inside the demo image was unreachable from the UI deployed alongside it.
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
    return clientOnlyBuild;
  } catch {
    return clientOnlyBuild;
  }
}

export const DEMO_BANNER =
  "Client-side compute — this page runs in your browser (WASM / WebGPU + @noble PQC). " +
  "No computation is sent to a server. Demo = inference only; the full local deployment " +
  "adds federated learning.";

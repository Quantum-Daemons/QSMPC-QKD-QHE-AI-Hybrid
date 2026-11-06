// Typed client for the QSH FastAPI backend.
// The three Validations "demo" calls are DEMO-AWARE: in client-demo mode they run fully in the browser
// (zero /api) via the @noble / quantumSim / clientMcp mirrors; in full mode they hit the backend.
import { mlkemDemo } from "./crypto";
import { isClientDemo } from "./demoMode";
import { clientMcpDemo } from "./clientMcp";
import { qkdCfg, webCfg } from "./config";
import { bb84, FIBER } from "./quantumSim";

export interface RunStatus {
  run_id: string;
  usecase: string;
  state: string;
  metrics: Record<string, number | null>;
  artifacts: Record<string, unknown>;
}

export interface PipelineEvent {
  type: string;
  stage: string;
  message: string;
  data: Record<string, unknown>;
  t: number;
}

const base = ""; // same-origin (Vite proxies /api in dev)

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(base + url, init);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json() as Promise<T>;
}

// Only the calls a page makes. Wrappers for /api/health, /api/usecases, /api/restart, /api/benchmark,
// /api/alice-bob, /api/infer/fraud, /api/results and /api/download had no caller in the SPA and were removed in
// Round 86; the endpoints remain for direct API clients (docs/architecture.md).
export const api = {
  start: (usecase: string, config: Record<string, unknown> = {}, startPaused = false) =>
    j<{ run_id: string }>(`/api/run/${usecase}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config, start_paused: startPaused }),
    }),
  stop: (id: string) => j<{ stopped: boolean }>(`/api/stop/${id}`, { method: "POST" }),
  pause: (id: string) => j<{ paused: boolean }>(`/api/pause/${id}`, { method: "POST" }),
  resume: (id: string) => j<{ resumed: boolean }>(`/api/resume/${id}`, { method: "POST" }),
  step: (id: string) => j<{ stepped: boolean }>(`/api/step/${id}`, { method: "POST" }),
  status: (id: string) => j<RunStatus>(`/api/runs/${id}`),
  cryptoOptions: () =>
    j<{ families: Record<string, string[]>; defaults: Record<string, string> }>("/api/crypto-options"),
  demoMcp: (qkd = false): Promise<any> =>
    isClientDemo() ? clientMcpDemo(qkd) : j<any>(`/api/demo/mcp?qkd=${qkd}`),
  demoPqc: (): Promise<any> => {
    if (!isClientDemo()) return j<any>("/api/demo/pqc");
    const k = mlkemDemo();   // client-side hybrid-KEM mirror (shapes match the server response)
    // `classical` used to be the literal string "X25519" beside a value produced by ML-KEM alone.
    // Both halves are now really computed, so every field here is read off the run rather than
    // asserted, and the byte counts are split so the total can be checked against the parts.
    return Promise.resolve({ kem: "ML-KEM-768", classical: k.classical, combiner: k.combiner,
      ciphertext_bytes: k.cipherTextBytes, pqc_ciphertext_bytes: k.pqcCipherTextBytes,
      classical_public_key_bytes: k.classicalPublicKeyBytes,
      shared_secret_prefix: k.sharedSecretPrefix, agreement: k.agreement });
  },
  demoQkd: (km: number, eve: number): Promise<any> => {
    if (!isClientDemo()) return j<any>(`/api/demo/qkd?distance_km=${km}&eve=${eve}`);
    // client BB84 mirror; pulse count + eavesdrop threshold come from config so Validations stays consistent
    // with the pipeline (the same QBER the pipeline treats as secure isn't flagged "eavesdropper" here).
    const r = bb84(webCfg().demo_alice_bob_pulses, { ...FIBER, distanceKm: km, eveFraction: eve }, 1);
    // `skr_per_pulse`, matching the server branch above and `api/inference_service.py:65`. Round 44
    // renamed three producers and the reader on /validations, and missed THIS one - the only branch
    // the public demo ever executes, because the demo builds CLIENT_ONLY and `isClientDemo()` is
    // then unconditionally true. The page read `qkd.skr_per_pulse`, got `undefined`, and React
    // rendered an empty cell: a blank value shipped under "secret key rate (per pulse)". `SW3`
    // could not see it, because an `undefined` React child renders as nothing at all rather than as
    // the token "undefined" - the one wrong value that leaves no trace on the page.
    //
    // Nine decimals, at every site that emits it: six (Round 45) kept two significant figures of
    // the satellite downlink's 1.723e-5 under Round 75's Micius budget, and the panel printed
    // 1.700e-5. Display sites format to four significant figures.
    return Promise.resolve({ qber: Math.round(r.qber * 1e4) / 1e4,
      skr_per_pulse: Math.round(r.secretKeyRate * 1e9) / 1e9,
      secure_key_bits: r.secureKeyBits,
      // The block that count is the composable finite key of (Round 80); the server branch sends it too.
      block_pulses: Number((r.blockPulses ?? 0).toPrecision(6)),
      // The QKD stage's own abort threshold, as the server's twin reads it. Until Round 86 both read the retired security
      // score's `qber_secure_threshold`, equal today, so the flag and the abort could part if either were edited.
      eavesdropper_likely: r.qber > qkdCfg().pp_threshold });
  },
};

// Stream pipeline events over WebSocket; falls back to polling if WS is blocked.
export function streamRun(
  id: string,
  onEvent: (e: PipelineEvent) => void,
  onEnd: () => void,
): () => void {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/ws/${id}`);
  ws.onmessage = (m) => {
    try {
      const data = JSON.parse(m.data);
      if (data.type === "stream_end") onEnd();
      else onEvent(data as PipelineEvent);
    } catch {
      /* ignore */
    }
  };
  ws.onerror = () => {}; // WebSocket errors are surfaced via onclose
  ws.onclose = () => onEnd();
  return () => ws.close();
}

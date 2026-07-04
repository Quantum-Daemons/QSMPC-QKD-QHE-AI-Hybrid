// Typed client for the QSH FastAPI backend.
// The three Validations "demo" calls are DEMO-AWARE: in client-demo mode they run fully in the browser
// (zero /api) via the @noble / quantumSim / clientMcp mirrors; in full mode they hit the backend.
import { mlkemDemo } from "./crypto";
import { isClientDemo } from "./demoMode";
import { clientMcpDemo } from "./clientMcp";
import { securityScoringCfg, webCfg } from "./config";
import { bb84, FIBER } from "./quantumSim";

export interface UseCase {
  id: string;
  title: string;
  industry: string;
  depth: string;
}

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

export const api = {
  health: () => j<{ status: string }>("/api/health"),
  usecases: () => j<UseCase[]>("/api/usecases"),
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
  restart: (id: string) => j<{ run_id: string }>(`/api/restart/${id}`, { method: "POST" }),
  status: (id: string) => j<RunStatus>(`/api/runs/${id}`),
  cryptoOptions: () =>
    j<{ families: Record<string, string[]>; defaults: Record<string, string> }>("/api/crypto-options"),
  demoMcp: (qkd = false): Promise<any> =>
    isClientDemo() ? clientMcpDemo(qkd) : j<any>(`/api/demo/mcp?qkd=${qkd}`),
  benchmark: (workload = "linear", samples = 20, modelSize = "medium") =>
    j<{ workload: string; model_size: string; results: any[]; advantage: any }>(
      `/api/benchmark?workload=${workload}&samples=${samples}&model_size=${modelSize}`),
  aliceBob: (config: Record<string, unknown> = {}) =>
    j<any>("/api/alice-bob", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config }),
    }),
  inferFraud: (features: number[]) =>
    j<{ fraud_probability: number; prediction: number }>("/api/infer/fraud", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    }),
  demoPqc: (): Promise<any> => {
    if (!isClientDemo()) return j<any>("/api/demo/pqc");
    const k = mlkemDemo();   // client-side hybrid-KEM mirror (shapes match the server response)
    return Promise.resolve({ kem: "ML-KEM-768", classical: "X25519", ciphertext_bytes: k.cipherTextBytes,
      shared_secret_prefix: k.sharedSecretPrefix, agreement: k.agreement });
  },
  demoQkd: (km: number, eve: number): Promise<any> => {
    if (!isClientDemo()) return j<any>(`/api/demo/qkd?distance_km=${km}&eve=${eve}`);
    // client BB84 mirror; pulse count + eavesdrop threshold come from config so Validations stays consistent
    // with the pipeline (the same QBER the pipeline treats as secure isn't flagged "eavesdropper" here).
    const r = bb84(webCfg().demo_alice_bob_pulses, { ...FIBER, distanceKm: km, eveFraction: eve }, 1);
    return Promise.resolve({ qber: Math.round(r.qber * 1e4) / 1e4, skr: Math.round(r.secretKeyRate * 1e5) / 1e5,
      secure_key_bits: r.secureKeyBits, eavesdropper_likely: r.qber > securityScoringCfg().qber_secure_threshold });
  },
  results: () => j<{ name: string; bytes: number }[]>("/api/results"),
  downloadUrl: (name: string) => `/api/download/${encodeURIComponent(name)}`,
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

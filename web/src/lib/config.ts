// Single source of truth for the SPA's cost model / tier architectures / workload multipliers / the
// medical-CNN shape / security + advantage weights / demo data sizes / in-browser-AI knobs.
//
// The values are GENERATED from src/qsh/config/defaults.py by scripts/export_config.py
// (config.defaults.json) — so the frontend never hand-duplicates the backend's numbers (this kills the
// FE<->BE cost-model drift the audit flagged). In full mode the app refreshes from /api/config so a
// runtime config.yaml override flows through; the demo keeps the baked JSON (zero /api).
import { isClientDemo } from "./demoMode";
import defaults from "./config.defaults.json";

export type AppConfig = typeof defaults;

let CONFIG: AppConfig = defaults;

// Non-blocking refresh from the backend (full mode only). Callers read the baked values immediately —
// they match the JSON; the refresh only matters when a config.yaml overrides a default.
export async function refreshConfigFromApi(): Promise<void> {
  if (isClientDemo()) return;
  try {
    const r = await fetch("/api/config");
    if (r.ok) CONFIG = { ...defaults, ...(await r.json()) };
  } catch {
    /* keep the baked defaults */
  }
}

export const appConfig = (): AppConfig => CONFIG;

// Typed accessors used by the demo cost model (clientCompute) + the in-browser AI (clientAI).
export const benchCfg = () => CONFIG.benchmark;
export const costPerOp = () => CONFIG.benchmark.cost_per_op;
export const tierArchitectures = () => CONFIG.benchmark.tier_architectures as Record<string, { label: string; hidden: Record<string, number[]> }>;
export const genericArchitecture = () => CONFIG.benchmark.generic_architecture as Record<string, number[]>;
export const workloadMultipliers = () => CONFIG.benchmark.workload_multipliers as unknown as Record<string, [number, string]>;
export const medicalCnn = () => CONFIG.benchmark.medical_cnn;
export const demoFeatureDim = () => CONFIG.benchmark.demo_feature_dim;
export const securityScoringCfg = () => CONFIG.security_scoring;
export const quantumAdvantageCfg = () => CONFIG.quantum_advantage;
export const dataCfg = () => CONFIG.data;
export const qkdCfg = () => CONFIG.qkd;
export const qrngCfg = () => CONFIG.qrng;
export const idsCfg = () => CONFIG.ids;
export const cryptoCfg = () => CONFIG.crypto;   // crypto-agility defaults incl. hybrid_combiner_mode
export const webCfg = () => CONFIG.web;

/** Provenance of the sample bundled INTO the demo (not the corpus the model was trained on).
 *  Generated from src/qsh/models/registry.py BUNDLED_DEMO_SAMPLES; CI checks the parity. */
export type DemoSample = { provenance: string; licence: string; url: string; licence_checked: string };
export const demoSample = (usecase: string): DemoSample | null =>
  ((CONFIG as any).demo_samples ?? {})[usecase] ?? null;

// --- model registry ---------------------------------------------------------
// Generated from src/qsh/models/registry.py. Repo ids, pinned revisions, artefact paths, dataset
// licences, published baselines and measured metrics all come from here so that no model name,
// URL or number is written into a component.

export type ModelArtifact = {
  path: string;
  fmt: "onnx" | "safetensors" | "json";
  precision: "fp32" | "int8" | "none";
  serves: "browser" | "server" | "both";
};

export type ModelSpec = {
  key: string;
  // t3_large is part of the documented tier taxonomy and no ModelSpec uses it. Kept rather
  // than deleted because the in-browser LLM (Transformers.js / WebLLM) IS a T3-shaped model -
  // too big for encrypted evaluation, run locally instead - it simply is not a registry entry,
  // since the registry describes models this project trains and publishes. If that ever
  // changes the slot is here; until then it is documented-empty rather than silently empty.
  tier: "t1_crypto" | "t2_industry" | "t3_large";
  label: string;
  architecture: string;
  params: number | null;
  runs_encrypted: boolean;
  hf_repo_id: string | null;
  hf_revision: string | null;
  /** The sha that WAS pinned before a retrain superseded the published artefact. Distinguishes
   *  "retrained since publication, needs republishing" from "never pinned", which look identical
   *  from `hf_revision` alone and are fixed by completely different actions. */
  hf_revision_stale: string | null;
  artifacts: ModelArtifact[];
  dataset: { name: string; licence: string; url: string; publishable: boolean };
  metric: {
    name: string; display: string;
    baseline: number | null; baseline_source: string;
    ceiling: number | null; ceiling_source: string;
    companion: string;
  };
  /** Measured by a training run, or null when the model has never been trained. Never a placeholder. */
  achieved: Record<string, unknown> | null;
  notes: string;
};

/** QSMPC field parameters. `prime: null` means the backend derives it from n as Sutradhar's
 *  construction requires (d prime, n <= d <= 2n). */
export const qsmpcCfg = () => ((CONFIG as any).qsmpc ?? {}) as
  { prime: number | null; n_parties: number; threshold: number };

/** Federated hyperparameters. `browser_*` are the deliberate lighter overrides the in-tab FedAvg
 *  uses so a round finishes inside a page interaction; the un-prefixed values are what the full
 *  server-side deployment trains with. Both are exported so the difference is visible rather than
 *  being two unrelated literals in two languages. */
export const federatedCfg = () => ((CONFIG as any).federated ?? {}) as {
  rounds: number; local_epochs: number; lr: number; hospitals: number; test_split: number;
  secure_aggregation: boolean; browser_rounds: number; browser_lr: number;
  browser_override_reason: string;
};

export const modelsFor = (usecase: string): ModelSpec[] =>
  (((CONFIG as any).models ?? {})[usecase] ?? []) as ModelSpec[];

export const allModels = (): Record<string, ModelSpec[]> => ((CONFIG as any).models ?? {});

/** The industry (plaintext) model for a use case — what the headline metric describes. */
export const teacherModel = (usecase: string): ModelSpec | null =>
  modelsFor(usecase).find((m) => m.tier === "t2_industry" && m.achieved) ??
  modelsFor(usecase).find((m) => m.tier === "t2_industry") ?? null;

/** The encrypted (FHE/MPC-feasible) model for a use case. */
export const studentModel = (usecase: string): ModelSpec | null =>
  modelsFor(usecase).find((m) => m.tier === "t1_crypto") ?? null;

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
  tier: "t1_crypto" | "t2_industry" | "t3_large";
  label: string;
  architecture: string;
  params: number | null;
  runs_encrypted: boolean;
  hf_repo_id: string | null;
  hf_revision: string | null;
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

// Murmur detection under differential privacy (Round 86): web/public/data/medical/murmur_dp.json, which
// scripts/data/build_murmur_dp_summary.py writes from the study's runs, every tuning route side by side. Aggregate figures only.

export interface Stat2 { mean: number; sd: number | null }
export interface AtThreshold { balanced_accuracy: Stat2; share_predicted_positive: Stat2 }

export interface MurmurRow {
  model: string;
  arm: string;
  kind: string;
  epsilon: number | null;
  seeds?: number;
  searches?: number;
  auroc: Stat2;
  lr?: number;
  at_thresholds?: Record<string, AtThreshold>;
  sigma?: number;
  noise_sd_over_gamma_c?: number;
  epsilon_accounted?: number | null;
}

interface Credit { citation: string; licence: string; licence_url?: string; download?: string; dataset?: string; model?: string }

export interface MurmurDp {
  attribution: { circor: Credit & { provider: string }; physionet_2016: Credit; panns: Credit & { use: string }; platform_citation: string };
  data: { records_labelled: number; people: number; people_with_two_records: number; train_people: number; test_people: number;
    test_records: number; test_records_present: number };
  recipe: { delta: number; clip: number; expected_batch: number; epochs: number; sites: number; params: Record<string, number> };
  routes: { search?: { eta: number; expected_k: number }; objpert?: { lambda: number; tau: number; sigma_out: number } };
  summary: Record<string, MurmurRow[]>;
  environment: Record<string, string | null>;
}

export const MURMUR_EPS = [8, 4, 2, 1] as const;
export const MURMUR_ARMS: [string, string][] = [
  ["central", "Central DP-SGD, people sampled"],
  ["central-full", "Central, every person every step"],
  ["federated", "Four sites, distributed discrete Gaussian"],
];
export const MURMUR_ROUTES: [string, string][] = [
  ["public", "Rates chosen on a PhysioNet 2016 stand-in (the table above)"],
  ["circor", "Rates chosen on CirCor validation people: epsilon leaves the choice out"],
  ["naive", "Stand-in from the feature CNN's own databases"],
  ["search", "Accounted random search over three public-chosen rates"],
  ["objpert", "Objective perturbation, no learning rate"],
];

/** One cell of a route's summary. */
export function pick(rows: MurmurRow[] | undefined, model: string, arm: string, kind: string, eps: number | null) {
  return rows?.find((r) => r.model === model && r.arm === arm && r.kind === kind && r.epsilon === eps);
}

/** Every summarised cell, one CSV row each. */
export function murmurDpRows(c: MurmurDp): Record<string, unknown>[] {
  return Object.entries(c.summary).flatMap(([route, rows]) => rows.map((r) => ({
    route, model: r.model, arm: r.arm, kind: r.kind, epsilon: r.epsilon, auroc: r.auroc.mean, auroc_sd: r.auroc.sd,
    balanced_accuracy_public_threshold: r.at_thresholds?.balanced?.balanced_accuracy.mean ?? null,
    balanced_accuracy_logit_0: r.at_thresholds?.zero?.balanced_accuracy.mean ?? null,
    share_positive_logit_0: r.at_thresholds?.zero?.share_predicted_positive.mean ?? null, learning_rate: r.lr ?? null,
  })));
}

// The heart-sound federation's recorded figures (Round 86): web/public/data/medical/pcg_fl.json, which
// scripts/data/build_pcg_fl_summary.py writes from scripts/train/medical_pcg_fl.py's runs. Aggregate figures only.
import type { Stat } from "./cxrFl";

export interface PcgArm {
  seeds: number;
  auroc: Stat;
  interval_first_seed: [number, number];
  auroc_good_quality: Stat;
  macc: Stat;
  sensitivity: Stat;
  specificity: Stat;
}

export interface PcgSource { recordings: number; abnormal: number; good_quality: number; hours: number }

export const PCG_SOURCES = ["a", "b", "c", "d", "e", "f"] as const;
export const PCG_RANDOM = "random 20% of recordings";

export interface PcgFl {
  attribution: { provider: string; dataset: string; download: string; citation: string; platform_citation: string; licence: string; licence_url: string };
  sources: Record<string, PcgSource>;
  training_recording_counts: { released_zip: number; physionet_page_introduction: number; physionet_page_announcement: number; cinc_2016_paper_table_1: number };
  recipe: { epochs: number; batch: number; lr: number; window_s: number; params: number; seeds: number[] };
  summary: Record<string, { pooled?: PcgArm; fedavg?: PcgArm }>;
  environment: Record<string, string | null>;
}

/** One row per held-out source, as the card's table draws it, for its CSV. */
export function pcgFlRows(c: PcgFl): Record<string, unknown>[] {
  return PCG_SOURCES.filter((s) => c.summary[s]).map((s) => {
    const r = c.summary[s];
    return {
      held_out: s, recordings: c.sources[s].recordings, abnormal_share: c.sources[s].abnormal / c.sources[s].recordings,
      pooled_auroc: r.pooled?.auroc.mean ?? null, pooled_auroc_sd: r.pooled?.auroc.sd ?? null, fedavg_auroc: r.fedavg?.auroc.mean ?? null,
      fedavg_auroc_sd: r.fedavg?.auroc.sd ?? null, pooled_macc: r.pooled?.macc.mean ?? null, fedavg_macc: r.fedavg?.macc.mean ?? null,
    };
  });
}

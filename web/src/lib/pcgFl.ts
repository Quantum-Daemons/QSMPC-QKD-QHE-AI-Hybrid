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
  /** The same study with each mel band's mean removed over the window (`--band-mean`). */
  band_mean?: { recipe: { seeds: number[] }; summary: Record<string, { pooled?: PcgArm; fedavg?: PcgArm }> };
  environment: Record<string, string | null>;
}

/** Mean over the held-out sources of one arm's mean AUROC before and after, and the sources whose mean rose or fell by
 * more than `within` (the rest moved less than that, inside the seeds' spread). */
export function heldOutShift(base: PcgFl["summary"], other: PcgFl["summary"], arm: "pooled" | "fedavg", within = 0.01) {
  const held = PCG_SOURCES.filter((s) => base[s]?.[arm] && other[s]?.[arm]);
  const mean = (sm: PcgFl["summary"]) => held.reduce((a, s) => a + (sm[s][arm]?.auroc.mean ?? 0), 0) / held.length;
  const delta = (s: string) => (other[s][arm]?.auroc.mean ?? 0) - (base[s][arm]?.auroc.mean ?? 0);
  return {
    before: mean(base), after: mean(other), within,
    rose: held.filter((s) => delta(s) > within), fell: held.filter((s) => delta(s) < -within),
    same: held.filter((s) => Math.abs(delta(s)) <= within),
  };
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

// The chest X-ray federation's recorded figures (Round 86): web/public/data/medical/cxr_fl.json, which
// scripts/data/build_cxr_fl_summary.py writes from scripts/train/medical_cxr_fl.py's runs. Aggregate figures only.

export interface Stat { mean: number; sd: number | null; per_seed: number[] }

export interface CamStats { n: number; pointing_hit_rate: Stat; pointing_hit_rate_15px: Stat; mass_in_box: Stat; box_area_share: Stat }

export interface CxrArm {
  seeds: number;
  auroc_overall: Stat; auroc_PA: Stat; auroc_AP: Stat;
  interval_overall_first_seed: [number, number];
  specificity_overall: Stat; specificity_PA: Stat; specificity_AP: Stat;
  sensitivity_overall: Stat; sensitivity_PA: Stat; sensitivity_AP: Stat;
  score_view_auroc_negatives: Stat; score_view_auroc_positives: Stat;
  view_probe_auroc?: Stat;
  cam?: { all: CamStats; PA: CamStats; AP: CamStats };
  params?: number;
  train_s: Stat;
}

export interface CxrHospital { hospital: number; ap_target: number; patients: number; train_images: number; val_images: number; ap_share: number; effusion_prevalence: number }

export interface CxrFl {
  attribution: { provider: string; dataset: string; download: string; citation: string; terms: string };
  data: {
    test: { images: number; patients: number; ap_share: number; prevalence_PA: number; prevalence_AP: number };
    hospitals: CxrHospital[];
    dropped_duplicates: number;
    view_only_auroc: number;
    boxes: { effusion_boxes: number; all_in_test_split: boolean; centre_baseline: { all: { pointing_hit_rate: number; pointing_hit_rate_15px: number; box_area_share: number } } };
  };
  recipe: { epochs: number; batch: number; lr: number; seeds: number[] };
  untrained_view_probe_auroc: { view_probe_auroc: Stat; cam: { all: CamStats } };
  summary: Record<string, CxrArm>;
  environment: Record<string, string | null>;
}

/** The arms in the order the card lists them. */
export const CXR_ARMS = [
  "centralised ResNet-18", "FedAvg ResNet-18",
  "hospital 1 alone, ResNet-18", "hospital 2 alone, ResNet-18", "hospital 3 alone, ResNet-18", "hospital 4 alone, ResNet-18",
  "PA images only, ResNet-18", "FedAvg ConvNeXt-atto",
] as const;

/** One row per arm, as the card's table draws it, for its CSV. */
export function cxrFlRows(c: CxrFl): Record<string, unknown>[] {
  return CXR_ARMS.filter((a) => c.summary[a]).map((a) => {
    const s = c.summary[a];
    return {
      arm: a, seeds: s.seeds, auroc: s.auroc_overall.mean, auroc_sd: s.auroc_overall.sd, auroc_PA: s.auroc_PA.mean, auroc_AP: s.auroc_AP.mean,
      specificity_PA_at_sens_0_80: s.specificity_PA.mean, specificity_AP_at_sens_0_80: s.specificity_AP.mean,
      score_view_auroc_negatives: s.score_view_auroc_negatives.mean, score_view_auroc_positives: s.score_view_auroc_positives.mean,
      cam_pointing_hit_rate: s.cam?.all.pointing_hit_rate.mean ?? null,
    };
  });
}

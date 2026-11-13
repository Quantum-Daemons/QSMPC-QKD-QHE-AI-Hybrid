/**
 * Each page's published figures beside this project's offline measurement and this run (Round 86, the owner's choice
 * of 2026-11-13: a table on every page, and where a row cannot be compared, the reason).
 *
 * A VERDICT IS ONE OF FOUR, AND THE RULE FOR EACH IS HERE, NOT IN THE MARKUP.
 *   reproduced      the same method on the same data, and the difference is inside a spread that was published or
 *                   measured: two standard deviations of the source's seeds for a single run here, one standard
 *                   deviation of the project's own seeds for its offline figure.
 *   differs         the same method on the same data, and the difference is outside that spread.
 *   no spread       the same method on the same data, and neither side published a spread, so the difference is
 *                   shown and no verdict is drawn from it.
 *   reference       a different method, model or protocol on the same data: a figure to read beside, never one this
 *                   project claims to reproduce. Which protocol differs is the registry's own baseline_source text.
 *   not comparable  the figures do not measure the same thing, with the reason.
 * "This run" is what this tab computed: the re-scored evaluation sample of each published model (clientCompute's
 * `verifications`), whose outcome is against the figure recorded with that sample, and on the medical page the
 * four-hospital federation itself, which is FLamby's benchmark run here.
 */
import type { ModelSpec } from "./config";
import type { VerificationSummary } from "./clientCompute";
import type { FedHeartResult } from "./clientFedHeart";

export type Verdict = "reproduced" | "differs" | "no spread" | "reference" | "not comparable";

export interface ComparisonRow {
  figure: string;
  source: string;
  published: string;
  offline: string;
  thisRun: string;
  verdict: Verdict;
  why: string;
}

type Relation = { relation: "same" | "reference" | "citation"; why: string };

/** How each registry model's published baseline relates to the model, read off the registry's baseline_source. */
export const MODEL_RELATION: Record<string, Relation> = {
  "finance_aml.gnn": { relation: "same", why: "Multi-PNA+EU itself, the architecture of the paper's figure" },
  "finance_aml.gbm": { relation: "reference", why: "same metric and corpus, a different protocol: the published figure splits by transaction order and updates IBM's Graph Feature Preprocessor over the test window" },
  "finance_aml.student": { relation: "reference", why: "a logistic regression distilled from the gradient-boosting model, beside the published GFP+XGBoost" },
  "finance_aml.mlp_large": { relation: "reference", why: "an MLP over this project's features, beside the published GFP+XGBoost" },
  "finance_aml.gnn_compact": { relation: "reference", why: "a compact message-passing model distilled from Multi-PNA+EU, beside Multi-PNA+EU itself" },
  "genomics.prs": { relation: "citation", why: "the published AUROC is for a model fitted with covariates on UK Biobank, with 162 of the score's 187 variants; this page scores 30 public individuals, so it is a citation and not a target" },
  "medical_fl.resnet18": { relation: "same", why: "ResNet-18 at 28 px on BreastMNIST's official split, as in MedMNIST v2" },
  "medical_fl.resnet50": { relation: "reference", why: "ResNet-50 at 28 px under a training regime that differs from the benchmark table's" },
  "medical_fl.student": { relation: "citation", why: "a single-convolution CNN reading 14 x 14 images, distilled for CKKS, beside the benchmark's ResNet-18 at 28 x 28" },
  "satellite_comms.telemanom": { relation: "same", why: "Telemanom on ESA-ADB's Mission1 lightweight channels 41 to 46, anomalies alone: the like-for-like row of the benchmark" },
  "satellite_comms.student": { relation: "reference", why: "a linear forecaster distilled from Telemanom, beside Telemanom's benchmark row" },
  "satellite_comms.opssat": { relation: "reference", why: "gradient-boosted trees over the published features, beside the FCNN, the best of 30 algorithms benchmarked" },
  "satellite_comms.opssat_student": { relation: "reference", why: "a logistic regression over a quadratic expansion, beside the benchmark's FCNN" },
};

const f4 = (x: number) => x.toFixed(4);
const signed = (x: number) => (x >= 0 ? `+${x.toFixed(4)}` : x.toFixed(4));

function thisRunOf(v: VerificationSummary | undefined): string {
  if (!v) return "not run in this tab";
  if (v.outcome === "PASS" || v.outcome === "FAIL") {
    return `${v.outcome}: ${f4(v.score!)} on ${v.n} rows of its sample (${v.split ?? "split not declared"}), recorded ${f4(v.expected!)}`;
  }
  if (v.outcome === "no-sample") return `no evaluation sample${v.noSampleReason ? `: ${v.noSampleReason}` : ""}`;
  return `could not run: ${v.error ?? "unknown"}`;
}

/** One row per registry model on the page. */
export function modelRows(models: ModelSpec[], verifications: VerificationSummary[] | undefined): ComparisonRow[] {
  return models.map((m) => {
    const published = m.metric?.baseline;
    const name = m.metric?.name ?? "";
    const offlineValue = typeof m.achieved?.[name] === "number" ? (m.achieved![name] as number) : null;
    const rel = MODEL_RELATION[m.key];
    const v = verifications?.find((x) => x.modelKey === m.key);
    const base = {
      figure: `${m.key}: ${m.metric?.display ?? name}`,
      source: m.metric?.baseline_source ? m.metric.baseline_source.split(". ")[0] : "-",
      published: published == null ? "none" : f4(published),
      offline: offlineValue == null ? "not measured here" : f4(offlineValue),
      thisRun: thisRunOf(v),
    };
    if (published == null) {
      return { ...base, verdict: "not comparable" as Verdict, why: m.metric?.baseline_source || "no published figure" };
    }
    if (!rel || rel.relation === "citation") {
      return { ...base, verdict: "not comparable" as Verdict, why: rel?.why ?? "no relation recorded for this model" };
    }
    if (rel.relation === "reference") {
      const d = offlineValue == null ? "" : ` (${signed(offlineValue - published)} against it)`;
      return { ...base, verdict: "reference" as Verdict, why: `${rel.why}${d}` };
    }
    if (offlineValue == null) return { ...base, verdict: "not comparable" as Verdict, why: `${rel.why}; not measured here` };
    const spread = (m.achieved as Record<string, any> | null)?.seed_spread as { mean: number; sd: number; n_seeds: number } | undefined;
    if (spread) {
      const inside = Math.abs(published - spread.mean) <= spread.sd;
      return { ...base, verdict: (inside ? "reproduced" : "differs") as Verdict,
               why: `${rel.why}; the published figure is ${inside ? "inside" : "outside"} one standard deviation of this project's `
                 + `${spread.n_seeds} seeds (${f4(spread.mean)} +/- ${f4(spread.sd)}), the figure shown being one of them` };
    }
    return { ...base, verdict: "no spread" as Verdict,
             why: `${rel.why}; ${signed(offlineValue - published)} against it, and no spread to judge that by` };
  });
}

/**
 * FLamby's Fed-Heart-Disease (du Terrail et al., NeurIPS 2022 Datasets and Benchmarks), as this project averaged its
 * five published result files (seeds 42 to 46) over the four client test sets, and this project's five seeds offline
 * (docs/verification-log.md, the Fed-Heart-Disease reproduction table; references/medical-fl-prior-work.md).
 */
export const FLAMBY_HEART = {
  source: "FLamby Fed-Heart-Disease, du Terrail et al., NeurIPS 2022 (five result files, seeds 42 to 46)",
  alone: [0.722, 0.629, 0.567, 0.637], fedavg: { mean: 0.685, sd: 0.028 }, fedyogi: { mean: 0.731, sd: 0.013 }, pooled: 0.735,
  offline: { alone: [0.735, 0.636, 0.601, 0.644], fedavg: { mean: 0.712, sd: 0.017 }, pooled: 0.715 },
};

export function flambyRows(r: FedHeartResult): ComparisonRow[] {
  const F = FLAMBY_HEART;
  const noSpread = (pub: number, run: number) =>
    `FLamby's spread for this figure is not recorded here, so the difference (${signed(run - pub)}) is shown without a verdict`;
  const rows: ComparisonRow[] = r.centres.map((c, k) => ({
    figure: `${c} alone, FLamby's metric`, source: F.source, published: f4(F.alone[k]), offline: f4(F.offline.alone[k]),
    thisRun: `${f4(r.mean.alone[k])} (seed ${r.seed})`, verdict: "no spread" as Verdict, why: noSpread(F.alone[k], r.mean.alone[k]),
  }));
  const d = r.mean.federated - F.fedavg.mean;
  const inside = Math.abs(d) <= 2 * F.fedavg.sd;
  rows.push({
    figure: `FedAvg, ${r.rounds} rounds, FLamby's metric`, source: F.source,
    published: `${f4(F.fedavg.mean)} +/- ${f4(F.fedavg.sd)}`, offline: `${f4(F.offline.fedavg.mean)} +/- ${f4(F.offline.fedavg.sd)}`,
    thisRun: `${f4(r.mean.federated)} (seed ${r.seed})`, verdict: inside ? "reproduced" : "differs",
    why: `the same preparation, model and FedAvg schedule; this run is ${signed(d)} from FLamby's mean, ${inside ? "inside" : "outside"} two of its seed standard deviations`,
  });
  rows.push({
    figure: "pooled in one place, FLamby's metric", source: F.source, published: f4(F.pooled), offline: f4(F.offline.pooled),
    thisRun: `${f4(r.mean.pooled)} (seed ${r.seed})`, verdict: "no spread", why: noSpread(F.pooled, r.mean.pooled),
  });
  rows.push({
    figure: "FedYogi, FLamby's metric", source: F.source, published: `${f4(F.fedyogi.mean)} +/- ${f4(F.fedyogi.sd)}`,
    offline: "not run", thisRun: "not run", verdict: "not comparable", why: "this page federates by FedAvg only",
  });
  return rows;
}

/** What a page with no model and no benchmark says instead of an empty table. */
export const PAGE_NOTES: Record<string, string> = {
  defi_lending: "The lending page publishes no result to reproduce: its pool follows Aave V3's published parameters (a close factor of 0.5 above a health factor of 0.95, and of 1 at or below it), which are inputs here, labelled illustrative, and the private check is compared with the same check in the clear on the page itself.",
  telecom_ids: "No single published figure on these corpora is trustworthy enough to set beside the detectors: each model's row below says why, in the registry's own words, and the 5G-NIDD card shows the published row counts beside the cleaned ones.",
};

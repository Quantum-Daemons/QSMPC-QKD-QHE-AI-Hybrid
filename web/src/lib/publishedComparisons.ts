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
  // "reference" since Round 86, when it was first measured here: the paper's figure is on its own split and this run on this
  // project's windows, the reason the gradient booster's row below is a reference too.
  "finance_aml.gnn": { relation: "reference", why: "Multi-PNA+EU itself from IBM's code, on this project's windows rather than the paper's split" },
  "finance_aml.gnn_causal": { relation: "reference", why: "Multi-PNA+EU from IBM's code under the causal protocol, which reads less than the paper's" },
  "finance_aml.gbm": { relation: "reference", why: "same metric and corpus, a different protocol: the published figure splits by transaction order and updates IBM's Graph Feature Preprocessor over the test window" },
  "finance_aml.student": { relation: "reference", why: "a logistic regression distilled from the gradient-boosting model, beside the published GFP+XGBoost" },
  "finance_aml.mlp_large": { relation: "reference", why: "an MLP over this project's features, beside the published GFP+XGBoost" },
  "finance_aml.gnn_compact": { relation: "reference", why: "a compact message-passing model distilled from Multi-PNA+EU, beside Multi-PNA+EU itself" },
  "genomics.prs": { relation: "citation", why: "the published AUROC is for a model fitted with covariates on UK Biobank, with 162 of the score's 187 variants; this page scores 30 public individuals, so it is a citation and not a target" },
  "medical_fl.resnet18": { relation: "same", why: "ResNet-18 at 28 px on BreastMNIST's official split, as in MedMNIST v2" },
  "medical_fl.resnet50": { relation: "reference", why: "ResNet-50 at 28 px under a training regime that differs from the benchmark table's" },
  "medical_fl.student": { relation: "citation", why: "a single-convolution CNN reading 14 x 14 images, distilled for CKKS, beside the benchmark's ResNet-18 at 28 x 28" },
  "medical_fl.square_student": { relation: "citation", why: "a single-convolution CNN with a square activation reading 14 x 14 images, run end to end under CKKS, beside the benchmark's ResNet-18 at 28 x 28" },
  "medical_fl.cxr_fedavg": { relation: "reference", why: "a one-finding ResNet-18 from random initialisation, trained under FedAvg at 224 px, beside the dataset paper's 14-finding ResNet-50 from ImageNet with its last layers trained, on the same official test split" },
  "satellite_comms.telemanom": { relation: "same", why: "Telemanom on ESA-ADB's Mission1 lightweight channels 41 to 46, anomalies alone: the like-for-like row of the benchmark" },
  "satellite_comms.student": { relation: "reference", why: "a linear forecaster distilled from Telemanom, beside Telemanom's benchmark row" },
  "satellite_comms.opssat": { relation: "reference", why: "gradient-boosted trees over the published features, beside the FCNN, the best of 30 algorithms benchmarked" },
  "satellite_comms.opssat_student": { relation: "reference", why: "a logistic regression over a quadratic expansion, beside the benchmark's FCNN" },
};

const f4 = (x: number) => x.toFixed(4);
const signed = (x: number) => (x >= 0 ? `+${x.toFixed(4)}` : x.toFixed(4));

/** This run's re-scoring of the model, NAMING its metric: a sample is scored on the metric it was published with, which
 *  need not be the row's (BreastMNIST's samples record accuracy beside an AUC row), and an unnamed 0.8590 beside a
 *  published AUC of 0.901 reads as the AUC (seen on the deployed medical page, 2026-11-13). */
function thisRunOf(v: VerificationSummary | undefined, rowMetric: string): string {
  if (!v) return "not run in this tab";
  if (v.outcome === "PASS" || v.outcome === "FAIL") {
    const other = v.metric && v.metric !== rowMetric ? ", not this row's metric" : "";
    return `${v.outcome}: ${v.metric ?? "metric"} ${f4(v.score!)} on ${v.n} rows of its sample (${v.split ?? "split not declared"}), `
      + `recorded ${f4(v.expected!)}${other}`;
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
      thisRun: thisRunOf(v, name),
    };
    if (published == null) {
      return { ...base, verdict: "not comparable" as Verdict, why: m.metric?.baseline_source || "no published figure" };
    }
    if (!rel || rel.relation === "citation") {
      return { ...base, verdict: "not comparable" as Verdict, why: rel?.why ?? "no relation recorded for this model" };
    }
    if (rel.relation === "reference") {
      const d = offlineValue == null ? "" : ` (${signed(offlineValue - published)} against it)`;
      // A record that also carries the figure the published method's own loop logs, over its seeds (IBM's Multi-GNN):
      // that is the like-for-like number for the paper's, so the row says it rather than leaving the exported one alone.
      const lg = (m.achieved as Record<string, any> | null)?.five_seeds?.logged_test_f1 as { mean: number; sample_sd: number; n: number } | undefined;
      const logged = lg ? `; the figure its own training loop logs, the paper's measure, is ${f4(lg.mean)} +/- ${f4(lg.sample_sd)} over ${lg.n} seeds here` : "";
      return { ...base, verdict: "reference" as Verdict, why: `${rel.why}${d}${logged}` };
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

/** A trained QCNN's headline as its weights file records it, and the re-scoring a QCNN run in this tab makes. */
export interface QcnnRecord { metric: string; value: number }
export interface QcnnRunRescore { label: string; headline: number; rows: number; matchesRecord: boolean }

const QCNN_METRIC: Record<string, string> = { minority_class_f1: "minority-class F1", accuracy: "accuracy", macro_f1: "macro F1" };

/**
 * Published quantum classifiers beside each page's trained QCNN (checklist CS13; the owner's choice of 2026-11-24:
 * same-split classical baselines for both techniques, literature figures only as labelled references). The QCNN is in
 * neither the registry nor the Hub, so `modelRows` never draws it, and its same-split baseline, logistic regression, is
 * in its run's own log line. Every figure here was read at its source on 2026-11-24. None is a QCNN on the page's own
 * data and split, so none can be a reproduction: the same data with another method or split is a reference, and other
 * data is not comparable.
 */
export const QCNN_REFERENCES: Record<string, { source: string; published: string; verdict: Verdict; why: string }[]> = {
  finance_aml: [{
    source: "Douros et al., arXiv:2607.21449 (accepted at IEEE QCE 2026)",
    published: "0.9210",
    verdict: "not comparable",
    why: "A hybrid network with a variational quantum layer, not a QCNN, trained federated: the F1 averaged over six "
      + "settings (its text beside Table I), with its classical baseline at 0.9170. The data are PaySim by the paper's "
      + "footnote link, but its split, fraud rate and any rebalancing are not reported, so the figure cannot stand beside "
      + "a held-out F1 at PaySim's own rate",
  }],
  medical_fl: [{
    source: "Xiang et al., Sci. Rep. 14:24699 (2024), doi:10.1038/s41598-024-74778-7",
    published: "0.9616",
    verdict: "reference",
    why: "A quantum-classical hybrid CNN on the same data (WDBC), its accuracy averaged over training iterations on a 30% "
      + "validation set with no separate test set (Table 4; the best iteration reads 0.9766, logistic regression's "
      + "average 0.9118): a different method and split on the same data",
  }, {
    source: "Yadav et al., PeerJ Comput. Sci. (2026), doi:10.7717/peerj-cs.3600",
    published: "0.9474 to 0.9561",
    verdict: "reference",
    why: "A hybrid QSVM-QNN, simulated, on the same data (WDBC) with seven selected features: held-out accuracy over three "
      + "repeats on a stratified 20% split of 114 rows: a different method and split on the same data",
  }],
  telecom_ids: [{
    source: "Al-Kuwari et al., IET Quantum Commun. (2026), doi:10.1049/qtc2.70028 (arXiv:2509.14282v2)",
    published: "0.947",
    verdict: "not comparable",
    why: "A hybrid quantum LSTM (9 qubits, simulated) on the authors' own simulated decoy-state BB84 telemetry, eight "
      + "balanced classes, at 50 epochs (Table VIII): the closest published task, on other data, and the paper does not "
      + "say how its F1 averages the classes",
  }],
};

/** One row per published reference, carrying the page's own QCNN figure and this tab's re-scoring when a QCNN ran. */
export function qcnnRows(usecase: string, record: QcnnRecord | null, rescore: QcnnRunRescore | null): ComparisonRow[] {
  const metric = record ? (QCNN_METRIC[record.metric] ?? record.metric) : rescore?.label ?? "its headline metric";
  return (QCNN_REFERENCES[usecase] ?? []).map((r) => ({
    figure: `QCNN (trained, not encrypted): ${metric}`,
    source: r.source,
    published: r.published,
    offline: record ? f4(record.value) : "not loaded",
    thisRun: rescore
      ? `re-scored ${f4(rescore.headline)} on ${rescore.rows} rows of the trainer's test split, `
        + (rescore.matchesRecord ? "equal to the record" : "not the record")
      : "not run in this tab: choose QCNN under Parameters",
    verdict: r.verdict,
    why: r.why,
  }));
}

/** What a page with no model and no benchmark says instead of an empty table. */
export const PAGE_NOTES: Record<string, string> = {
  defi_lending: "The lending page publishes no result to reproduce: its pool follows Aave V3's published parameters (a close factor of 0.5 above a health factor of 0.95, and of 1 at or below it), which are inputs here, labelled illustrative, and the private check is compared with the same check in the clear on the page itself.",
  telecom_ids: "No single published figure on these corpora is trustworthy enough to set beside the detectors: each model's row below says why, in the registry's own words, and the 5G-NIDD card shows the published row counts beside the cleaned ones.",
};

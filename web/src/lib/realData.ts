// Datasets shipped to the CLIENT-SIDE demo (web/public/data/), so the demo scores real inputs with
// real models rather than sampling noise — matching the FULL deployment. Coverage: medical (Wisconsin
// breast-cancer, real measurements), genomics (published PGS000072 weights over real 1000 Genomes
// genotypes), finance (PaySim SIMULATOR output, i.e. synthetic rows) and satellite (real OPS-SAT-AD
// telemetry segments since Round 84; the NASA SMAP/MSL windows it replaced carry no data licence from
// any provider and are no longer served). Only telecom stays fully synthetic - no network-attack corpus with an
// affirmative commercial grant exists (UNSW-NB15 needs the authors' agreement; CIC-IDS2017 states no
// licence at all) — and the demo logs that rather than quietly substituting. Tiny static JSON,
// fetched and cached once.
//
// These are BUNDLED SAMPLES, not the corpora the published models were trained on, and they are not
// uniformly commercial-OK: the genomics payload's PGS000072 effect weights are CC BY 4.0, while its 30
// individuals' 1000 Genomes dosages carry no licence instrument (IGSR's disclaimer); both are served
// here and neither goes to the Hub. Licensing prose lives in
// DATA_LICENSES.md and the machine-readable authority is
// src/qsh/models/registry.py (BUNDLED_DEMO_SAMPLES), which is what the demo actually prints via
// config.demoSample(). No licence summary is restated here on purpose: the one that used to sit at
// this line got four of five wrong (WDBC is CC BY 4.0, not public domain; PGS is per-score, not
// blanket CC-BY; IGSR grants no blanket licence; no provider states one for SMAP/MSL) and shipped
// that way. A stale copy reads as authoritative, which is worse than a pointer.
//
// THE HELD-OUT ROWS ARE A SEPARATE THING (Round 84, `heldOutCorpus` at the bottom). The samples
// above feed the page's one-row explanation and the federated demo. What the private-inference figure
// encrypts is a published model on rows it was never fitted on, each with its own payload naming the
// split, the licence and its link.

/** One row in every space the payload declares, keyed by that space's own name.
 *
 *  Built in ONE place so a consumer selecting by name cannot pick up an array the file never
 *  labelled. `samples` is written last, so a payload naming the same space twice resolves to its
 *  primary declaration rather than to an entry in the extras map. An array of a different length
 *  cannot be indexed by the same row index and is dropped rather than mis-indexed.
 */
function rowBySpace(ds: any, idx: number, width: number): Record<string, number[]> {
  const out: Record<string, number[]> = {};
  const n = (ds.samples as number[][] | undefined)?.length ?? 0;
  const extras = (ds.samples_by_space ?? {}) as Record<string, number[][]>;
  for (const [space, rows] of Object.entries(extras)) {
    if (Array.isArray(rows) && rows.length === n) out[space] = rows[idx].slice(0, width);
  }
  if (typeof ds.feature_space === "string" && Array.isArray(ds.samples)) {
    out[ds.feature_space] = (ds.samples as number[][])[idx].slice(0, width);
  }
  return out;
}

export interface RealInput {
  values: number[];                          // the demo features (REAL): 8 on the tabular pages, every score variant on genomics
  /** THE SAME ROW in every space the payload declares, keyed by the space's own name.
   *
   *  `values` is StandardScaler output, because the distilled CKKS-linear model and the exact-linear
   *  SHAP panel are both fitted in that space and must stay there. But `qcnn_weights.json` for
   *  finance declares `input_space: "raw"`, so `qcnnInferTrained` standardises whatever it is handed
   *  a SECOND time, with the trainer's moments rather than this sample's. Measured before this map
   *  existed, through the browser's own encoder: wire 0 (`amount`) collapsed to a tanh sd of
   *  3.3e-05, 556 of 3200 values saturated, and 101 of 400 rows - 25.25% - were given a different
   *  class than the same rows correctly fed.
   *
   *  KEYED BY NAME, not a boolean. A consumer asks for the space it was trained on and gets nothing
   *  if the file cannot serve it. A `input_space === "raw" ? a : b` selection has a silent default
   *  branch, so an artifact retrained into a third space would land on whichever array the else
   *  names - which is precisely how this defect shipped. */
  bySpace?: Record<string, number[]>;
  names: string[];                           // feature names (mirror the backend / real rsIDs)
  provenance: string;                        // human label for the event log + UI badge
  model?: { W: number[][]; b: number[] };    // a REAL model (trained LR / PRS weights) for faithful SHAP
  metrics?: Record<string, number>;          // extra real metrics, e.g. polygenic_risk_score
  /** Column means E[x] over the shipped sample, for the SHAP reference point.
   *
   *  Without this `linearShap` defaults its baseline to the ZERO vector, and the attributions are
   *  then measured against a reference case that does not occur. The finance `amount` column has a
   *  mean of 11.5 on a log scale, so a zero reference means every bar for it reads against "a
   *  transaction of log-amount zero" rather than against a typical one, inflating it relative to
   *  every other feature. The completeness self-check cannot catch this: it holds for ANY baseline. */
  baseline?: number[];
  /** What each feature IS, in the payload's words, when it says (PP14, Round 86): finance's four are AML-style names
   *  for PaySim quantities (`cross_border` is the TRANSFER/CASH_OUT type; PaySim records no geography). */
  definitions?: Record<string, string>;
}

/** Column means over the sample rows - the reference point SHAP attributions are measured from. */
function columnMeans(rows: number[][], n: number): number[] {
  const out = new Array(n).fill(0);
  if (!rows.length) return out;
  for (const r of rows) for (let i = 0; i < n; i++) out[i] += (r[i] ?? 0) / rows.length;
  return out;
}

const cache = new Map<string, unknown>();
async function fetchJSON(url: string): Promise<any> {
  if (cache.has(url)) return cache.get(url);
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → HTTP ${r.status}`);
  const j = await r.json();
  cache.set(url, j);
  return j;
}

/**
 * Returns REAL input + a REAL model where a real sample ships: medical (WDBC, CC BY 4.0), finance
 * (PaySim simulator rows) and genomics, whose PGS000072 weights are CC BY 4.0 and whose 1000 Genomes
 * genotypes carry no licence instrument (DATA_LICENSES.md sections 4 and 5). `null` for
 * everything else (the caller keeps its synthetic features and logs that honestly).
 * `u` is the pipeline's seeded RNG so a run is deterministic.
 */
export async function realInput(usecase: string, u: () => number): Promise<RealInput | null> {
  try {
    if (usecase === "medical_fl") {
      // REAL Wisconsin Diagnostic Breast Cancer sample + a REAL logistic-regression model (8 features).
      // The row comes from `heldout` where the payload has one (Round 84): the class the page reports
      // and the bars explaining it are then a prediction on a row the regression never saw. The
      // SHAP reference point stays E[x] over the FITTING rows, the population the model was fitted
      // to, which is what a baseline is supposed to describe.
      const ds = await fetchJSON("/data/medical/breast_cancer_sample.json");
      const rows = ds.samples as number[][];
      // The held-out block in the shape `rowBySpace` reads, so the row arrives in EVERY space the
      // file ships (the medical QCNN declares `raw`; the regression and SHAP use `zscore`).
      const src = (ds.heldout?.samples as number[][] | undefined)?.length
        ? { samples: ds.heldout.samples, samples_by_space: ds.heldout.samples_by_space,
            feature_space: ds.feature_space }
        : ds;
      const k = Math.floor(u() * src.samples.length) % src.samples.length;
      const row = (src.samples as number[][])[k];
      return { values: row.slice(0, 8), bySpace: rowBySpace(src, k, 8), names: ds.names,
               provenance: ds.provenance, model: ds.model, baseline: columnMeans(rows, 8) };
    }
    if (usecase === "genomics") {
      // REAL PGS-Catalog PGS000072 effect weights (Graff et al., Nat Commun 12:970, 2021, CC BY 4.0)
      // + REAL 1000 Genomes phase-3 genotypes (read through Ensembl; IGSR attaches no licence
      // instrument). No synthetic sampling; real genotypes throughout.
      //
      // THE FULL SCORE (Round 84). The row is the person's dosage at EVERY scored variant and the
      // model is the published weights in the two-logit form W = [0, w], b = [0, -cut], so CKKS
      // evaluates the whole score in one ciphertext and the class is "score above the cut". A sign
      // test on a few large-weight variants, which this replaced, cannot answer for the score.
      // The cut comes from the bundled panel alone - see `prsPanel`.
      const p = await prsPanel();
      const k = Math.floor(u() * p.rows.length) % p.rows.length;   // a REAL 1000G individual
      const n = p.weights.length;
      return {
        values: p.rows[k].slice(), names: p.keys.slice(), provenance: p.provenance,
        model: { W: [new Array(n).fill(0), p.weights.slice()], b: [0, -p.cut] },
        metrics: {
          polygenic_risk_score: Number(p.scores[k].toFixed(4)),     // the score in the clear
          prs_variants: n,
          prs_high_risk_percentile: p.percentile,
          prs_high_risk_cut: Number(p.cut.toFixed(6)),
          prs_cut_population: p.rows.length,
        },
        // Population mean dosage at every variant. The zero vector would mean "an individual carrying
        // no effect allele anywhere", which is not a typical genome and at several of these loci is
        // not present in the sample at all.
        baseline: p.meanDosage,
      };
    }
    if (usecase === "finance_aml") {
      // PaySim SIMULATOR output - synthetic transactions (Kaggle ealaxi/paysim1, CC-BY-SA-4.0) - and
      // a logistic surrogate distilled from a gradient-boosting teacher, both fitted on these bundled
      // rows; it is NOT the AMLworld student the private-inference figure encrypts. The model reads
      // the FOUR columns in `names`, AML-style labels for PaySim quantities that the payload defines
      // in `feature_definitions`: amount is log(1 + amount), cross_border the TRANSFER/CASH_OUT type
      // (PaySim records no geography), night_tx a step in hours 0 to 5, new_payee the first row
      // naming that destination. `excluded_features` says why the bundled TSV's other four are not
      // inputs: device_risk and geo_mismatch are built from PaySim's balance columns, which the
      // dataset's description says must not be used for fraud detection because fraudulent
      // transactions are cancelled; amount_zscore copies amount; velocity_24h is constant here.
      const ds = await fetchJSON("/data/finance/paysim_sample.json");
      const rows = ds.samples as number[][];
      const width = (ds.names as string[]).length;
      const idx = Math.floor(u() * rows.length) % rows.length;
      const row = rows[idx];                                           // a PaySim simulator row (synthetic)
      // The same transaction in raw units, for any consumer whose artifact declares `raw`. Indexed,
      // not re-drawn: a second random pick would hand the QCNN a different transaction from the one
      // the CKKS tier and the SHAP panel are describing, and the page would attribute one row's
      // explanation to another row's prediction. `raw` is the same four columns as `zscore`, in
      // `names` order, unstandardised: what the finance QCNN artifact reads since its Round 84
      // retrain, in the order its `feature_names` lists and the circuit reads by position. Each
      // space keeps the width the file ships rather than this one, so a raw block of another width
      // reaches the QCNN branch whole and is refused there instead of being cut to its first columns.
      return { values: row.slice(0, width), bySpace: rowBySpace(ds, idx, Number.MAX_SAFE_INTEGER),
               names: ds.names, provenance: ds.provenance, model: ds.model,
               baseline: columnMeans(rows, width), definitions: ds.feature_definitions ?? undefined };
    }
  } catch (e) {
    console.warn("[realData] real dataset unavailable → synthetic fallback:", e);
  }
  return null;   // telecom (+ any load failure) → synthetic, logged by the caller
}

/** Each cohort's effect-allele count at the score's first variant, over the bundled 1000 Genomes
 *  slice, split into `cohorts` contiguous groups - the split `np.array_split(dosages[:, 0], n)` makes
 *  in qsh/usecases/genomics, where the first (len mod n) groups take one extra individual.
 *
 *  Round 80. The browser's secure sum added `floor(u() * 100)` per party - random stand-ins - while
 *  the page and its diagram said the cohorts pool allele counts. These are the counts that sentence
 *  describes, from genotypes the page already ships. `null` when the file cannot be read: the caller
 *  refuses rather than pooling stand-ins under the allele-count name. */
export async function realAlleleCounts(
  cohorts: number,
): Promise<{ counts: number[]; variant: string; individuals: number } | null> {
  try {
    const prs = await fetchJSON(PRS_PAYLOAD_PATH);
    const samples = prs.dosage_samples as number[][];
    const keys = (prs.keys ?? prs.rsids) as string[] | undefined;
    const column = samples.map((row) => row[0] ?? 0);
    const n = Math.max(1, Math.min(Math.floor(cohorts), column.length));
    const base = Math.floor(column.length / n), extra = column.length % n;
    const counts: number[] = [];
    for (let k = 0, at = 0; k < n; k++) {
      const size = base + (k < extra ? 1 : 0);
      counts.push(column.slice(at, at + size).reduce((a, b) => a + b, 0));
      at += size;
    }
    return { counts, variant: keys?.[0] ?? "variant 0", individuals: column.length };
  } catch {
    return null;
  }
}

/** The genomics payload: PGS000072's scored variants, 30 1000 Genomes individuals, and their credits.
 *  Generated by `scripts/data/build_prs_browser_payload.py`; the registry's `DEMO_SAMPLE_FILES` names it. */
export const PRS_PAYLOAD_PATH = "/data/genomics/pgs000072_breast_cancer.json";

/** The percentile of the full score the genomics page calls high risk (Round 84).
 *
 *  Twin of `HIGH_RISK_PERCENTILE` in src/qsh/usecases/genomics, and the percentile the Models card's
 *  `global_high_risk_cut` is measured at (scripts/train/genomics_prs.py); a test pins the three. A
 *  demo cut over the bundled panel, not a clinical threshold. */
export const PRS_HIGH_RISK_PERCENTILE = 90;

/** The `percentile`-th percentile of `values`, as numpy's default (`method="linear"`) computes it.
 *
 *  Virtual index (n - 1) * q on the sorted values, and linear interpolation between its two
 *  neighbours, taken from the upper one when the fraction is at least a half, as numpy's `_lerp`
 *  does - so the browser's cut and the Python twin's agree to the last bit on the same scores. */
export function percentileLinear(values: number[], percentile: number): number {
  const s = [...values].sort((a, b) => a - b);
  const n = s.length;
  if (!n) return Number.NaN;
  const v = (n - 1) * (percentile / 100);
  if (v >= n - 1) return s[n - 1];
  if (v <= 0) return s[0];
  const lo = Math.floor(v), g = v - lo, a = s[lo], b = s[lo + 1], d = b - a;
  return g >= 0.5 ? b - d * (1 - g) : a + d * g;
}

/** The genomics page's bundled panel and the full score over it (Round 84). */
export interface PrsPanel {
  scoreId: string;
  provenance: string;
  /** Variant identifiers in the scoring file's order (rsIDs for PGS000072). */
  keys: string[];
  weights: number[];
  /** Every bundled individual's effect-allele dosage at every variant, in `keys` order. */
  rows: number[][];
  /** Each row's full score in the clear, sum_i w_i * d_i. */
  scores: number[];
  percentile: number;
  /** The high-risk cut: the `percentile`-th percentile of `scores`, by `percentileLinear`. From the
   *  bundled rows alone, the scored person included. */
  cut: number;
  /** Mean dosage per variant, the SHAP reference point: the payload's `population_mean_dosage`
   *  (over every individual its generator read, 2,504) where it ships, else the bundled rows' means.
   *  The first is a population estimate and the second one from a redistribution-limited slice, and
   *  the reference point should not be degraded by a licence constraint. */
  meanDosage: number[];
  /** Individuals behind `meanDosage`. */
  meanDosageOver: number;
}

/** Load the genomics payload as a `PrsPanel`. Generic over the number of variants and individuals:
 *  another score is a different payload, not a code change. Throws on a payload whose rows do not
 *  match its weights, rather than scoring a person on the wrong variants.
 *
 *  The payload is GENERATED by `scripts/data/build_prs_browser_payload.py`. Only 30 of the 2,504
 *  individuals ship: IGSR attaches no licence instrument to the genotypes, and DATA_LICENSES section 4
 *  justifies a slice of that size, not the full matrix. Its `variants_unscored` lists the score's
 *  variants with no 1000 Genomes genotype, which the weights and rows leave out. */
export async function prsPanel(): Promise<PrsPanel> {
  const prs = await fetchJSON(PRS_PAYLOAD_PATH);
  const weights = (prs.weights as number[] | undefined) ?? [];
  // `keys` (the scoring file's identifiers; rsIDs for PGS000072), or `rsids` in an older payload.
  const keys = ((prs.keys ?? prs.rsids) as string[] | undefined) ?? [];
  const rows = (prs.dosage_samples as number[][] | undefined) ?? [];
  const m = weights.length;
  if (!m || keys.length !== m || !rows.length || rows.some((r) => !Array.isArray(r) || r.length !== m)) {
    throw new Error(`the genomics payload is malformed: ${m} weights, ${keys.length} keys, `
      + `${rows.length} rows of widths ${[...new Set(rows.map((r) => r?.length))].join("/")}`);
  }
  const scores = rows.map((d) => weights.reduce((a, wi, i) => a + wi * (d[i] ?? 0), 0));
  const pop = prs.population_mean_dosage as number[] | undefined;
  const popOk = Array.isArray(pop) && pop.length === m;
  return {
    scoreId: String(prs.score_id ?? ""), provenance: String(prs.provenance ?? ""),
    keys, weights, rows, scores, percentile: PRS_HIGH_RISK_PERCENTILE,
    cut: percentileLinear(scores, PRS_HIGH_RISK_PERCENTILE),
    meanDosage: popOk ? pop : columnMeans(rows, m),
    meanDosageOver: popOk ? Number(prs.n_individuals_available ?? rows.length) : rows.length,
  };
}

/** The OPS-SAT-AD segments the satellite page scores, and the file they come from. */
export const SATELLITE_SEGMENTS_PATH = "/data/satellite/opssat_segments.json";

export interface SatelliteAnomaly {
  provenance: string; anomalyScore: number; isAnomaly: boolean;
  /** Telemetry channel names where the payload has them. OPS-SAT-AD's precomputed segment
   *  features carry no channel column, so this is empty there rather than invented. */
  channels: string[];
  /** The real linear model, the scored segment and the SHAP reference point.
   *  Returned so the run can compute exact linear SHAP for satellite the same way the three
   *  inference use cases do. The model here IS linear (a 2xN margin over one segment's features),
   *  so the method applies unchanged - no new or invented attribution is involved. */
  model: { W: number[][]; b: number[] };
  window: number[];
  baseline: number[];
  featureNames: string[];
  /** What the reference point is, in words, from the payload: the mean of the FITTING segments. */
  baselineNote: string;
  /** What the model is, from the payload, so the run cannot describe it as anything else. */
  modelNote: string;
  /** The segment's true label (1 = anomalous): it is a held-out row with a known answer. */
  trueLabel: number | null;
  licence: string;
  licenceUrl: string | null;
}

/**
 * A plaintext anomaly verdict over ONE held-out OPS-SAT-AD segment, with the payload's own linear
 * model - a JS dot product, no FHE or MPC (the satellite flow has no `private_inference` stage).
 *
 * Round 84 replaced the NASA SMAP/MSL windows this read. No provider grants a licence for that data
 * (the Telemanom licence is Caltech's and covers code; JPL staff are Caltech employees, so 17 U.S.C.
 * 105 does not reach it), and DATA_LICENSES.md's rule is that silence is not permission. OPS-SAT-AD is
 * CC BY 4.0. The segments shipped are the 159 the regression was NOT fitted on, so the verdict is a
 * prediction on an unseen segment; the regression is a demo model fitted on 370 other test-partition
 * segments, not the published OPS-SAT models, and `modelNote` says so wherever it is printed.
 */
export async function realSatelliteAnomaly(u: () => number): Promise<SatelliteAnomaly | null> {
  try {
    const ds = await fetchJSON(SATELLITE_SEGMENTS_PATH);
    const segments = ds.segments as number[][];
    const k = Math.floor(u() * segments.length) % segments.length;
    const w = segments[k];                                          // a REAL held-out OPS-SAT segment
    const W = ds.model.W as number[][]; const b = ds.model.b as number[];
    const margin = w.reduce((a, v, i) => a + (W[1][i] - W[0][i]) * v, 0) + (b[1] - b[0]);
    const score = 1 / (1 + Math.exp(-margin));
    const names = (ds.names as string[] | undefined) ?? w.map((_, i) => `f${i}`);
    return { provenance: ds.provenance, anomalyScore: Number(score.toFixed(4)),
             isAnomaly: score > 0.5, channels: (ds.channel_names as string[] | undefined) ?? [],
             model: { W, b }, window: w,
             // The payload's own reference point (the fitting segments' mean), never an average of
             // the held-out rows being explained.
             baseline: (ds.baseline as number[] | undefined) ?? new Array(w.length).fill(0),
             featureNames: names,
             baselineNote: String(ds.baseline_note ?? "mean of the fitting segments"),
             modelNote: String(ds.real_model ?? ""),
             trueLabel: Number.isInteger(ds.labels?.[k]) ? ds.labels[k] : null,
             licence: String(ds.licence ?? ""), licenceUrl: (ds.licence_url as string | undefined) ?? null };
  } catch (e) {
    console.warn("[realData] satellite OPS-SAT segments unavailable -> representative:", e);
    return null;
  }
}

/**
 * The page's REAL bundled sample as a labelled training set, for in-browser federated learning.
 *
 * `clientQFL` used to build its own two-cluster toy unconditionally, so "federated accuracy" was
 * the same number on every industry page for a given seed and described no dataset the page showed.
 * Returns null where a use case has no labelled bundled sample, and the caller says so rather than
 * presenting the toy's accuracy as an industry result.
 */
export async function realFederatedSet(
  usecase: string,
): Promise<{ X: number[][]; Y: number[] } | null> {
  const files: Record<string, string> = {
    medical_fl: "/data/medical/breast_cancer_sample.json",
    finance_aml: "/data/finance/paysim_sample.json",
  };
  const path = files[usecase];
  if (!path) return null;
  try {
    const ds = await fetchJSON(path);
    const X = (ds.samples as number[][] | undefined)?.map((r) => r.slice(0, 8));
    const Y = (ds.labels as number[] | undefined)?.map((v) => Number(v) | 0);
    if (!X?.length || !Y?.length || X.length !== Y.length) return null;
    return { X, Y };
  } catch {
    return null;      // absent or malformed: the caller reports the toy explicitly
  }
}

/**
 * Every row of the page's own corpus, in the SAME 8-feature space `realInput` builds one row in.
 *
 * WHY IT LIVES HERE. The encrypted path needs a corpus to evaluate the bundled model over, and the
 * one thing that must not drift is the feature space: encrypting rows in a different space from the
 * model's would produce an agreement figure that measures nothing. `realInput` defines that space
 * per use case - `row.slice(0, 8)` for the tabular pages, every score variant for genomics -
 * so the corpus is built from the same source with the same selection, in the same file.
 *
 * Returns `null` where the page has no corpus of its own, rather than an empty array, so a caller
 * cannot mistake "no corpus" for "a corpus with nothing in it".
 *
 * ROUND 84: NEVER THE MODEL'S FITTING ROWS. Medical returns the 171 rows held out from the bundled
 * regression, not the 398 it was fitted on. Finance returns null: its 400 PaySim rows are the
 * surrogate's fitting rows, and its private-inference figure is now the published AMLworld student on
 * its held-out test window (`heldOutCorpus`), which is a different model in a different feature
 * space and so cannot come through this 8-feature path. Genomics keeps its 30 reference-panel
 * individuals, which PGS000072 was not fitted on either.
 */
export async function realInputCorpus(usecase: string): Promise<number[][] | null> {
  try {
    if (usecase === "medical_fl") {
      const ds = await fetchJSON("/data/medical/breast_cancer_sample.json");
      const rows = ds.heldout?.samples as number[][] | undefined;
      return rows?.length ? rows.map((r) => r.slice(0, 8)) : null;
    }
    if (usecase === "genomics") {
      // Every bundled individual at EVERY variant (Round 84), through the same `prsPanel` `realInput`
      // reads, so the rows the panel figure encrypts and the row the page scores cannot diverge.
      return (await prsPanel()).rows.map((r) => r.slice());
    }
  } catch (e) {
    console.warn("[realData] corpus unavailable:", e);
  }
  return null;
}

/** One axis of an overlay plane, named as `real_2d.json` names it. */
export interface PlaneAxis { key: string; label: string; scale?: "log" }

/** Who may use the rows and on what terms, as the payload states it. `url` is null only where no
 *  licence is granted at all (the generated telecom corpus): there is then nothing to link to. */
export interface DataLicence { name: string; url: string | null; attribution: string; modifications: string }

/**
 * A published model and rows it was NEVER fitted on, ready to be encrypted (Round 84).
 *
 * The private-inference figure's job is to show what the encrypted model does on data it has not
 * seen. Each field below exists because the figure once said something false without it: `fitNote`
 * because finance and medical called the model's own fitting rows "held-out"; `plane` because the
 * overlay beside the figure drew a different corpus from the one encrypted; `licence.url` and
 * `licence.modifications` because CC BY 4.0 and CDLA-Sharing ask for a link and a change notice,
 * not a name.
 */
export interface HeldOutCorpus {
  usecase: string;
  /** The registry key of the scored model; null for medical's bundled regression, which has none. */
  modelKey: string | null;
  /** The model in a phrase, for the caption. */
  modelLabel: string;
  /** In the K-logit form the batch evaluates. A single-logit model gets an all-zero class-0 row, so
   *  its argmax is the sign of the logit and that row costs no homomorphic work. */
  model: { W: number[][]; b: number[] };
  /** The rows, preprocessed exactly as the model's own file declares. */
  rows: number[][];
  labels: number[];
  classNames: string[];
  /** The same rows on the overlay's plane, in the overlay's units, index-aligned with `rows`. */
  plane: { x: PlaneAxis; y: PlaneAxis; points: [number, number][] } | null;
  corpus: string;
  /** What the rows are: the split, in a phrase. */
  split: string;
  /** Which rows the model WAS fitted on, so the caption can say these are not them. */
  fitNote: string;
  licence: DataLicence;
  /** The plaintext figures the payload's builder computed, for the run to reproduce. */
  expected: Record<string, unknown> | null;
}

/** Apply the preprocessing a weights file declares, step by step, refusing an unknown step. */
/** scikit-learn's PolynomialFeatures(degree=2, include_bias=False): [x_i], then x_i * x_j for i <= j. */
export function quadraticExpansion(row: number[]): number[] {
  const out = row.slice();
  for (let i = 0; i < row.length; i++) {
    for (let j = i; j < row.length; j++) out.push(row[i] * row[j]);
  }
  return out;
}

function preprocess(row: number[], steps: string[], mean: number[], scale: number[]): number[] {
  // Computed on the plaintext row, before encryption, so the encrypted step is still one dot product.
  const input = steps[0] === "quadratic expansion" ? quadraticExpansion(row) : row;
  const perValue = steps[0] === "quadratic expansion" ? steps.slice(1) : steps;
  return input.map((v, i) => {
    let x = v;
    for (const step of perValue) {
      if (step === "subtract mean") x -= mean[i] ?? 0;
      else if (step === "divide by scale") x /= scale[i] || 1;
      // The same refusal `onnxInference.applyPreprocessing` makes: a guessed transform turns a
      // published model into a different one that still produces plausible numbers.
      else throw new Error(`unknown preprocessing step "${step}" - refusing to guess`);
    }
    return x;
  });
}

/** The x/y columns of the plane, from the rows themselves (finance, telecom: raw columns). */
function planeFromColumns(rows: number[][], names: string[], plane: { x: PlaneAxis; y: PlaneAxis }) {
  const xi = names.indexOf(plane.x.key), yi = names.indexOf(plane.y.key);
  if (xi < 0 || yi < 0) return null;
  return { x: plane.x, y: plane.y, points: rows.map((r) => [r[xi], r[yi]] as [number, number]) };
}

function licenceOf(ds: any): DataLicence {
  return { name: String(ds.licence ?? ""), url: (ds.licence_url as string | null | undefined) ?? null,
           attribution: String(ds.attribution ?? ""), modifications: String(ds.modifications ?? "") };
}

/** A published student and its held-out rows from the two payloads `build_heldout_payloads.py`
 *  writes: the rows file names its weights file, so the pair cannot be mismatched by a caller. */
async function publishedStudentHeldOut(usecase: string, path: string): Promise<HeldOutCorpus | null> {
  const ds = await fetchJSON(path);
  const w = await fetchJSON(`/${String(ds.weights).replace(/^\/+/, "")}`);
  const names = ds.feature_names as string[];
  const raw = ds.samples as number[][];
  if (!raw?.length || !Array.isArray(w?.W)) return null;
  if (w.n_features !== names.length || raw[0].length !== names.length) {
    throw new Error(`${path}: ${raw[0].length} columns for a ${w.n_features}-feature model`);
  }
  const rows = raw.map((r) => preprocess(r, w.input?.preprocessing ?? [], w.input?.mean ?? [],
                                         w.input?.scale ?? []));
  const W = w.W as number[][], b = (w.b as number[]) ?? [];
  // One logit (finance): give class 0 an all-zero row, so argmax over [0, z] is the sign of z.
  const model = W.length === 1
    ? { W: [new Array(W[0].length).fill(0), W[0]], b: [0, b[0] ?? 0] }
    : { W, b };
  return {
    usecase, modelKey: String(ds.model_key ?? w.student_key ?? ""),
    modelLabel: `the published ${w.n_parameters}-parameter CKKS-linear student`,
    model, rows, labels: (ds.labels as number[]).map((v) => Number(v) | 0),
    classNames: (ds.class_names as string[]) ?? model.W.map((_, j) => `class ${j}`),
    plane: ds.plane ? planeFromColumns(raw, names, ds.plane) : null,
    corpus: String(ds.corpus ?? ""), split: String(ds.split ?? ""), fitNote: String(ds.fit_note ?? ""),
    licence: licenceOf(ds), expected: (ds.expected as Record<string, unknown>) ?? null,
  };
}

/**
 * The private-inference figure's rows for a use case: a model and rows it was never fitted on.
 *
 *   finance_aml  the published 115-parameter AMLworld student (`finance_aml.student`) on the 8,192
 *                test-window rows `finance/real_2d.json` draws, through its own z-scoring.
 *   medical_fl   the page's bundled WDBC regression on the 171 rows held out from its fit.
 *   telecom_ids  the published 64-parameter student (`telecom_ids.student`) on the 600 rows of its
 *                own held-out quarter.
 *   satellite_comms  the published 210-parameter student (`satellite_comms.opssat_student`) on the
 *                529 segments of the benchmark's test partition, through its quadratic expansion and
 *                the scaler its ONNX graph applies.
 *
 * `null` for every other use case: genomics' 30 reference-panel individuals are external rather than
 * a held-out split and keep their own path.
 */
export async function heldOutCorpus(usecase: string): Promise<HeldOutCorpus | null> {
  try {
    if (usecase === "finance_aml") return await publishedStudentHeldOut(usecase, "/data/finance/student_heldout.json");
    if (usecase === "telecom_ids") return await publishedStudentHeldOut(usecase, "/data/telecom/student_heldout.json");
    if (usecase === "satellite_comms") return await publishedStudentHeldOut(usecase, "/data/satellite/student_heldout.json");
    if (usecase === "medical_fl") {
      const ds = await fetchJSON("/data/medical/breast_cancer_sample.json");
      const h = ds.heldout;
      if (!h?.samples?.length) return null;
      const plane = ds.plane && Array.isArray(h.plane) && h.plane.length === h.samples.length
        ? { x: ds.plane.x as PlaneAxis, y: ds.plane.y as PlaneAxis, points: h.plane as [number, number][] }
        : null;
      return {
        usecase, modelKey: null,
        modelLabel: `the page's bundled ${(ds.names as string[]).length}-feature WDBC logistic regression`,
        model: ds.model, rows: (h.samples as number[][]).map((r) => r.slice(0, 8)),
        labels: (h.labels as number[]).map((v) => Number(v) | 0),
        classNames: (ds.class_names as string[]) ?? ["class 0", "class 1"],
        plane, corpus: "Wisconsin Diagnostic Breast Cancer (WDBC, UCI dataset 17)",
        split: String(h.split ?? h.note ?? ""), fitNote: String(ds.fit_note ?? ds.real_model ?? ""),
        licence: licenceOf(ds),
        expected: typeof h.plaintext_accuracy === "number" ? { plaintext_accuracy: h.plaintext_accuracy } : null,
      };
    }
  } catch (e) {
    console.warn("[realData] held-out corpus unavailable:", e);
  }
  return null;
}

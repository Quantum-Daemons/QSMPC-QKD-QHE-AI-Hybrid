// Datasets shipped to the CLIENT-SIDE demo (web/public/data/), so the demo scores real inputs with
// real models rather than sampling noise — matching the FULL deployment. Coverage: medical (Wisconsin
// breast-cancer, real measurements), genomics (published PGS000004 weights over real 1000 Genomes
// genotypes), finance (PaySim SIMULATOR output, i.e. synthetic rows) and satellite (real NASA
// SMAP/MSL telemetry). Only telecom stays fully synthetic — no network-attack corpus with an
// affirmative commercial grant exists (UNSW-NB15 needs the authors' agreement; CIC-IDS2017 states no
// licence at all) — and the demo logs that rather than quietly substituting. Tiny static JSON,
// fetched and cached once.
//
// These are BUNDLED SAMPLES, not the corpora the published models were trained on, and they are not
// uniformly commercial-OK: genomics in particular is display-only here and its derived weights are
// never published. Licensing prose lives in DATA_LICENSES.md and the machine-readable authority is
// src/qsh/models/registry.py (BUNDLED_DEMO_SAMPLES), which is what the demo actually prints via
// config.demoSample(). No licence summary is restated here on purpose: the one that used to sit at
// this line got four of five wrong (WDBC is CC BY 4.0, not public domain; PGS is per-score, not
// blanket CC-BY; IGSR grants no blanket licence; no provider states one for SMAP/MSL) and shipped
// that way. A stale copy reads as authoritative, which is worse than a pointer.

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
  values: number[];                          // the 8 demo features (REAL)
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
 * Returns REAL input + a REAL model for use cases backed by a commercial-OK dataset (medical, genomics),
 * or `null` for everything else (→ the caller keeps its synthetic features and logs that honestly).
 * `u` is the pipeline's seeded RNG so a run is deterministic.
 */
export async function realInput(usecase: string, u: () => number): Promise<RealInput | null> {
  try {
    if (usecase === "medical_fl") {
      // REAL Wisconsin Diagnostic Breast Cancer sample + a REAL logistic-regression model (8 features).
      const ds = await fetchJSON("/data/medical/breast_cancer_sample.json");
      const rows = ds.samples as number[][];
      const row = rows[Math.floor(u() * rows.length) % rows.length];
      return { values: row.slice(0, 8), names: ds.names, provenance: ds.provenance, model: ds.model,
               baseline: columnMeans(rows, 8) };
    }
    if (usecase === "genomics") {
      // REAL PGS-Catalog PGS000004 (PRS313) effect weights + REAL 1000 Genomes phase-3 genotypes
      // (Ensembl / IGSR, which grants NO blanket licence; PGS scoring files are licensed per score).
      // Real effect-allele dosages ship in `dosage_samples`. The full 313-variant score, sum_i w_i *
      // dosage_i, is computed IN PLAINTEXT below (`polygenic_risk_score`). What CKKS evaluates, and
      // what SHAP explains, is narrower: the top 8 of the 313 variants by |weight|, as the two-logit
      // model W = [-w/2, +w/2], b = 0, so the encrypted step is a SIGN TEST on an 8-variant partial
      // score, not the PRS. No synthetic sampling; real genotypes throughout.
      //
      // PRS313 since Round 31, and the payload is now GENERATED by
      // `scripts/data/build_prs_browser_payload.py`. It used to be a hand-built committed file with
      // three readers and no writer, so the browser's view of this use case could only be changed by
      // editing JSON and hoping it stayed consistent with the Python pipeline computing the same score.
      // Only 30 of 2,504 individuals ship: IGSR grants no blanket redistribution licence and
      // DATA_LICENSES section 4 justifies a slice of that size, not the full matrix.
      const prs = await fetchJSON("/data/genomics/pgs000004_prs313.json");
      const w = prs.weights as number[];
      // `keys` on PRS313, `rsids` on PRS77. PGS000004 publishes NO rsID for any of its 313 variants,
      // so its identifiers are `chr:pos:effect_allele`; 28% of those positions carry several rsIDs and
      // about 15% resolve to none, so inventing one would label a different variant's genotypes.
      const rs = (prs.keys ?? prs.rsids) as string[];
      const samples = prs.dosage_samples as number[][];
      const dosage = samples[Math.floor(u() * samples.length) % samples.length];   // a REAL 1000G individual
      const prsScore = w.reduce((a, wi, i) => a + wi * (dosage[i] ?? 0), 0);        // PRS over ALL 313 real SNPs
      const top = (prs.top8_idx as number[]) ?? w.map((_, i) => i)
        .sort((a, b) => Math.abs(w[b]) - Math.abs(w[a])).slice(0, 8);
      const values = top.map((i) => dosage[i] ?? 0);                               // top-8 real SNP dosages (features)
      const names = top.map((i) => rs[i] ?? `snp_${i}`);
      const wsel = top.map((i) => w[i]);                                           // their REAL effect weights
      const model = { W: [wsel.map((x) => -x / 2), wsel.map((x) => x / 2)], b: [0, 0] };  // one-vs-rest margin = PRS
      return {
        values, names, provenance: prs.provenance, model,
        metrics: { polygenic_risk_score: Number(prsScore.toFixed(4)) },
        // Population mean dosage at the same SNPs. The zero vector would mean "an individual carrying
        // no effect allele anywhere", which is not a typical genome and at several of these loci is
        // not present in the sample at all.
        //
        // Prefer the payload's `population_mean_dosage`, which the generator computes over ALL 2,504
        // individuals; the fallback averages only the 30 that ship. Both are means at the same loci,
        // but one is a population estimate and the other is an estimate from a redistribution-limited
        // slice, and the SHAP reference point should not be degraded by a licence constraint.
        baseline: (prs.population_mean_dosage as number[] | undefined)
          ? top.map((i) => (prs.population_mean_dosage as number[])[i] ?? 0)
          : top.map((i) => samples.reduce((a, d) => a + (d[i] ?? 0), 0) / Math.max(1, samples.length)),
      };
    }
    if (usecase === "finance_aml") {
      // PaySim SIMULATOR output - synthetic transactions (Kaggle ealaxi/paysim1, CC-BY-SA-4.0) - as 8
      // features with AML-style names that RELABEL PaySim columns (scripts/prep_real_kaggle_data.py):
      // device_risk is PaySim's origin-balance-inconsistency flag, geo_mismatch the destination not
      // credited, cross_border the TRANSFER/CASH_OUT type, velocity_24h a constant and amount_zscore a
      // copy of amount. The model is a logistic surrogate distilled from a gradient-boosting teacher,
      // both fitted on these bundled rows; it is NOT the AMLworld student the full deployment
      // encrypts. The SHAP attribution reflects this surrogate on simulator rows.
      const ds = await fetchJSON("/data/finance/paysim_sample.json");
      const rows = ds.samples as number[][];
      const idx = Math.floor(u() * rows.length) % rows.length;
      const row = rows[idx];                                           // a PaySim simulator row (synthetic)
      // The same transaction in raw units, for any consumer whose artifact declares `raw`. Indexed,
      // not re-drawn: a second random pick would hand the QCNN a different transaction from the one
      // the CKKS tier and the SHAP panel are describing, and the page would attribute one row's
      // explanation to another row's prediction.
      return { values: row.slice(0, 8), bySpace: rowBySpace(ds, idx, 8),
               names: ds.names, provenance: ds.provenance, model: ds.model,
               baseline: columnMeans(rows, 8) };
    }
  } catch (e) {
    console.warn("[realData] real dataset unavailable → synthetic fallback:", e);
  }
  return null;   // telecom (+ any load failure) → synthetic, logged by the caller
}

/** A real telemetry-anomaly verdict over a REAL NASA SMAP/MSL window + the real linear model, scored IN
 *  PLAINTEXT - a JS dot product, with no FHE or MPC (the satellite use case has no `private_inference`
 *  stage, so its anomaly score is surfaced at QSDC delivery; the CKKS version runs only in Python). */
/** Each cohort's effect-allele count at PGS000004's first variant, over the bundled 1000 Genomes
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
    const prs = await fetchJSON("/data/genomics/pgs000004_prs313.json");
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

export interface SatelliteAnomaly {
  provenance: string; anomalyScore: number; isAnomaly: boolean; channels: string[];
  /** The real linear model, the scored window and E[x] over the bundled windows.
   *  Returned so the run can compute exact linear SHAP for satellite the same way the three
   *  inference use cases do. The model here IS linear (a 2xN margin over a NASA telemetry window),
   *  so the method applies unchanged - no new or invented attribution is involved. */
  model: { W: number[][]; b: number[] };
  window: number[];
  baseline: number[];
  featureNames: string[];
}

export async function realSatelliteAnomaly(u: () => number): Promise<SatelliteAnomaly | null> {
  try {
    const ds = await fetchJSON("/data/satellite/telemetry.json");
    const windows = ds.windows as number[][];
    const w = windows[Math.floor(u() * windows.length) % windows.length];   // a REAL NASA telemetry window
    const W = ds.model.W as number[][]; const b = ds.model.b as number[];
    const margin = w.reduce((a, v, i) => a + (W[1][i] - W[0][i]) * v, 0) + (b[1] - b[0]);  // real linear model
    const score = 1 / (1 + Math.exp(-margin));
    const names = (ds.names as string[] | undefined) ?? w.map((_, i) => `f${i}`);
    return { provenance: ds.provenance, anomalyScore: Number(score.toFixed(4)),
             isAnomaly: score > 0.5, channels: ds.channel_names,
             model: { W, b }, window: w,
             // E[x] over the bundled windows, matching the reference point the other use cases use.
             baseline: columnMeans(windows, w.length), featureNames: names };
  } catch (e) {
    console.warn("[realData] satellite real telemetry unavailable → representative:", e);
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
 * per use case - `row.slice(0, 8)` for the tabular pages, the top-8 PRS SNP indices for genomics -
 * so the corpus is built from the same source with the same selection, in the same file.
 *
 * Returns `null` where the page has no corpus of its own, rather than an empty array, so a caller
 * cannot mistake "no corpus" for "a corpus with nothing in it".
 */
export async function realInputCorpus(usecase: string): Promise<number[][] | null> {
  try {
    if (usecase === "medical_fl") {
      const ds = await fetchJSON("/data/medical/breast_cancer_sample.json");
      const rows = ds.samples as number[][] | undefined;
      return rows?.length ? rows.map((r) => r.slice(0, 8)) : null;
    }
    if (usecase === "finance_aml") {
      const ds = await fetchJSON("/data/finance/paysim_sample.json");
      const rows = ds.samples as number[][] | undefined;
      return rows?.length ? rows.map((r) => r.slice(0, 8)) : null;
    }
    if (usecase === "genomics") {
      const prs = await fetchJSON("/data/genomics/pgs000004_prs313.json");
      const w = prs.weights as number[];
      const samples = prs.dosage_samples as number[][] | undefined;
      if (!samples?.length) return null;
      // The SAME top-8 selection `realInput` uses, including its fallback, so the two cannot diverge.
      const top = (prs.top8_idx as number[]) ?? w.map((_, i) => i)
        .sort((a, b) => Math.abs(w[b]) - Math.abs(w[a])).slice(0, 8);
      return samples.map((d) => top.map((i) => d[i] ?? 0));
    }
  } catch (e) {
    console.warn("[realData] corpus unavailable:", e);
  }
  return null;
}

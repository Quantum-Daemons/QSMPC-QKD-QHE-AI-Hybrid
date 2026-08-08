// Datasets shipped to the CLIENT-SIDE demo (web/public/data/), so the demo scores real inputs with
// real models rather than sampling noise — matching the FULL deployment. Coverage: medical (Wisconsin
// breast-cancer, real measurements), genomics (published PGS000001 weights over real 1000 Genomes
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

export interface RealInput {
  values: number[];                          // the 8 demo features (REAL)
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
      // REAL PGS-Catalog PGS000001 effect weights + REAL 1000 Genomes phase-3 genotypes (Ensembl / IGSR,
      // which grants NO blanket licence; PGS scoring files are licensed per score). A real
      // individual's effect-allele dosages are shipped in
      // `dosage_samples`; the demo PRS = Σ wᵢ·dosageᵢ is computed over genuine human genotypes, and the
      // REAL model for SHAP = the top-8 SNPs' real PGS weights → a real PRS readout (no synthetic sampling).
      const prs = await fetchJSON("/data/genomics/pgs000001_prs.json");
      const w = prs.weights as number[];
      const rs = prs.rsids as string[];
      const samples = prs.dosage_samples as number[][];
      const dosage = samples[Math.floor(u() * samples.length) % samples.length];   // a REAL 1000G individual
      const prsScore = w.reduce((a, wi, i) => a + wi * (dosage[i] ?? 0), 0);        // PRS over ALL 77 real SNPs
      const top = (prs.top8_idx as number[]) ?? w.map((_, i) => i)
        .sort((a, b) => Math.abs(w[b]) - Math.abs(w[a])).slice(0, 8);
      const values = top.map((i) => dosage[i] ?? 0);                               // top-8 real SNP dosages (features)
      const names = top.map((i) => rs[i] ?? `snp_${i}`);
      const wsel = top.map((i) => w[i]);                                           // their REAL effect weights
      const model = { W: [wsel.map((x) => -x / 2), wsel.map((x) => x / 2)], b: [0, 0] };  // one-vs-rest margin = PRS
      return {
        values, names, provenance: prs.provenance, model,
        metrics: { polygenic_risk_score: Number(prsScore.toFixed(4)) },
        // Population mean dosage at the same SNPs, over the shipped 1000 Genomes individuals. The
        // zero vector would mean "an individual carrying no effect allele anywhere", which is not a
        // typical genome and is not even present in the sample at several of these loci.
        baseline: top.map((i) => samples.reduce((a, d) => a + (d[i] ?? 0), 0) / Math.max(1, samples.length)),
      };
    }
    if (usecase === "finance_aml") {
      // PaySim SIMULATOR output — synthetic transactions (Kaggle ealaxi/paysim1, CC-BY-SA-4.0) → the 8 AML features +
      // a REAL distilled CKKS-linear model (XGBoost → logistic surrogate), matching the full deployment's
      // encrypted inference. The SHAP attribution then reflects the real distilled fraud model on real data.
      const ds = await fetchJSON("/data/finance/paysim_sample.json");
      const rows = ds.samples as number[][];
      const row = rows[Math.floor(u() * rows.length) % rows.length];   // a REAL PaySim transaction
      return { values: row.slice(0, 8), names: ds.names, provenance: ds.provenance, model: ds.model,
               baseline: columnMeans(rows, 8) };
    }
  } catch (e) {
    console.warn("[realData] real dataset unavailable → synthetic fallback:", e);
  }
  return null;   // telecom (+ any load failure) → synthetic, logged by the caller
}

/** A real telemetry-anomaly verdict over a REAL NASA SMAP/MSL window + the real linear model (the
 *  satellite use case has no `private_inference` stage, so its anomaly score is surfaced at QSDC delivery). */
export interface SatelliteAnomaly { provenance: string; anomalyScore: number; isAnomaly: boolean; channels: string[]; }

export async function realSatelliteAnomaly(u: () => number): Promise<SatelliteAnomaly | null> {
  try {
    const ds = await fetchJSON("/data/satellite/telemetry.json");
    const windows = ds.windows as number[][];
    const w = windows[Math.floor(u() * windows.length) % windows.length];   // a REAL NASA telemetry window
    const W = ds.model.W as number[][]; const b = ds.model.b as number[];
    const margin = w.reduce((a, v, i) => a + (W[1][i] - W[0][i]) * v, 0) + (b[1] - b[0]);  // real linear model
    const score = 1 / (1 + Math.exp(-margin));
    return { provenance: ds.provenance, anomalyScore: Number(score.toFixed(4)),
             isAnomaly: score > 0.5, channels: ds.channel_names };
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

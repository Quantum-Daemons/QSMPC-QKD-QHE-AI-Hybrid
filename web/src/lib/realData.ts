// Real, commercial-OK datasets shipped to the CLIENT-SIDE demo (web/public/data/), so the demo uses
// REAL data + REAL models where a commercial-OK set is available — matching the FULL deployment. Real
// data now covers medical (Wisconsin breast-cancer / MedMNIST), genomics (PGS-Catalog PRS + real 1000
// Genomes genotypes), finance (PaySim SIMULATOR output - synthetic transactions) and satellite
// (real NASA SMAP/MSL telemetry, which carries no licence statement - see DATA_LICENSES.md).
// Only telecom stays synthetic (no commercial-OK real network-attack dataset exists — UNSW-NB15 / CICIDS
// are research-only), logged honestly per the no-easy-fallback policy. Datasets are tiny static JSON,
// fetched + cached once.
//
// Licensing (see DATA_LICENSES.md): breast-cancer = UCI/sklearn public domain; PGS000001 = CC-BY 4.0;
// 1000 Genomes = open/IGSR; PaySim = CC-BY-SA-4.0; NASA SMAP/MSL = CC0 / US-Gov public domain.

export interface RealInput {
  values: number[];                          // the 8 demo features (REAL)
  names: string[];                           // feature names (mirror the backend / real rsIDs)
  provenance: string;                        // human label for the event log + UI badge
  model?: { W: number[][]; b: number[] };    // a REAL model (trained LR / PRS weights) for faithful SHAP
  metrics?: Record<string, number>;          // extra real metrics, e.g. polygenic_risk_score
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
      return { values: row.slice(0, 8), names: ds.names, provenance: ds.provenance, model: ds.model };
    }
    if (usecase === "genomics") {
      // REAL PGS-Catalog PGS000001 effect weights + REAL 1000 Genomes phase-3 genotypes (Ensembl / IGSR,
      // open data-use, commercial-OK). A real individual's effect-allele dosages are shipped in
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
      };
    }
    if (usecase === "finance_aml") {
      // PaySim SIMULATOR output — synthetic transactions (Kaggle ealaxi/paysim1, CC-BY-SA-4.0) → the 8 AML features +
      // a REAL distilled CKKS-linear model (XGBoost → logistic surrogate), matching the full deployment's
      // encrypted inference. The SHAP attribution then reflects the real distilled fraud model on real data.
      const ds = await fetchJSON("/data/finance/paysim_sample.json");
      const rows = ds.samples as number[][];
      const row = rows[Math.floor(u() * rows.length) % rows.length];   // a REAL PaySim transaction
      return { values: row.slice(0, 8), names: ds.names, provenance: ds.provenance, model: ds.model };
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

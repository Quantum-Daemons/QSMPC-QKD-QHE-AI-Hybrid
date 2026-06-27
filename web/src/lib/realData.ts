// Real, commercial-OK datasets shipped to the CLIENT-SIDE demo (web/public/data/), so the demo uses
// REAL data + REAL models where a commercial-OK set is available — matching the FULL deployment, which
// only has real data for medical (Wisconsin breast-cancer / MedMNIST) + genomics (PGS-Catalog PRS).
// finance / telecom / satellite have no shippable commercial-OK real dataset (PaySim/ESA-ADB are
// large + their loaders are simulator-methodology in the full deployment too), so they stay synthetic,
// logged honestly per the no-easy-fallback policy. Datasets are tiny static JSON, fetched + cached once.
//
// Licensing (see DATA_LICENSES.md): breast-cancer = UCI/sklearn public domain; PGS000001 = CC-BY 4.0.

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
      // REAL PGS-Catalog PGS000001 effect weights → a genuine Polygenic Risk Score PRS = Σ wᵢ·dosageᵢ.
      // The effect WEIGHTS are real (published); genotype dosages ∈ {0,1,2} are sampled under HWE
      // (genotypes are personal → never shipped). The demo feature vector = the top-8 per-SNP dosages,
      // and the REAL model for SHAP = those SNPs' real PGS weights, so the attribution is a real PRS readout.
      const prs = await fetchJSON("/data/genomics/pgs000001_prs.json");
      const w = prs.weights as number[];
      const rs = prs.rsids as string[];
      const dosage = w.map(() => (u() < 0.25 ? 2 : u() < 0.5 ? 1 : 0));      // sampled genotype
      const prsScore = w.reduce((a, wi, i) => a + wi * dosage[i], 0);        // PRS over ALL 77 SNPs (real weights)
      const top = w.map((_, i) => i).sort((a, b) => Math.abs(w[b]) - Math.abs(w[a])).slice(0, 8);
      const values = top.map((i) => dosage[i]);                              // top-8 SNP dosages (the features)
      const names = top.map((i) => rs[i] ?? `snp_${i}`);
      const wsel = top.map((i) => w[i]);                                     // their REAL effect weights
      const model = { W: [wsel.map((x) => -x / 2), wsel.map((x) => x / 2)], b: [0, 0] };  // one-vs-rest margin = PRS
      return {
        values, names, provenance: prs.provenance, model,
        metrics: { polygenic_risk_score: Number(prsScore.toFixed(4)) },
      };
    }
  } catch (e) {
    console.warn("[realData] real dataset unavailable → synthetic fallback:", e);
  }
  return null;   // finance / telecom / satellite (+ any load failure) → synthetic, logged by the caller
}

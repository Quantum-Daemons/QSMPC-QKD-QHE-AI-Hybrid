// Client-side OptiQKD mirror of src/qsh/qkd/optimize.py — a measured, gradient-free coordinate ascent over
// the two-intensity decoy-state + efficient-BB84 SKR model (Ma-Qi-Zhao-Lo 2005; the analytic expectation of
// the backend Monte-Carlo), or the entanglement-based E91/CHSH model. Reports the SKR improvement + a
// convergence profile vs the default-parameter baseline on the SAME simulated channel — a simulation, not
// deployed QKD. Search space + defaults + protocol come from config (config.defaults.json / /api/config).
import { qkdCfg } from "./config";
import { ChannelParams, h2, intrinsicQber, transmittance } from "./quantumSim";

export interface OptiQKDParams { muSignal: number; muDecoy: number; basisProb: number; ppThreshold: number; }
export interface OptiQKDResult {
  bestParams: OptiQKDParams; bestSkr: number; baselineSkr: number;
  /** Percentage gain over the default parameters, or **null** when the baseline SKR is 0 -- a dead
   *  link has no meaningful denominator, and reporting +100% there contradicted secure_key_bits. */
  improvementPct: number | null;
  /** True when the un-optimised channel already yields no secure key (e.g. a high Eve fraction). */
  baselineIsDead: boolean;
  bestQber: number; baselineQber: number; nEvals: number; method: string; latencyMs: number;
  convergenceEvals: number; convergenceCurve: number[];
}
type Model = (p: ChannelParams, prm: OptiQKDParams, fEC?: number) => { skr: number; qber: number };

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

// Two-intensity decoy-state SKR per pulse (Ma-Qi-Zhao-Lo 2005): a signal (mu) + decoy (nu) intensity pair
// lower-bounds the single-photon yield Y1 and upper-bounds the single-photon error e1; the GLLP rate then
// keeps only the secure single-photon contribution. mu_decoy genuinely moves the rate via the Y1/e1 bounds.
const skrModel: Model = (p, prm, fEC = 1.16) => {
  const eta = transmittance(p);
  const mu = Math.max(1e-3, prm.muSignal);
  const nu = Math.min(Math.max(1e-4, prm.muDecoy), mu * 0.9);   // decoy strictly weaker than the signal
  const y0 = 2 * p.darkCount;                                   // background yield (two detectors)
  const eDet = intrinsicQber(p);
  const siftRatio = prm.basisProb ** 2 + (1 - prm.basisProb) ** 2;
  const e0 = 0.5;
  const qMu = y0 + 1 - Math.exp(-eta * mu);
  const eMu = qMu > 0 ? (e0 * y0 + eDet * (1 - Math.exp(-eta * mu))) / qMu : 0.5;
  if (eMu > prm.ppThreshold || qMu <= 0 || mu <= nu) return { skr: 0, qber: eMu };
  const qNu = y0 + 1 - Math.exp(-eta * nu);
  const eNu = qNu > 0 ? (e0 * y0 + eDet * (1 - Math.exp(-eta * nu))) / qNu : 0.5;
  const y1 = Math.max(0, (mu / (mu * nu - nu * nu)) * (
    qNu * Math.exp(nu) - (nu * nu) / (mu * mu) * qMu * Math.exp(mu) - (mu * mu - nu * nu) / (mu * mu) * y0));
  const q1 = y1 * mu * Math.exp(-mu);
  const e1 = y1 > 0 ? Math.min(0.5, Math.max(0, (eNu * qNu * Math.exp(nu) - e0 * y0) / (y1 * nu))) : 0.5;
  const r = q1 * (1 - h2(e1)) - fEC * qMu * h2(eMu);           // GLLP single-photon rate per pulse
  return { skr: Math.max(0, r) * siftRatio, qber: eMu };
};

// E91 (entanglement-based, CHSH) SKR mirror of qkd/e91.py: a Werner-state visibility V (degraded by the
// channel) gives S = 2*sqrt(2)*V and QBER = (1-V)/2; a Bell violation (S>2) is required for a key. Only
// basis_prob + pp_threshold apply here (the decoy intensities do not).
const e91Skr: Model = (p, prm, fEC = 1.16) => {
  const eta = transmittance(p);
  const vEff = clamp(qkdCfg().e91_visibility * (1 - intrinsicQber(p)), 0, 1);
  const s = 2 * Math.SQRT2 * vEff;
  const qber = Math.min(0.5, (1 - vEff) / 2);
  const siftRatio = prm.basisProb ** 2 + (1 - prm.basisProb) ** 2;
  if (s <= 2 || qber > prm.ppThreshold) return { skr: 0, qber };
  const secureFraction = Math.max(0, 1 - h2(qber) - fEC * h2(qber));
  return { skr: secureFraction * eta * siftRatio, qber };
};

export function optimizeQKD(channel: ChannelParams, protocol?: string, nSamples?: number): OptiQKDResult {
  const t0 = performance.now();
  const q = qkdCfg();
  const model: Model = (protocol ?? q.protocol) === "e91" ? e91Skr : skrModel;
  // sample budget (config.qkd_opt_samples) sets the coordinate-ascent depth: ~8 evaluations
  // (4 params × 2 directions) per round, so a larger budget explores more + tightens convergence.
  const budget = Math.max(4, Math.round(nSamples ?? q.optimize_n_samples));
  const rounds = Math.max(1, Math.round(budget / 8));
  const lo: OptiQKDParams = { muSignal: q.opt_mu_signal_min, muDecoy: q.opt_mu_decoy_min, basisProb: q.opt_basis_prob_min, ppThreshold: q.opt_pp_threshold_min };
  const hi: OptiQKDParams = { muSignal: q.opt_mu_signal_max, muDecoy: q.opt_mu_decoy_max, basisProb: q.opt_basis_prob_max, ppThreshold: q.opt_pp_threshold_max };
  const keys: (keyof OptiQKDParams)[] = ["muSignal", "muDecoy", "basisProb", "ppThreshold"];
  const b0 = model(channel, { muSignal: q.mu_signal, muDecoy: q.mu_decoy, basisProb: q.basis_prob, ppThreshold: q.pp_threshold });
  let nEvals = 1;
  const curve: number[] = [];                                  // running-best SKR after each evaluation
  const record = (v: number) => { curve.push(curve.length ? Math.max(curve[curve.length - 1], v) : v); };

  let best: OptiQKDParams = { muSignal: (lo.muSignal + hi.muSignal) / 2, muDecoy: (lo.muDecoy + hi.muDecoy) / 2, basisProb: (lo.basisProb + hi.basisProb) / 2, ppThreshold: (lo.ppThreshold + hi.ppThreshold) / 2 };
  let bestVal = model(channel, best).skr; nEvals++; record(bestVal);
  const step: OptiQKDParams = { muSignal: (hi.muSignal - lo.muSignal) / 8, muDecoy: (hi.muDecoy - lo.muDecoy) / 8, basisProb: (hi.basisProb - lo.basisProb) / 8, ppThreshold: (hi.ppThreshold - lo.ppThreshold) / 8 };
  for (let r = 0; r < rounds; r++) {
    let improved = false;
    for (const k of keys) {
      for (const d of [step[k], -step[k]]) {
        const cand: OptiQKDParams = { ...best, [k]: clamp(best[k] + d, lo[k], hi[k]) };
        const v = model(channel, cand).skr; nEvals++;
        if (v > bestVal) { bestVal = v; best = cand; improved = true; }
        record(bestVal);
      }
    }
    for (const k of keys) step[k] *= 0.6;
    if (!improved) break;
  }
  const bestRes = model(channel, best);
  // A dead link has no percentage improvement to report. This used to fall to `bestRes.skr > 0 ? 100 : 0`,
  // so a high Eve fraction -- baseline SKR 0 -- printed a triumphant "SKR +100%" beside the same run's
  // secure_key_bits of 0. When the baseline is zero the honest answer is "undefined", and callers must
  // say that rather than render a number.
  const baselineIsDead = !(b0.skr > 1e-12);
  const improvement = baselineIsDead ? null : (100 * (bestRes.skr - b0.skr)) / b0.skr;
  const final = curve.length ? curve[curve.length - 1] : bestRes.skr;
  const convergenceEvals = Math.max(1, curve.findIndex((v) => v >= 0.95 * final) + 1);   // evals to 95% of final
  return {
    bestParams: best, bestSkr: bestRes.skr, baselineSkr: b0.skr, improvementPct: improvement,
    baselineIsDead,
    bestQber: bestRes.qber, baselineQber: b0.qber, nEvals, method: "coordinate",
    latencyMs: performance.now() - t0, convergenceEvals, convergenceCurve: curve,
  };
}

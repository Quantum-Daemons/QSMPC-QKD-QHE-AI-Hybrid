// Client-side OptiQKD mirror of src/qsh/qkd/optimize.py — a measured, gradient-free coordinate ascent
// over the decoy-state + efficient-BB84 SKR model (the analytic expectation of the backend Monte-Carlo).
// Reports the SKR improvement vs the default-parameter baseline on the SAME simulated channel — a
// simulation, not deployed QKD. Search space + defaults come from config (config.defaults.json / /api/config).
import { qkdCfg } from "./config";
import { ChannelParams, h2, intrinsicQber, transmittance } from "./quantumSim";

export interface OptiQKDParams { muSignal: number; basisProb: number; ppThreshold: number; }
export interface OptiQKDResult {
  bestParams: OptiQKDParams; bestSkr: number; baselineSkr: number; improvementPct: number;
  bestQber: number; baselineQber: number; nEvals: number; method: string; latencyMs: number;
}

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

// Analytic decoy-state SKR: n_sift/n = P(>=1 photon detected) * biased-basis sift ratio; multi-photon
// (>=2) pulses add a QBER penalty; GLLP secure fraction; abort above the post-processing threshold.
function skrModel(p: ChannelParams, prm: OptiQKDParams, fEC = 1.16): { skr: number; qber: number } {
  const T = transmittance(p);
  const detectProb = 1 - Math.exp(-prm.muSignal * T);
  const siftRatio = prm.basisProb ** 2 + (1 - prm.basisProb) ** 2;
  const mpFrac = 1 - (1 + prm.muSignal) * Math.exp(-prm.muSignal);
  const qber = Math.min(0.5, intrinsicQber(p) + 0.5 * mpFrac);
  if (qber > prm.ppThreshold) return { skr: 0, qber };
  const secureFraction = Math.max(0, 1 - h2(qber) - fEC * h2(qber));
  return { skr: secureFraction * detectProb * siftRatio, qber };
}

export function optimizeQKD(channel: ChannelParams): OptiQKDResult {
  const t0 = performance.now();
  const q = qkdCfg();
  const lo: OptiQKDParams = { muSignal: q.opt_mu_signal_min, basisProb: q.opt_basis_prob_min, ppThreshold: q.opt_pp_threshold_min };
  const hi: OptiQKDParams = { muSignal: q.opt_mu_signal_max, basisProb: q.opt_basis_prob_max, ppThreshold: q.opt_pp_threshold_max };
  const keys: (keyof OptiQKDParams)[] = ["muSignal", "basisProb", "ppThreshold"];
  const b0 = skrModel(channel, { muSignal: q.mu_signal, basisProb: q.basis_prob, ppThreshold: q.pp_threshold });
  let nEvals = 1;

  let best: OptiQKDParams = { muSignal: (lo.muSignal + hi.muSignal) / 2, basisProb: (lo.basisProb + hi.basisProb) / 2, ppThreshold: (lo.ppThreshold + hi.ppThreshold) / 2 };
  let bestVal = skrModel(channel, best).skr; nEvals++;
  const step: OptiQKDParams = { muSignal: (hi.muSignal - lo.muSignal) / 8, basisProb: (hi.basisProb - lo.basisProb) / 8, ppThreshold: (hi.ppThreshold - lo.ppThreshold) / 8 };
  for (let r = 0; r < 6; r++) {
    let improved = false;
    for (const k of keys) {
      for (const d of [step[k], -step[k]]) {
        const cand: OptiQKDParams = { ...best, [k]: clamp(best[k] + d, lo[k], hi[k]) };
        const v = skrModel(channel, cand).skr; nEvals++;
        if (v > bestVal) { bestVal = v; best = cand; improved = true; }
      }
    }
    for (const k of keys) step[k] *= 0.6;
    if (!improved) break;
  }
  const bestRes = skrModel(channel, best);
  const improvement = b0.skr > 1e-12 ? (100 * (bestRes.skr - b0.skr)) / b0.skr : (bestRes.skr > 0 ? 100 : 0);
  return {
    bestParams: best, bestSkr: bestRes.skr, baselineSkr: b0.skr, improvementPct: improvement,
    bestQber: bestRes.qber, baselineQber: b0.qber, nEvals, method: "coordinate",
    latencyMs: performance.now() - t0,
  };
}

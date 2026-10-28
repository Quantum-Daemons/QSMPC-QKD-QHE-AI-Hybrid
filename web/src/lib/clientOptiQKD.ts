// Client-side OptiQKD mirror of src/qsh/qkd/optimize.py — a measured, gradient-free coordinate ascent over
// the two-intensity decoy-state + efficient-BB84 SKR model (Ma-Qi-Zhao-Lo 2005; the analytic expectation of
// the backend Monte-Carlo), or the entanglement-based E91/CHSH model. Reports the SKR improvement + a
// convergence profile vs the default-parameter baseline on the SAME simulated channel — a simulation, not
// deployed QKD. Search space + defaults + protocol come from config (config.defaults.json / /api/config).
import { qkdCfg } from "./config";
import { ChannelParams, clickMix, deadTimeFactor, decoySecureFraction, eDetForBound, h2, intrinsicQber, sourceOf,
  transmittance } from "./quantumSim";

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
// f_EC from config, for the same reason as `quantumSim`. Leaving the SURROGATE on a literal
// while wiring `bb84`/`e91` would be the worst of the three states: the reported skr and the
// optimiser's own baseline would use different f_EC, silently corrupting
// `qkd_skr_improvement_pct` - and invisibly, for as long as config equals the literal.
const skrModel: Model = (p, prm, fEC = qkdCfg().error_correction_eff) => {
  const eta = transmittance(p);
  const mu = Math.max(1e-3, prm.muSignal);
  const nu = Math.min(Math.max(1e-4, prm.muDecoy), mu * 0.9);   // decoy strictly weaker than the signal
  const y0 = 2 * p.darkCount;                                   // background yield (two detectors)
  // `eDetForBound`, NOT `intrinsicQber`. THIS FILE IS THE FOURTH IMPLEMENTATION OF THE DECOY
  // ALGEBRA and commit 3c63ea8 named only three - `bb84.py`, `quantumSim.ts`, `check_qkd.mjs` - in
  // the very commit whose subject is that a fix reached one language. A grep for `e0 * y0` returns
  // four files; three of them were corrected.
  //
  // Measured gap while it was left behind: the surrogate baseline read 0.001966695848615933 on
  // fibre against `bb84()`'s 0.001976547318720314 (0.498% low) and 4.793756674e-5 against
  // 5.552428392e-5 on the free-space downlink (13.664% low), and both numbers are printed in the
  // same stage - `qkd_surrogate_skr_default` beside `skr_per_pulse`.
  const eDet = eDetForBound(p);
  const siftRatio = prm.basisProb ** 2 + (1 - prm.basisProb) ** 2;
  // THE SHARED BOUND, not a fifth copy of its algebra. This file carried its own transcription and
  // was, by its own comment above, the copy a fix once missed. Round 75 changed the estimator to
  // Ma et al.'s one-decoy Eq. (41); calling `decoySecureFraction` is what makes that one change.
  // The vacuum flag and the state mix are the LINK's (Round 77): the search tunes intensities, not
  // whether the source sends a vacuum state, so a Micius run keeps its Vacuum+Weak bound.
  const src = sourceOf(p);
  const { perPulse, eMu } = decoySecureFraction(eta, mu, nu, eDet, y0, fEC, siftRatio, src.vacuum);
  if (eMu > prm.ppThreshold) return { skr: 0, qber: eMu };
  // The dead time, as `bb84()` charges it (Round 76). Without it the optimiser would chase a brighter
  // signal into detectors that cannot keep up: at 1 GHz the factor falls as mu rises.
  const cDt = deadTimeFactor(p, clickMix(src, y0 + 1 - Math.exp(-eta * mu), y0 + 1 - Math.exp(-eta * nu), y0));
  // Per pulse sent (Round 78), as `bb84Rate` counts it: the signal share carries the key.
  return { skr: perPulse * cDt * src.pSignal, qber: eMu };
};

// E91 (entanglement-based, CHSH) SKR mirror of qkd/e91.py: a Werner-state visibility V (degraded by the
// channel; with an eavesdropper the state is a Werner/intercept-resend mixture, not a Werner state) gives
// S = 2*sqrt(2)*V and QBER = (1-V)/2; a Bell violation (S>2) is required for a key, though the rate is
// already zero before S reaches 2 at the configured fEC. Only
// basis_prob + pp_threshold apply here (the decoy intensities do not).
const e91Skr: Model = (p, prm, fEC = qkdCfg().error_correction_eff) => {
  const eta = transmittance(p);
  // (1 - 2e), NOT (1 - e). A channel that flips a bit with probability e has visibility (1 - 2e):
  // visibility spans [-1, 1] of correlation, an error probability spans [0, 1/2] of QBER, and the
  // factor of two between them is exactly what went missing. Understated QBER by 24.8% at the
  // shipped channel, always in the demo's favour.
  //
  // Round 53 fixed `qkd/e91.py` and NOT these two, which is the ninth instance of one fix reaching
  // one language - and the worst-directed one yet, because this demo is CLIENT-SIDE ONLY. The Python
  // copy does not run for a visitor; these do. Checked against independent composition, which needs
  // no visibility algebra: V = 0.97, e = 0.015200 gives 0.029744 both ways, agreeing to 0.000%.
  const vEff = clamp(qkdCfg().e91_visibility * (1 - 2 * intrinsicQber(p)), 0, 1);
  const s = 2 * Math.SQRT2 * vEff;
  const qber = Math.min(0.5, (1 - vEff) / 2);
  const siftRatio = prm.basisProb ** 2 + (1 - prm.basisProb) ** 2;
  if (s <= 2 || qber > prm.ppThreshold) return { skr: 0, qber };
  const secureFraction = Math.max(0, 1 - h2(qber) - fEC * h2(qber));
  return { skr: secureFraction * deadTimeFactor(p, eta + 2 * p.darkCount) * eta * siftRatio, qber };
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
  // The baseline is the link's OWN source: on the satellite, Micius's 0.8 / 0.1 (Round 77). It was the
  // fibre page's finite-key optimum on every link, so the satellite's "improvement" was measured
  // against intensities the satellite never ran. Twin of `optimize_qkd(baseline=...)`.
  const own = sourceOf(channel);
  const b0 = model(channel, { muSignal: own.mu, muDecoy: own.nu, basisProb: q.basis_prob, ppThreshold: q.pp_threshold });
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

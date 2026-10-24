// Pure-TypeScript quantum-channel + small-circuit simulator for the client-side demo.
// Mirrors the Python qkd/channel.py + qkd/bb84.py analytic models (no WASM build needed)
// so the in-browser BB84 / QBER / secure-key-rate numbers track the backend.
import { qkdCfg } from "./config";

// Deterministic seeded RNG (mulberry32) so demo runs are reproducible per seed.
export function rng(seed: number): () => number {
  let a = (seed >>> 0) || 1;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Standard-normal sample via Box–Muller from a seeded uniform generator.
function gauss(u: () => number): number {
  const a = Math.max(1e-12, u()), b = u();
  return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
}

export function h2(x: number): number {
  if (x <= 0 || x >= 1) return 0;
  return -x * Math.log2(x) - (1 - x) * Math.log2(1 - x);
}

/** QSDC secrecy capacity, Qi et al., Light Sci. Appl. 8, 22 (2019), Eq. (8).
 *
 *      Cs = 1 - h(e) - g * h(e_x + e_z)
 *
 *  EXTRACTED SO IT CAN BE CHECKED ACROSS LANGUAGES. It lived inline in the `qsdc_telemetry` arm of a
 *  `switch` inside a generator in `clientCompute.ts`, which cannot be imported, so nothing could
 *  compare it against `src/qsh/qkd/qsdc.py`. This project has now shipped THREE corrections to this
 *  one expression - the 0.11 threshold, the omitted detector efficiency in `g`, and the pooled
 *  entropy argument - and each had to be applied by hand in two files, with no gate able to notice
 *  if the second application were forgotten. `tests/test_qsdc_agrees_across_languages.py` pins it.
 *
 *  The second argument is the SUM of the per-basis check error rates, not a pooled rate. Eq. (5)
 *  gives Eve's term as `h(xi)`, and at the maximising `p = 1/2` xi collapses to exactly `e_x + e_z`;
 *  the paper's own `I(A:E) = g*Q_Bob*h(e_x+e_z) = 9.1e-4` is reproduced only by `h(0.016)`, at
 *  9.1248e-4, against 4.08e-4 for `h(0.006)`. Doubling one measured rate is sound here only because
 *  nothing in this channel model distinguishes the bases. */
export function qsdcSecrecyCapacity(e: number, g: number): number {
  return 1 - h2(e) - g * h2(Math.min(0.5, 2 * e));
}

/** A free-space link at the given range, fixed losses and eavesdropper fraction. */
export function freeSpaceLink(o: { distanceKm: number; atmosphericDb: number; pointingDb: number;
                                   eveFraction: number }): ChannelParams {
  const fs = FREE_SPACE.freeSpace!;
  return { ...FREE_SPACE, distanceKm: o.distanceKm, eveFraction: o.eveFraction,
    freeSpace: { ...fs, atmosphericDb: o.atmosphericDb, pointingDb: o.pointingDb } };
}

/** ONE QSDC link evaluation - error rate, Eve's gain g (detector efficiency included, Qi et al.
 *  2019) and secrecy capacity - shared by the satellite stage and every tooltip that quotes a C_s.
 *  Four quoted figures had drifted: written under the pooled-entropy formula without detector
 *  efficiency, they still printed (+0.4944, -0.6283, -14.6474, -5.2033) after both corrections.
 *  A number computed here cannot disagree with what a run computes. */
export function qsdcLink(ch: ChannelParams, etaD = qkdCfg().free_space_detector_efficiency) {
  const lossDb = channelLossDb(ch) - 10 * Math.log10(etaD);
  const g = 10 ** (lossDb / 10);
  const e = intrinsicQber(ch);
  return { e, g, lossDb, cs: qsdcSecrecyCapacity(e, g) };
}

export interface ChannelParams {
  distanceKm: number;
  /** Fibre only. A free-space link is not attenuated per kilometre - see `freeSpace` below. */
  lossDbPerKm: number;
  detectorEff: number;
  darkCount: number;
  misalignment: number;
  eveFraction: number;
  /** Present on the satellite downlink, absent on fibre. Its presence selects the loss model. */
  freeSpace?: {
    beamDivergenceUrad: number;
    rxApertureM: number;
    atmosphericDb: number;
    pointingDb: number;
  };
}

// Both channels come from config, not from literals here.
//
// They used to be plain objects, and every call site spread them while overriding one or two fields -
// so the remaining constants could not be changed without editing this file. That shape had already
// produced a real bug on the fibre side: config and literal agreed numerically, so editing
// `config.yaml` moved the Python channel model and left the browser on the constant, with nothing
// anywhere to indicate the two had diverged. The free-space channel had it worse - it had no config
// counterpart at all, so four of its six physical constants were unreachable from configuration.
//
// Read at module load from the statically-imported defaults, so there is no async gap and no second
// source of truth. `eveFraction` stays 0 here because it is a per-run control, not channel physics:
// the panel supplies it.
export const FIBER: ChannelParams = {
  distanceKm: qkdCfg().distance_km,
  lossDbPerKm: qkdCfg().fiber_loss_db_per_km,
  detectorEff: qkdCfg().detector_efficiency,
  darkCount: qkdCfg().dark_count_prob,
  misalignment: qkdCfg().misalignment,
  eveFraction: 0,
};

export const FREE_SPACE: ChannelParams = {
  distanceKm: qkdCfg().free_space_distance_km,
  lossDbPerKm: 0,                       // unused: `freeSpace` below selects the diffraction model
  detectorEff: qkdCfg().free_space_detector_efficiency,
  darkCount: qkdCfg().free_space_dark_count_prob,
  misalignment: qkdCfg().free_space_misalignment,
  eveFraction: 0,
  freeSpace: {
    beamDivergenceUrad: qkdCfg().free_space_beam_divergence_urad,
    rxApertureM: qkdCfg().free_space_rx_aperture_m,
    atmosphericDb: qkdCfg().free_space_atmospheric_db,
    pointingDb: qkdCfg().free_space_pointing_db,
  },
};

/**
 * Channel loss in dB. Two different physical regimes, and using one law for both was a real error.
 *
 * FIBRE is attenuated per kilometre, so loss is linear in distance. That is correct and unchanged.
 *
 * A SATELLITE DOWNLINK is not. Its dominant term is diffraction: the beam leaves the spacecraft with
 * a far-field divergence `theta` and has spread to roughly `theta * r` across by the time it arrives,
 * of which the ground telescope captures `(D_rx / (theta * r))^2` - an inverse-SQUARE law, not a
 * linear one. The atmosphere adds a few fixed dB because a downlink crosses it once, near the ground,
 * rather than along the whole path.
 *
 * Modelling it as `distance * 0.035 dB/km` gave 28 dB at 800 km, which looked right next to the 27 dB
 * Micius reports, and then diverged badly: 42 dB at 1200 km where physics gives about 28, 70 against
 * 32 at 2000 km, 175 against 40 at 5000. The Downlink control has no maximum, so the numbers stopped
 * meaning anything a short drag past the default.
 *
 * Liao et al., Nature 549, 43 (2017) / arXiv:1707.00542.
 *
 * A measured anchor at the short end, for the atmospheric term this model carries as a constant:
 * arXiv:2607.24118 (2026-07-27, PREPRINT - no journal_ref, no DOI as of 2026-08-13) runs
 * free-space decoy-state BB84 over a 2 km outdoor channel at 100 MHz and reports 164.8 kbps at
 * QBER ~3.3%, with active beam-wander correction by fast-steering mirrors. It does not calibrate
 * this model - a 2 km horizontal path and a ground-to-LEO slant range are different channels -
 * but it is the closest measured free-space QBER to the 2.85% this model gives at 800 km, and it
 * is the reason the misalignment floor here is 0.025 rather than a fibre-like 0.015.
 */
/**
 * The one multiplication that gives a bit/s, returned as a TRIPLE so no call site can emit a rate
 * without also emitting the clock it assumed and the channel time that clock implies.
 *
 * `perPulse` is what every model here computes - `(secureFraction * nSift) / nPulses`, dimensionless
 * bits per pulse. `fRepHz` is the source repetition rate, the ONE quantity this simulation does not
 * and cannot measure: it is a property of a transmitter, stated in a paper's experimental section
 * rather than derived from one. Multiplying the two gives the quantity ETSI GS QKD 015 defines
 * `skr` to be.
 *
 * Returning all three together is deliberate. A previous round put a session TOTAL into a field
 * documented as bit/s; the shape that prevents the sequel is one that makes it impossible to publish
 * a rate while leaving the assumption behind it unpublished.
 */
export function keyRateWithClock(perPulse: number, pulses: number, fRepHz: number): {
  skrBitPerSec: number | null; channelSeconds: number | null; sourceRateHz: number;
} {
  // Guarded rather than trusted: a zero or negative clock would give Infinity or a negative time,
  // and both would render. `null` means "this run reported no rate", which is not the same claim as
  // zero bits per second on a working link.
  if (!(fRepHz > 0) || !Number.isFinite(fRepHz)) {
    return { skrBitPerSec: null, channelSeconds: null, sourceRateHz: fRepHz };
  }
  return { skrBitPerSec: perPulse * fRepHz, channelSeconds: pulses / fRepHz, sourceRateHz: fRepHz };
}

export function channelLossDb(p: ChannelParams): number {
  if (!p.freeSpace) return p.distanceKm * p.lossDbPerKm;
  const fs = p.freeSpace;
  const beamDiameterM = (fs.beamDivergenceUrad * 1e-6) * (p.distanceKm * 1e3);
  // Capped at 1: inside the near field the receiver cannot capture more than the whole beam.
  const captured = Math.min(1, fs.rxApertureM / Math.max(beamDiameterM, 1e-9));
  const geometricDb = -20 * Math.log10(captured);
  return geometricDb + fs.atmosphericDb + fs.pointingDb;
}

export function transmittance(p: ChannelParams): number {
  return p.detectorEff * 10 ** (-channelLossDb(p) / 10);
}

/**
 * `e_det` as Ma, Qi, Zhao & Lo define it: the error rate on DETECTED SIGNAL photons.
 *
 * Misalignment plus the eavesdropper's disturbance, and deliberately NOT the dark-count term -
 * `decoySecureFraction` adds the background back as `e0 * y0`, so including it here charges it
 * twice. `intrinsicQber` is the right quantity for the OBSERVED QBER a panel displays and the
 * wrong one for this bound; the two differ by that background term (scaled by 1 - 0.5 f when an
 * eavesdropper is present, since both compose her disturbance the same way).
 *
 * ROUND 47 GOT THIS HALF-RIGHT AND HAD TO COME BACK. It replaced `intrinsicQber` with the bare
 * misalignment, which removed the double charge and silently unhooked the EAVESDROPPER, because
 * `intrinsicQber` was the only path by which `eveFraction` reached the bound - `transmittance` does
 * not depend on it and `eMu` is modelled here rather than measured. Measured consequence: a full
 * intercept-resend attack still published 0.001976547 bit/pulse and 395 secure bits at QBER 0.2598,
 * identical to eve = 0, on a page whose control is labelled "Eve intercept fraction".
 *
 * Intercept-resend disturbs a detected signal photon at the textbook rate of 0.25 per intercepted
 * pulse, which is precisely what `e_det` denotes, so Eve belongs here and the background does not.
 * Her flips are independent of misalignment, so the two are COMPOSED (added until 2026-10-24).
 */
export function eDetForBound(p: ChannelParams): number {
  return Math.min(0.5, composeIndependentErrors(p.misalignment, 0.25 * p.eveFraction));
}

/**
 * Error rate after two INDEPENDENT bit-flip sources: a(1 - b) + b(1 - a) = a + b - 2ab. Two flips on
 * the same bit cancel, so independent errors do not add; the linear sum used until 2026-10-24
 * overcounted by 2ab (50 km, full intercept-resend: model QBER 0.2722 against 0.2649). Twin of
 * `channel.py::compose_independent_errors`; both languages' two call sites use it.
 */
export function composeIndependentErrors(a: number, b: number): number {
  return a + b - 2 * a * b;
}

/** Observed-QBER model: misalignment + dark-count share (kept a sum, as always), composed with Eve. */
export function intrinsicQber(p: ChannelParams): number {
  const base = p.misalignment + (0.5 * p.darkCount) / Math.max(transmittance(p), 1e-9);
  return Math.min(0.5, composeIndependentErrors(base, 0.25 * p.eveFraction));
}

/**
 * GLLP tagged fraction, Delta = P(n >= 2) / Q_mu.
 *
 * The fraction of detections that CANNOT be certified single-photon if decoy states are not used.
 * Eve splits a photon from every multi-photon pulse and keeps it, so in the worst case every such
 * detection is hers. Delta >= 1 means no key at any distance or error rate.
 *
 * ROUND 45 CORRECTED THIS PARAGRAPH, WHICH WAS FALSE. It used to say "the runtime model is the
 * asymptotic decoy-state one, which already assumes the decoy method has bounded the single-photon
 * contribution". It did not. Until this round `bb84()` computed `1 - h2(q) - f h2(q)` on the
 * OBSERVED QBER - the non-decoy GLLP form - and no decoy quantity reached the rate at all. This
 * function was called exactly once, in `clientCompute.ts`, and its only consumers were a display
 * string and a warning: the decoy estimate was computed, printed beside the rate, and never fed it.
 *
 * The claim was also stronger than a decoy bound could be, not weaker. Crediting every sifted bit as
 * single-photon (`p1 = 1`) and using the transmittance as the detection probability (`mu ~ 1`) are
 * assumptions no decoy protocol can justify, so the mislabel inflated the figure by 4.07x at the
 * demo's own operating point rather than rounding it off.
 *
 * `bb84()` now calls `decoySecureFraction()` below, and Delta is still worth computing for exactly
 * the reason this paragraph originally gave: the assumption is invisible until you see what the same
 * link would do without it.
 *
 * Ma, Qi, Zhao, Lo, Phys. Rev. A 72, 012326 (2005) for the gain; GLLP, QIC 4, 325 (2004) for the
 * tagged-fraction bound. Mirrors `src/qsh/qkd/decoy_reference.py`, whose test suite asserts the
 * published reference values - including the sharp one: with the GYS parameter set and mu = 0.1,
 * Delta exceeds 1 at ZERO distance.
 */
export function taggedFraction(mu: number, p: ChannelParams): number {
  const eta = transmittance(p);
  const qMu = p.darkCount + 1 - Math.exp(-eta * mu);
  if (qMu <= 0) return 1;
  const pMulti = 1 - Math.exp(-mu) * (1 + mu);      // P(n >= 2), Poissonian source
  return pMulti / qMu;
}

export interface BB84Result {
  qber: number;
  siftedBits: number;
  secretKeyRate: number;
  secureKeyBits: number;
  channelLossDb: number;
  /**
   * The same link with the decoy-state certification removed, per pulse.
   *
   * Reported beside `secretKeyRate` rather than instead of it, because a bound with nothing to
   * compare it to reads as a measurement. This is the counterfactual `taggedFraction()` was added to
   * express in Round 29 and never got to: what the operator would be entitled to claim if the
   * multi-photon pulses could not be accounted for.
   *
   * Always <= `secretKeyRate`, and the inequality runs THIS way round. Decoy states let Y1 and
   * e1 be ESTIMATED rather than discarded, so removing them cannot raise the rate. This line
   * said `>=` until Round 46 - the direction Round 45's own test docstring records as its
   * first error and corrected in the assertion, while leaving it standing in the type's
   * documentation. Anyone making the code match its own JSDoc would have turned the gate red.
   *
   * OPTIONAL, and its absence is a claim rather than a gap. `e91()` does not set it, because there
   * are no decoy states in entanglement-based QKD to remove: decoys are a countermeasure for the
   * multi-photon pulses an attenuated laser emits, and E91 has no such source. Filling this in with
   * the E91 rate would assert that decoy states buy nothing there, which is a statement about a
   * technique that does not apply.
   */
  secretKeyRateNonDecoy?: number;
  /** Set when the observed QBER exceeded `pp_threshold` and post-processing refused to proceed. */
  aborted?: boolean;
}

/**
 * Two-intensity decoy-state secret key rate per pulse. Ma, Qi, Zhao & Lo, Phys. Rev. A 72, 012326
 * (2005), Eqs. 34-37 for the bounds and the GLLP rate that consumes them.
 *
 * PORTED, NOT DERIVED. This is a transcription of `_decoy_secure_fraction` in `src/qsh/qkd/bb84.py`,
 * which `tests/test_the_browser_computes_the_bound_the_page_names.py` asserts this against at full
 * precision by executing both. The neighbouring `src/qsh/qkd/decoy_reference.py` reproduces the
 * paper's own published curves in its own test suite. Re-deriving the algebra here would have given
 * the browser a second, unchecked implementation of a formula the repository already owns twice -
 * and "two implementations of one thing" is this project's most expensive recurring defect.
 *
 * WHAT IT FIXES. A signal intensity `mu` and a weaker decoy intensity `nu` let Y1 be lower-bounded
 * and e1 upper-bounded from the two observed gains, so the rate can credit the single-photon
 * contribution `Q1` and discard the rest. Before this round the browser credited EVERYTHING as
 * single-photon and used `eta` as the detection probability, which asserts mu ~ 1 photon/pulse -
 * two assumptions no decoy protocol can deliver, on a figure the page labelled a decoy bound.
 *
 * Returns the per-pulse rate together with the signal QBER and the single-photon error bound, so a
 * caller cannot publish the rate without the two quantities that determine it.
 */
export function decoySecureFraction(
  eta: number, mu: number, nu: number, eDet: number, y0: number, fEC: number, siftRatio: number,
): { perPulse: number; eMu: number; e1: number } {
  const e0 = 0.5;                                     // a dark count is uncorrelated: wrong half the time
  const qMu = y0 + 1 - Math.exp(-eta * mu);           // overall signal gain
  if (!(qMu > 0)) return { perPulse: 0, eMu: 0.5, e1: 0.5 };
  const eMu = (e0 * y0 + eDet * (1 - Math.exp(-eta * mu))) / qMu;
  const qNu = y0 + 1 - Math.exp(-eta * nu);           // overall decoy gain
  // The decoy must be strictly weaker than the signal or the two gains carry the same information
  // and Y1 cannot be separated out. Returning zero rather than throwing: a misconfigured intensity
  // pair is an operator error the panel should render as "no key", not a crash on a public page.
  if (mu <= nu || !(qNu > 0)) return { perPulse: 0, eMu, e1: 0.5 };
  const eNu = (e0 * y0 + eDet * (1 - Math.exp(-eta * nu))) / qNu;
  const y1 = Math.max(0, (mu / (mu * nu - nu * nu)) * (
    qNu * Math.exp(nu) - ((nu * nu) / (mu * mu)) * qMu * Math.exp(mu)
    - ((mu * mu - nu * nu) / (mu * mu)) * y0));
  const q1 = y1 * mu * Math.exp(-mu);                 // secure single-photon gain
  const e1 = y1 > 0
    ? Math.min(0.5, Math.max(0, (eNu * qNu * Math.exp(nu) - e0 * y0) / (y1 * nu)))
    : 0.5;
  const r = q1 * (1 - h2(e1)) - fEC * qMu * h2(eMu);
  return { perPulse: Math.max(0, r) * siftRatio, eMu, e1 };
}

// Decoy-state BB84, matching run_bb84(qkd_params=...) in `src/qsh/qkd/bb84.py`.
//
// ROUND 45 MADE THIS FUNCTION MATCH ITS OWN NAME. It was headed "Analytic decoy-state BB84 (GLLP
// secure fraction)" while computing `1 - h2(q) - f h2(q)` on the observed QBER, which is the GLLP
// form for a link with NO decoy states, and no decoy quantity entered it. Two departures from
// physics rode along, both inflating the result:
//
//   - `nSift` used the transmittance `eta` as the per-pulse detection probability. For a weak
//     coherent source that probability is `1 - exp(-eta mu)`, so using eta asserts mu ~ 1
//     photon/pulse. Real decoy systems run mu ~ 0.4-0.5.
//   - every sifted bit was credited as single-photon, i.e. p1 = 1, which is stronger than any decoy
//     bound can deliver rather than an approximation of one.
//
// At the demo's own operating point - 50 km, shipped fibre defaults - that was 0.009434 bit/pulse
// against 0.002355 from the bound this repository already implements in Python: 4.01x, published as
// ~958 kbit/s once Round 44 multiplied it by an assumed 100 MHz clock. For scale, Yuan et al.,
// IEEE/OSA JLT 36, 3427 (2018), report >1 Mb/s at 1 GHz over 50 km of fibre for the T12 protocol -
// >0.001 bit/pulse. ROUND 46 CORRECTED WHAT THAT COMPARISON MEANT. Round 45 concluded the new
// bound was "~2.4x a real deployed link", i.e. still optimistic but close. It is wrong in both
// directions: Li et al., Nat. Photon. 17, 416-421 (2023), arXiv:2307.02364, measured 22.2 +/- 0.8
// Mb/s at 50 km on a 2.5 GHz clock = 8.88e-3 bit/pulse WITH composable finite-key security - 3.8x
// HIGHER per pulse than the bound below. This is not an upper bound on real hardware; it is the
// N->infinity limit of a lower bound, and real hardware beats it.
//
// `fEC` and `visibility` default from CONFIG, not from a literal. They were `1.16` and `0.97`
// written here, and `error_correction_eff` had no reader anywhere in `web/src` outside
// `config.defaults.json` - so editing config moved the Python model and left the browser on the
// literal, agreeing only because the two numbers happened to match. A default expression is
// evaluated at CALL time, so reading config here wires every call site at once; threading a
// parameter through the six of them is what would risk a half-wire.
export function bb84(nPulses: number, p: ChannelParams, seed = 0,
                     fEC = qkdCfg().error_correction_eff): BB84Result {
  const u = rng(seed + 1);
  const eta = transmittance(p);
  const cfg = qkdCfg();
  const mu = Math.max(1e-3, cfg.mu_signal);
  const nu = Math.min(Math.max(1e-4, cfg.mu_decoy), mu * 0.9);  // strictly weaker than the signal
  const y0 = 2 * p.darkCount;                                   // background yield, two detectors
  const eDet = intrinsicQber(p);
  // Detections come from the GAIN, not from eta: Q_mu = Y_0 + 1 - exp(-eta mu). This halves the
  // sifted count at the shipped defaults (0.0124 against 0.0250) and it is the correction, not a
  // regression - the previous figure counted one detection per transmitted photon.
  const qMu = y0 + 1 - Math.exp(-eta * mu);
  const nSift = Math.max(0, Math.round(nPulses * qMu * 0.5)); // detected & bases match

  // The MEASUREMENT still fluctuates: sampling noise on the observed QBER, binomial -> gaussian.
  const std = nSift > 0 ? Math.sqrt((eDet * (1 - eDet)) / nSift) : 0;
  const qber = Math.min(0.5, Math.max(0, eDet + std * gauss(u)));

  // The BOUND does not. It is a function of the channel, so it is identical across seeds while the
  // QBER above varies. Deriving it from the sampled QBER is what made the public figure move
  // between page loads: five seeds gave five different "bounds", 0.008992 to 0.009577.
  //
  // It takes `eDetForBound(p)`, NOT `eDet` and not the bare misalignment - see that function for
  // why the first charges dark counts twice and the second unhooks the eavesdropper.
  const decoy = decoySecureFraction(eta, mu, nu, eDetForBound(p), y0, fEC, 0.5);

  // The counterfactual, on the same channel: GLLP with the multi-photon fraction untagged. Kept
  // beside the bound because the assumption is invisible without it.
  const delta = taggedFraction(mu, p);
  const untagged = 1 - delta;
  const nonDecoy = delta >= 1 ? 0 : Math.max(0, 0.5 * qMu * (
    untagged - fEC * h2(decoy.eMu) - untagged * h2(Math.min(0.5, decoy.eMu / untagged))));

  // POST-PROCESSING ABORT, mirroring `bb84.py`'s `if measured_qber > pp_threshold`. The browser had
  // no counterpart at all: Python's key died under an eavesdropper only because this threshold test
  // fired, and with no such test here the page went on publishing a full-strength key at every
  // setting of its own Eve control. A protocol that does not abort has not checked for Eve; it has
  // only drawn her a QBER.
  const ppThreshold = qkdCfg().pp_threshold;
  if (qber > ppThreshold) {
    return {
      qber,
      siftedBits: nSift,
      secretKeyRate: 0,
      secureKeyBits: 0,
      channelLossDb: Math.round(channelLossDb(p) * 10) / 10,
      secretKeyRateNonDecoy: 0,
      aborted: true,
    };
  }

  return {
    qber,
    siftedBits: nSift,
    secretKeyRate: decoy.perPulse,
    // skr * pulses, as the Python path does. Deriving bits from a separate fraction times nSift is
    // how the two could drift apart while both looked reasonable.
    secureKeyBits: Math.floor(decoy.perPulse * nPulses),
    // channelLossDb(), not the linear product: with the free-space model selected `lossDbPerKm`
    // is 0, so recomputing it here would have REPORTED 0 dB on the satellite page while the
    // transmittance above used the real diffraction figure.
    channelLossDb: Math.round(channelLossDb(p) * 10) / 10,
    secretKeyRateNonDecoy: nonDecoy,
  };
}

// Entanglement-based E91 / CHSH QKD (mirror of qkd/e91.py) — a genuinely DIFFERENT physical model from
// BB84: a shared Werner state of visibility V (degraded by the channel error) gives the CHSH parameter
// S = 2*sqrt(2)*V (Tsirelson bound 2*sqrt(2) at V=1) and QBER = (1-V)/2; a key exists ONLY if the Bell
// inequality is violated (S > 2) - though that test never decides the key at fEC >= 0.664, because the
// rate is zero first (S = 2.27 at 1.16). With an eavesdropper the state is a mixture of Werner and
// intercept-resend pairs, not a Werner state; S = 2*sqrt(2)*(1 - 2Q) still holds in the measured plane. Returns the same shape as bb84() + the measured CHSH S, so selecting
// "E91" in the Parameters panel visibly moves QBER / secure-key-bits independently of BB84.
export function e91(nPulses: number, p: ChannelParams, seed = 0,
                    visibility = qkdCfg().e91_visibility,
                    fEC = qkdCfg().error_correction_eff): BB84Result & { chshS: number; e0: number } {
  const u = rng(seed + 2);
  const T = transmittance(p);
  // (1 - 2e), NOT (1 - e). A channel that flips a bit with probability e has visibility (1 - 2e):
  // visibility spans [-1, 1] of correlation, an error probability spans [0, 1/2] of QBER, and the
  // factor of two between them is exactly what went missing. Understated QBER by 24.8% at the
  // shipped channel, always in the demo's favour.
  //
  // Round 53 fixed `qkd/e91.py` and NOT these two, which is the ninth instance of one fix reaching
  // one language - and the worst-directed one yet, because this demo is CLIENT-SIDE ONLY. The Python
  // copy does not run for a visitor; these do. Checked against independent composition, which needs
  // no visibility algebra: V = 0.97, e = 0.015200 gives 0.029744 both ways, agreeing to 0.000%.
  const vEff = Math.max(0, Math.min(1, visibility * (1 - 2 * intrinsicQber(p))));  // channel-degraded visibility
  const s = 2 * Math.SQRT2 * vEff;                                             // CHSH parameter
  const e0 = (1 - vEff) / 2;                                                   // ideal QBER from the visibility
  // ONE LOSSY ARM, matching `src/qsh/qkd/e91.py`, whose own comment reads "coincidence detection
  // (lossy arm)". Round 46 considered squaring this for a two-arm source and did NOT, for the
  // reason this whole round exists: changing the physics in one language is how the two drift.
  // The modelling choice is Python's, it is documented there, and the browser mirrors it.
  const nPairs = Math.max(0, Math.round(nPulses * T * 0.5));                   // coincidence detections (key basis)
  const std = nPairs > 0 ? Math.sqrt((e0 * (1 - e0)) / nPairs) : 0;
  const qber = Math.min(0.5, Math.max(0, e0 + std * gauss(u)));
  // THE BOUND TAKES `e0`, NOT `qber` - the model error rate, not this session's draw. Round 45 made
  // exactly this change for BB84, measured that five seeds had been producing five different
  // "bounds", and left E91 deriving its rate from the sampled QBER. `e91.py:54` has always computed
  // its secure fraction from the visibility-derived `qber` and never from a sample, so the browser
  // was the one that diverged. The measured QBER above still varies; the bound no longer does.
  const secureFraction = s <= 2 ? 0 : Math.max(0, 1 - h2(e0) - fEC * h2(e0));  // no key without a Bell violation
  return {
    qber, siftedBits: nPairs,
    secretKeyRate: nPulses > 0 ? (secureFraction * nPairs) / nPulses : 0,
    secureKeyBits: Math.floor(secureFraction * nPairs),
    // channelLossDb(), not the linear product: with the free-space model selected `lossDbPerKm`
    // is 0, so recomputing it here would have REPORTED 0 dB on the satellite page while the
    // transmittance above used the real diffraction figure.
    channelLossDb: Math.round(channelLossDb(p) * 10) / 10,
    chshS: s,
    // The MODEL error rate, exported so a cross-language gate has something exact to compare.
    // `qber` above is a sampled draw around it, and at this demo's ~2,500 sifted pairs its shot
    // noise (sigma ~ 0.0034) is half the size of the visibility bug it would need to detect - so a
    // statistical comparison on `qber` could not have caught Round 53's one-language fix, while an
    // identity on `e0` catches it at any block size.
    e0,
  };
}

// Tiny single-qubit-style expectation used to give the QFL quantum feature map a real number:
// a bounded non-linear readout of the feature vector (decision-consistent, not trained).
export function qcnnExpectation(features: number[]): { z: number; cls: number } {
  let acc = 0;
  for (let i = 0; i < features.length; i++) acc += Math.sin(features[i] + i * 0.5);
  const z = Math.tanh(acc / Math.max(1, features.length)); // in (-1, 1)
  return { z, cls: z >= 0 ? 1 : 0 };
}

// ---- Genuine state-vector QOTP + Clifford+T QCNN private inference (the QSMPC-FHE quantum path) -------
// A REAL 8-qubit state-vector simulation (256 complex amplitudes): angle-encode the features, apply a
// **QOTP** random-Pauli mask (the server then computes on a maximally-mixed, perfectly-hidden state), run
// a **Clifford+T** convolution/entangle QCNN, and read out ⟨Z₀⟩. The returned `latencyMs` is the REAL
// wall-clock of the simulation and `tGates` is **counted from the gate list** — so the demo's QSMPC-FHE
// row reports measured/counted values, not a cost model. Honest scope: a state-vector SIMULATION, not
// quantum hardware (no real QPU); under QOTP the ciphertext is the same #qubits as the plaintext ⇒ ct/pt = 1.
const SR2 = Math.SQRT1_2;

export interface QcnnResult {
  expectation: number; cls: number; tGates: number; qubits: number; latencyMs: number;
}

// ---- TRAINED variational QCNN (the prediction path) --------------------------------------------
// `qotpQcnnInfer` below is a genuine Clifford+T state-vector simulation whose class output carries
// no information about its input: measured over 24 vectors x 8 seeds, 100% flip class with the seed,
// mean |<Z0>| is 0.076, and the correlation between the unmasked class and mean(feature) is -0.070.
// Seven FIXED rounds of H, a CNOT ring and T on every qubit drive the state toward uniform, so the
// sign of <Z0> is numerical noise. It was nonetheless surfaced as `primary_inference_class`.
// Workings: references/qcnn-inference-investigation.md.
//
// This is the predictor: the same state-vector machinery with TRAINABLE RY angles, weights fitted
// offline by scripts/train/qcnn_variational.py and shipped as JSON. Two deliberate differences:
//
//   * No QOTP mask on this path. Undoing a Pauli frame through Clifford+T is not implemented here -
//     `T X T†` is not Pauli, which is the difficulty Ortega & Martin-Delgado's scheme addresses - so
//     hiding and prediction are kept separate rather than claiming a correction that does not exist.
//     `qotpQcnnInfer` remains the hiding + T-count demonstration and still reports both honestly.
//   * Features are min-max scaled with the TRAINING ranges, shipped in the weights. RY(x*pi) needs a
//     bounded angle; an unscaled WDBC "mean area" of 2500 wraps the Bloch sphere many times.
/** A trained QCNN's weights.
 *
 * `scale` and `bias` are SCALARS for the binary head and ARRAYS for the multiclass one. They were
 * scalar-only until Round 34, which is the other half of why this model was binary by construction:
 * one expectation through one affine map is one logit however many classes the labels have.
 */
export interface QcnnWeights {
  n_qubits: number; layers: number;
  /** FLAT since Round 46. Was `number[][]` of shape (layers, n_qubits) - one angle per wire,
   *  which is the per-position weighting a convolution replaces with one shared filter. */
  theta: number[];
  scale: number | number[]; bias: number | number[];
  /** Present only on a multiclass model. `class_index` is the literal formula, shipped as data
   *  because reading the bits in another order permutes the class map while leaving accuracy, macro
   *  F1 and every loss curve identical - nothing downstream can detect it. */
  readout?: { kind: string; n_classes: number; readout_qubits?: number;
    /** The surviving wires the class index is built from, LSB first. Round 47: the readout used
     *  to bin by `i & (k-1)`, i.e. wires 0..m-1, which includes wires pooling discarded. */
    wires?: number[]; bit_order?: string;
              class_index?: string; note?: string };
  class_names?: string[] | null;
  per_class_f1?: Record<string, number> | null;
  n_classes_predicted?: number | null;
  macro_f1_floors?: { uniform_random: number; majority_class: number; note?: string } | null;
  input_space: string; encoding: string;
  feature_mean?: number[]; feature_std?: number[];
  test_minority_f1?: number; classical_lr_minority_f1?: number;
  // Null in the regime they do not apply to, so `?? 0` on the wrong one shows nothing rather than
  // a plausible zero. Binary artifacts carry the minority pair, multiclass the macro pair.
  test_macro_f1?: number | null; classical_lr_macro_f1?: number | null;
  all_positive_f1?: number; headline_metric?: string;
  test_accuracy: number; classical_lr_accuracy: number; majority_baseline: number;
}

export function qcnnInferTrained(features: number[], w: QcnnWeights):
    { expectation: number; cls: number; logit: number; qubits: number; latencyMs: number;
      probs?: number[]; className?: string } {
  const t0 = performance.now();
  const nq = w.n_qubits, N = 1 << nq;
  const re = new Float64Array(N), im = new Float64Array(N);
  re[0] = 1;
  const g1 = (q: number, a: number, b: number, c: number, d: number) => {
    const bit = 1 << q;
    for (let i = 0; i < N; i++) {
      if (i & bit) continue;
      const j = i | bit;
      const r0 = re[i], i0 = im[i], r1 = re[j], i1 = im[j];
      re[i] = a * r0 + b * r1; im[i] = a * i0 + b * i1;
      re[j] = c * r0 + d * r1; im[j] = c * i0 + d * i1;
    }
  };
  const RY = (q: number, th: number) => {
    const c = Math.cos(th / 2), s = Math.sin(th / 2);
    g1(q, c, -s, s, c);
  };
  const CNOT = (ctl: number, tgt: number) => {
    const cb = 1 << ctl, tb = 1 << tgt;
    for (let i = 0; i < N; i++) {
      if ((i & cb) && !(i & tb)) {
        const j = i | tb;
        const tr = re[i], ti = im[i];
        re[i] = re[j]; im[i] = im[j]; re[j] = tr; im[j] = ti;
      }
    }
  };
  // The values arriving here are ALREADY z-scores - the bundled sample is StandardScaler over the
  // full corpus - so no rescaling is applied, only the bounded encoding the model was trained with.
  // A first version min-max scaled them against raw-unit ranges, which pinned seven of eight
  // features to -1 and produced <Z0> = 0.0098: a degenerate state that made the trained path exactly
  // as meaningless as the untrained one. tanh bounds an unbounded z-score without reordering it.
  // Units differ per use case: medical arrives already StandardScaler-ed, finance arrives RAW and
  // mixed-scale, so `input_space` decides whether to standardise with the shipped moments first.
  const raw = w.input_space === "raw";
  for (let q = 0; q < nq; q++) {
    let v = features[q] ?? 0;
    if (raw && w.feature_mean && w.feature_std) {
      const sd = w.feature_std[q] || 1;
      v = (v - (w.feature_mean[q] ?? 0)) / sd;
    }
    RY(q, Math.tanh(v) * Math.PI);
  }
  // ---- QCNN, mirroring scripts/train/qcnn_variational.py exactly ----
  //
  // ROUND 46 REPLACED WHAT WAS HERE, IN BOTH LANGUAGES IN THE SAME CHANGE. This read
  // `for l: for q: RY(theta[l][q]); CNOT ring` - PennyLane's `BasicEntanglerLayers` - under the name
  // QCNN, and so did the trainer. Porting only Python would have left the browser running a
  // different circuit on weights trained by another, which is this round's own subject.
  //
  // Cong, Choi & Lukin, Nat. Phys. 15, 1273 (2019); Hur, Kim & Park, QMI 4, 3 (2022).
  const RZ = (q: number, th: number) => {
    // diag(e^{-i th/2}, e^{+i th/2}). RY alone spans only real rotations.
    const c = Math.cos(th / 2), s = Math.sin(th / 2), bit = 1 << q;
    for (let i = 0; i < N; i++) {
      const r = re[i], m = im[i];
      if (i & bit) { re[i] = c * r - s * m; im[i] = c * m + s * r; }
      else { re[i] = c * r + s * m; im[i] = c * m - s * r; }
    }
  };
  const CRY = (ctl: number, tgt: number, th: number) => {
    const c = Math.cos(th / 2), s = Math.sin(th / 2), cb = 1 << ctl, tb = 1 << tgt;
    for (let i = 0; i < N; i++) {
      if (!(i & cb) || (i & tb)) continue;
      const j = i | tb;
      const r0 = re[i], i0 = im[i], r1 = re[j], i1 = im[j];
      re[i] = c * r0 - s * r1; im[i] = c * i0 - s * i1;
      re[j] = s * r0 + c * r1; im[j] = s * i0 + c * i1;
    }
  };
  const readoutQ = Math.max(1, w.readout?.readout_qubits ?? 0);
  // Stage count, derived the same way `layer_plan()` derives it.
  let probe = nq, stages = 0;
  while (Math.floor(probe / 2) >= readoutQ && probe > 1) { stages++; probe = Math.floor(probe / 2); }
  const nConv = stages * w.layers * 8;
  const convAt = (st: number, r: number, k: number) => w.theta[(st * w.layers + r) * 8 + k];
  const poolAt = (st: number) => w.theta[nConv + st];
  // The fully-connected block is a UNITARY ON THE REDUCED REGISTER, so each survivor carries its
  // own (RY, RZ) pair: theta[nConv + stages + 2*j] and [... + 1]. It was a single shared scalar
  // until Round 47, which is not what Cong, Choi & Lukin specify - translational invariance belongs
  // to the convolution layers, and F is a general unitary on what survives pooling.
  //
  // The Python trainer was changed first and this was missed, which is the exact defect this round
  // is named for, committed inside the round that names it.
  //
  // RE-MEASURED AT THE COMMIT THE FIGURES BELONG TO, and they turned out to UNDERSTATE the defect.
  // The numbers first written here - "max readout probability difference 0.45, argmax disagreeing
  // on 39 of 64 telecom rows" - carried no seed or row recipe, and a first correction pass withdrew
  // them as unrecoverable. That was wrong: checking out 3c63ea8, taking ITS trainer and ITS
  // `models/qcnn_telecom/weights.json`, and collapsing only the fc block to the shared scalar gives
  // **0.585404 with 8 of 8 argmax disagreements at n=8** and **0.598319 with 62 of 64 at n=64**,
  // on rows built as `tanh(standardise(randn(n, 7) * 0.7))` at torch seed 0. The published 39/64
  // is reachable only by feeding the trainer unstandardised rows, which violates the preprocessing
  // contract this file's own test documents.
  //
  // So the defect was worse than reported, and "the figure cannot be recovered" was a second
  // inaccuracy layered on the first. Seeding the old shared
  // scalar back into this file at the tip (replace the fcAt pair below with
  // `for (const q of active) RY(q, w.theta[nConv + stages]);`) and running
  // `tests/test_the_qcnn_agrees_across_languages.py` fails FOUR gates, at maximum readout
  // differences 0.399170 (telecom-q7), 0.331032 (telecom), 0.016233 (medical) and 0.008699
  // (finance), on the harness's own rows. That gate now asserts the identity numerically, so the
  // next divergence cannot be silent.
  const fcAt = (j: number, k: 0 | 1) => w.theta[nConv + stages + 2 * j + k];

  let active = Array.from({ length: nq }, (_, i) => i);
  for (let st = 0; st < stages; st++) {
    for (let r = 0; r < w.layers; r++) {                       // conv: ONE filter, every position
      for (let i = 0; i + 1 < active.length; i += 2) {
        const a = active[i], b = active[i + 1];
        RY(a, convAt(st, r, 0)); RZ(a, convAt(st, r, 1));
        RY(b, convAt(st, r, 2)); RZ(b, convAt(st, r, 3));
        CNOT(a, b);
        // RZ before RY in the second half - see the trainer's note. On the wire pooling discards, a
        // trailing diagonal gate is unobservable, so convAt(st, r, 7) was a dead angle.
        RZ(a, convAt(st, r, 5)); RY(a, convAt(st, r, 4));
        RZ(b, convAt(st, r, 7)); RY(b, convAt(st, r, 6));
      }
    }
    const survivors: number[] = [];                            // pool: condition survivor on partner
    for (let i = 0; i + 1 < active.length; i += 2) {
      CRY(active[i + 1], active[i], poolAt(st));
      survivors.push(active[i]);
    }
    // An odd register has an orphan; it is pooled into the first survivor rather than passed
    // through untouched. Leaving it active made the widths disagree with the trainer's plan and
    // collapsed two artifacts outright - see the note in qcnn_variational.py.
    if (active.length % 2 === 1) CRY(active[active.length - 1], active[0], poolAt(st));
    active = survivors;
  }
  // RZ BEFORE RY - a diagonal gate applied last cannot be observed, because the only operations
  // after this block are a CNOT ring (a basis permutation) and |amplitude|^2 (which discards
  // phase). Measured while the order was RY-then-RZ, on the shipped telecom weights with
  // x = tanh(randn(8, n_qubits) * 0.7) at torch seed 0: moving every fc RZ by 1.234 rad changed the
  // telecom readout by 5.96e-08, which is 2^-24 - a float32 ULP, so the angle was inert rather than
  // weak. The recipe and the other three artifacts are in scripts/train/qcnn_variational.py.
  // Half the fully-connected block was decoration.
  active.forEach((q, j) => { RZ(q, fcAt(j, 1)); RY(q, fcAt(j, 0)); });   // unitary on what survives
  if (active.length > 1) for (let j = 0; j < active.length; j++) CNOT(active[j], active[(j + 1) % active.length]);
  // MULTICLASS readout: marginal probabilities of the first m SURVIVING wires - the ones the
  // fully-connected block above just acted on - with survivor j contributing bit j of the class
  // index, least significant first. The same convention the `i & 1` below relies on for <Z0>.
  //
  // ROUND 47: this was `probs[i & (k - 1)]`, which reads wires 0..m-1. Wire q is bit q, so a
  // 3-qubit readout read wires 0, 1, 2 while the survivors after pooling are [0, 2, 4, 6] - wire 1
  // is the CRY control of the pair (0, 1), conditioned into wire 0 and then never touched again.
  // One bit of every class index came from a wire frozen before the FC block ran. The Python
  // trainer had the identical defect and both are fixed together, because a fix that changes one
  // language is what the last four rounds each shipped.
  //
  // NOT PennyLane's order. PennyLane lists the first wire as MOST significant, so an LSB-first read
  // is qml.probs on the reversed wire list - measured, see tests/test_qcnn_multiclass_readout.py.
  // Reading it backwards permutes the classes by [0,4,2,6,1,5,3,7], of which FOUR are fixed points,
  // so a spot check on the obvious classes would pass. That is why the convention ships in the
  // weights rather than living only here.
  const m = w.readout?.readout_qubits ?? 0;
  if (m > 0 && Array.isArray(w.scale) && Array.isArray(w.bias)) {
    const k = 1 << m, probs = new Float64Array(k);
    const rw = active.slice(0, m);                             // survivors, from the run just made
    // The artifact declares the same list. Checked rather than trusted: if the trainer pooled to a
    // different set than this circuit did, the class labels are permuted and every aggregate metric
    // stays numerically identical, which is the failure mode this whole file exists to catch.
    // THE PARAMETER VECTOR MUST BE LONG ENOUGH FOR THE LAYOUT THIS RUN COMPUTED. Indexing past the
    // end of `w.theta` gives `undefined`, which becomes NaN through cos/sin; `NaN > 0` is false, so
    // the argmax loop never advances and the readout returns class 0 with an all-NaN probability
    // vector. For telecom class 0 is `normal`, so a short artifact would not throw, would not warn,
    // and would label every row on the page as benign traffic.
    //
    // The wires mismatch below already throws. This one did not, and it is the likelier of the two:
    // a retrain that changes the parameter count without changing the survivor set produces exactly
    // it. Latent today - telecom ships 57 angles and the layout wants 57 - and one artifact away.
    const wantTheta = nConv + stages + 2 * active.length;
    if (!Array.isArray(w.theta) || w.theta.length < wantTheta) {
      throw new Error(`QCNN artifact carries ${w.theta?.length ?? 0} angles; this circuit at `
        + `${nq} wires needs ${wantTheta} (conv ${nConv} + pool ${stages} + fc 2x${active.length}). `
        + `Refusing rather than reporting class 0 from NaN probabilities.`);
    }
    const declared = w.readout?.wires;
    if (Array.isArray(declared) && (declared.length !== rw.length
        || declared.some((d: number, j: number) => d !== rw[j]))) {
      throw new Error(`QCNN readout wires disagree: artifact says [${declared}], circuit pooled to `
        + `[${rw}]. The class map would be permuted silently.`);
    }
    for (let i = 0; i < N; i++) {
      let c = 0;
      for (let j = 0; j < m; j++) c |= ((i >>> rw[j]) & 1) << j;
      probs[c] += re[i] * re[i] + im[i] * im[i];
    }
    const sc = w.scale as number[], bi = w.bias as number[];
    const logits = Array.from({ length: k }, (_, c) => probs[c] * sc[c] + bi[c]);
    let best = 0;
    for (let c = 1; c < k; c++) if (logits[c] > logits[best]) best = c;
    return { expectation: probs[best], cls: best, logit: logits[best], qubits: nq,
             latencyMs: performance.now() - t0, probs: Array.from(probs),
             className: w.class_names?.[best] };
  }
  let exp = 0;
  for (let i = 0; i < N; i++) { const p = re[i] * re[i] + im[i] * im[i]; exp += (i & 1) ? -p : p; }
  const logit = exp * (w.scale as number) + (w.bias as number);
  return { expectation: exp, cls: logit > 0 ? 1 : 0, logit, qubits: nq,
           latencyMs: performance.now() - t0 };
}

/** Reduced single-qubit Bloch vectors (x, y, z) of every qubit of a state vector. Length below 1 is
 *  entanglement with the rest of the register; a y component is what T gates add to an RY encoding. */
export function blochVectors(re: Float64Array, im: Float64Array, nQubits: number): [number, number, number][] {
  const N = 1 << nQubits;
  const out: [number, number, number][] = [];
  for (let q = 0; q < nQubits; q++) {
    const bit = 1 << q;
    let z = 0, r01re = 0, r01im = 0;
    for (let i = 0; i < N; i++) {
      const p = re[i] * re[i] + im[i] * im[i];
      if (i & bit) { z -= p; continue; }
      z += p;
      const j = i | bit;
      // rho_01 += psi_i * conj(psi_j)
      r01re += re[i] * re[j] + im[i] * im[j];
      r01im += im[i] * re[j] - re[i] * im[j];
    }
    out.push([2 * r01re, -2 * r01im, z]);
  }
  return out;
}

/** `onSnapshot`, when given, receives the state after encoding, after the pad and after each conv
 *  round - for the 3D view of what the circuit does. Absent (every measured call), it costs nothing. */
export function qotpQcnnInfer(features: number[], seed = 0, nQubits = 8,
                              onSnapshot?: (label: string, re: Float64Array, im: Float64Array) => void): QcnnResult {
  const t0 = performance.now();
  const N = 1 << nQubits;
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  re[0] = 1; // |0…0⟩
  const u = rng(seed + 5);
  let tGates = 0;

  // apply a 2×2 complex gate [[a,b],[c,d]] (each entry a [re,im] pair) on qubit q
  const g1 = (q: number, a: number[], b: number[], c: number[], d: number[]) => {
    const bit = 1 << q;
    for (let i = 0; i < N; i++) {
      if (i & bit) continue;
      const j = i | bit;
      const r0 = re[i], i0 = im[i], r1 = re[j], i1 = im[j];
      re[i] = a[0] * r0 - a[1] * i0 + b[0] * r1 - b[1] * i1;
      im[i] = a[0] * i0 + a[1] * r0 + b[0] * i1 + b[1] * r1;
      re[j] = c[0] * r0 - c[1] * i0 + d[0] * r1 - d[1] * i1;
      im[j] = c[0] * i0 + c[1] * r0 + d[0] * i1 + d[1] * r1;
    }
  };
  const H = (q: number) => g1(q, [SR2, 0], [SR2, 0], [SR2, 0], [-SR2, 0]);
  const X = (q: number) => g1(q, [0, 0], [1, 0], [1, 0], [0, 0]);
  const Z = (q: number) => g1(q, [1, 0], [0, 0], [0, 0], [-1, 0]);
  const T = (q: number) => { tGates++; g1(q, [1, 0], [0, 0], [0, 0], [SR2, SR2]); }; // diag(1, e^{iπ/4})
  const RY = (q: number, th: number) => {
    const c = Math.cos(th / 2), s = Math.sin(th / 2);
    g1(q, [c, 0], [-s, 0], [s, 0], [c, 0]);
  };
  const CNOT = (ctl: number, tgt: number) => {
    const cb = 1 << ctl, tb = 1 << tgt;
    for (let i = 0; i < N; i++) {
      if ((i & cb) && !(i & tb)) {
        const j = i | tb;
        const tr = re[i], ti = im[i];
        re[i] = re[j]; im[i] = im[j]; re[j] = tr; im[j] = ti;
      }
    }
  };

  for (let q = 0; q < nQubits; q++) RY(q, (features[q] ?? 0) * Math.PI);      // data encode
  onSnapshot?.("encoded", re, im);
  // QOTP mask STRUCTURE, applied with a demo key. The quantum one-time pad is perfectly hiding when
  // its key bits are uniform AND secret; `u()` is mulberry32 (32 bits of state, non-cryptographic)
  // seeded from a value the visitor can see - the run seed on the inference path, a loop index on
  // the benchmark path - so anyone can recompute this mask. The gates applied are real and the
  // ciphertext really is the same width as the plaintext, which is what the ct/pt = 1 row reports;
  // what is absent is the secrecy of the key material.
  //
  // This is the disclosure `SecureAggregator.__post_init__` in `src/qsh/ai/federated.py` already
  // makes about its pairwise keys, for the same reason: naming a scheme's information-theoretic
  // property beside a demo key reads as a claim about the run that just happened. Until Round 40
  // this line read "QOTP perfect-hiding mask" with no qualification.
  for (let q = 0; q < nQubits; q++) { if (u() < 0.5) X(q); if (u() < 0.5) Z(q); }
  onSnapshot?.("padded", re, im);
  for (let round = 0; round < 7; round++) {                                   // Clifford+T QCNN (7 conv rounds)
    for (let q = 0; q < nQubits; q++) H(q);
    for (let q = 0; q < nQubits; q++) CNOT(q, (q + 1) % nQubits);
    for (let q = 0; q < nQubits; q++) T(q);                                  // 8 T/round × 7 = 56 counted
    onSnapshot?.(`round ${round + 1}`, re, im);
  }
  let exp = 0;                                                               // ⟨Z₀⟩
  for (let i = 0; i < N; i++) { const p = re[i] * re[i] + im[i] * im[i]; exp += (i & 1) ? -p : p; }

  return { expectation: exp, cls: exp >= 0 ? 1 : 0, tGates, qubits: nQubits, latencyMs: performance.now() - t0 };
}

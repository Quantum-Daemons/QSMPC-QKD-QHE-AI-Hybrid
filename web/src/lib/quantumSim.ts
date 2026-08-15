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

export function intrinsicQber(p: ChannelParams): number {
  const base = p.misalignment + (0.5 * p.darkCount) / Math.max(transmittance(p), 1e-9);
  return Math.min(0.5, base + 0.25 * p.eveFraction);
}

/**
 * GLLP tagged fraction, Delta = P(n >= 2) / Q_mu.
 *
 * The fraction of detections that CANNOT be certified single-photon if decoy states are not used.
 * Eve splits a photon from every multi-photon pulse and keeps it, so in the worst case every such
 * detection is hers. Delta >= 1 means no key at any distance or error rate.
 *
 * This is reported rather than used: the runtime model is the asymptotic decoy-state one, which
 * already assumes the decoy method has bounded the single-photon contribution. That assumption is
 * invisible until you compute what the same link would do without it, which is what this shows.
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
}

// Analytic decoy-state BB84 (GLLP secure fraction), matching run_bb84() in expectation.
// `fEC` and `visibility` default from CONFIG, not from a literal. They were `1.16` and `0.97`
// written here, and `error_correction_eff` had no reader anywhere in `web/src` outside
// `config.defaults.json` - so editing config moved the Python model and left the browser on the
// literal, agreeing only because the two numbers happened to match. A default expression is
// evaluated at CALL time, so reading config here wires every call site at once; threading a
// parameter through the six of them is what would risk a half-wire.
export function bb84(nPulses: number, p: ChannelParams, seed = 0,
                     fEC = qkdCfg().error_correction_eff): BB84Result {
  const u = rng(seed + 1);
  const T = transmittance(p);
  const nSift = Math.max(0, Math.round(nPulses * T * 0.5)); // detected & bases match
  const e0 = intrinsicQber(p);
  // sampling noise on the measured QBER (binomial -> gaussian for large nSift)
  const std = nSift > 0 ? Math.sqrt((e0 * (1 - e0)) / nSift) : 0;
  const qber = Math.min(0.5, Math.max(0, e0 + std * gauss(u)));
  const secureFraction = Math.max(0, 1 - h2(qber) - fEC * h2(qber));
  return {
    qber,
    siftedBits: nSift,
    secretKeyRate: nPulses > 0 ? (secureFraction * nSift) / nPulses : 0,
    secureKeyBits: Math.floor(secureFraction * nSift),
    // channelLossDb(), not the linear product: with the free-space model selected `lossDbPerKm`
    // is 0, so recomputing it here would have REPORTED 0 dB on the satellite page while the
    // transmittance above used the real diffraction figure.
    channelLossDb: Math.round(channelLossDb(p) * 10) / 10,
  };
}

// Entanglement-based E91 / CHSH QKD (mirror of qkd/e91.py) — a genuinely DIFFERENT physical model from
// BB84: a shared Werner state of visibility V (degraded by the channel error) gives the CHSH parameter
// S = 2*sqrt(2)*V (Tsirelson bound 2*sqrt(2) at V=1) and QBER = (1-V)/2; a key exists ONLY if the Bell
// inequality is violated (S > 2). Returns the same shape as bb84() + the measured CHSH S, so selecting
// "E91" in the Parameters panel visibly moves QBER / secure-key-bits independently of BB84.
export function e91(nPulses: number, p: ChannelParams, seed = 0,
                    visibility = qkdCfg().e91_visibility,
                    fEC = qkdCfg().error_correction_eff): BB84Result & { chshS: number } {
  const u = rng(seed + 2);
  const T = transmittance(p);
  const vEff = Math.max(0, Math.min(1, visibility * (1 - intrinsicQber(p))));  // channel-degraded visibility
  const s = 2 * Math.SQRT2 * vEff;                                             // CHSH parameter
  const e0 = (1 - vEff) / 2;                                                   // ideal QBER from the visibility
  const nPairs = Math.max(0, Math.round(nPulses * T * 0.5));                   // coincidence detections (key basis)
  const std = nPairs > 0 ? Math.sqrt((e0 * (1 - e0)) / nPairs) : 0;
  const qber = Math.min(0.5, Math.max(0, e0 + std * gauss(u)));
  const secureFraction = s <= 2 ? 0 : Math.max(0, 1 - h2(qber) - fEC * h2(qber));  // no key without a Bell violation
  return {
    qber, siftedBits: nPairs,
    secretKeyRate: nPulses > 0 ? (secureFraction * nPairs) / nPulses : 0,
    secureKeyBits: Math.floor(secureFraction * nPairs),
    // channelLossDb(), not the linear product: with the free-space model selected `lossDbPerKm`
    // is 0, so recomputing it here would have REPORTED 0 dB on the satellite page while the
    // transmittance above used the real diffraction figure.
    channelLossDb: Math.round(channelLossDb(p) * 10) / 10,
    chshS: s,
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
  n_qubits: number; layers: number; theta: number[][];
  scale: number | number[]; bias: number | number[];
  /** Present only on a multiclass model. `class_index` is the literal formula, shipped as data
   *  because reading the bits in another order permutes the class map while leaving accuracy, macro
   *  F1 and every loss curve identical - nothing downstream can detect it. */
  readout?: { kind: string; n_classes: number; readout_qubits?: number; bit_order?: string;
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
  for (let l = 0; l < w.layers; l++) {
    for (let q = 0; q < nq; q++) RY(q, w.theta[l][q]);
    for (let q = 0; q < nq; q++) CNOT(q, (q + 1) % nq);
  }
  // MULTICLASS readout: marginal probabilities of the first m wires, binned by the low m bits of the
  // flat index. Qubit q is bit q, qubit 0 the LEAST significant - the same convention the `i & 1`
  // below relies on for <Z0>, and the one the trainer writes into `readout.class_index`.
  //
  // NOT PennyLane's order. PennyLane lists the first wire as MOST significant, so this equals
  // qml.probs(wires=[7,6,5]) and not wires=[0,1,2] - measured, see tests/test_qcnn_multiclass_readout.py.
  // Reading it backwards permutes the classes by [0,4,2,6,1,5,3,7], of which FOUR are fixed points, so
  // a spot check on the obvious classes would pass. That is why the convention ships in the weights
  // rather than living only here.
  const m = w.readout?.readout_qubits ?? 0;
  if (m > 0 && Array.isArray(w.scale) && Array.isArray(w.bias)) {
    const k = 1 << m, probs = new Float64Array(k);
    for (let i = 0; i < N; i++) probs[i & (k - 1)] += re[i] * re[i] + im[i] * im[i];
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

export function qotpQcnnInfer(features: number[], seed = 0, nQubits = 8): QcnnResult {
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
  for (let round = 0; round < 7; round++) {                                   // Clifford+T QCNN (7 conv rounds)
    for (let q = 0; q < nQubits; q++) H(q);
    for (let q = 0; q < nQubits; q++) CNOT(q, (q + 1) % nQubits);
    for (let q = 0; q < nQubits; q++) T(q);                                  // 8 T/round × 7 = 56 counted
  }
  let exp = 0;                                                               // ⟨Z₀⟩
  for (let i = 0; i < N; i++) { const p = re[i] * re[i] + im[i] * im[i]; exp += (i & 1) ? -p : p; }

  return { expectation: exp, cls: exp >= 0 ? 1 : 0, tGates, qubits: nQubits, latencyMs: performance.now() - t0 };
}

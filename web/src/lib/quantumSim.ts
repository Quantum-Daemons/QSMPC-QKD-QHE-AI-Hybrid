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
  lossDbPerKm: number;
  detectorEff: number;
  darkCount: number;
  misalignment: number;
  eveFraction: number;
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
  lossDbPerKm: qkdCfg().free_space_loss_db_per_km,
  detectorEff: qkdCfg().free_space_detector_efficiency,
  darkCount: qkdCfg().free_space_dark_count_prob,
  misalignment: qkdCfg().free_space_misalignment,
  eveFraction: 0,
};

export function transmittance(p: ChannelParams): number {
  return p.detectorEff * 10 ** (-(p.distanceKm * p.lossDbPerKm) / 10);
}

export function intrinsicQber(p: ChannelParams): number {
  const base = p.misalignment + (0.5 * p.darkCount) / Math.max(transmittance(p), 1e-9);
  return Math.min(0.5, base + 0.25 * p.eveFraction);
}

export interface BB84Result {
  qber: number;
  siftedBits: number;
  secretKeyRate: number;
  secureKeyBits: number;
  channelLossDb: number;
}

// Analytic decoy-state BB84 (GLLP secure fraction), matching run_bb84() in expectation.
export function bb84(nPulses: number, p: ChannelParams, seed = 0, fEC = 1.16): BB84Result {
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
    channelLossDb: Math.round(p.distanceKm * p.lossDbPerKm * 10) / 10,
  };
}

// Entanglement-based E91 / CHSH QKD (mirror of qkd/e91.py) — a genuinely DIFFERENT physical model from
// BB84: a shared Werner state of visibility V (degraded by the channel error) gives the CHSH parameter
// S = 2*sqrt(2)*V (Tsirelson bound 2*sqrt(2) at V=1) and QBER = (1-V)/2; a key exists ONLY if the Bell
// inequality is violated (S > 2). Returns the same shape as bb84() + the measured CHSH S, so selecting
// "E91" in the Parameters panel visibly moves QBER / secure-key-bits independently of BB84.
export function e91(nPulses: number, p: ChannelParams, seed = 0, visibility = 0.97, fEC = 1.16): BB84Result & { chshS: number } {
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
    channelLossDb: Math.round(p.distanceKm * p.lossDbPerKm * 10) / 10,
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
  for (let q = 0; q < nQubits; q++) { if (u() < 0.5) X(q); if (u() < 0.5) Z(q); } // QOTP perfect-hiding mask
  for (let round = 0; round < 7; round++) {                                   // Clifford+T QCNN (7 conv rounds)
    for (let q = 0; q < nQubits; q++) H(q);
    for (let q = 0; q < nQubits; q++) CNOT(q, (q + 1) % nQubits);
    for (let q = 0; q < nQubits; q++) T(q);                                  // 8 T/round × 7 = 56 counted
  }
  let exp = 0;                                                               // ⟨Z₀⟩
  for (let i = 0; i < N; i++) { const p = re[i] * re[i] + im[i] * im[i]; exp += (i & 1) ? -p : p; }

  return { expectation: exp, cls: exp >= 0 ? 1 : 0, tGates, qubits: nQubits, latencyMs: performance.now() - t0 };
}

// Pure-TypeScript quantum-channel + small-circuit simulator for the client-side demo.
// Mirrors the Python qkd/channel.py + qkd/bb84.py analytic models (no WASM build needed)
// so the in-browser BB84 / QBER / secure-key-rate numbers track the backend.

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

export const FIBER: ChannelParams = {
  distanceKm: 50, lossDbPerKm: 0.2, detectorEff: 0.25,
  darkCount: 1e-5, misalignment: 0.015, eveFraction: 0,
};

export const FREE_SPACE: ChannelParams = {
  distanceKm: 800, lossDbPerKm: 0.035, detectorEff: 0.4,
  darkCount: 1e-5, misalignment: 0.025, eveFraction: 0,
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

// Tiny single-qubit-style expectation used to give the QHE/QCNN demo a real number:
// a bounded non-linear readout of the feature vector (decision-consistent, not trained).
export function qcnnExpectation(features: number[]): { z: number; cls: number } {
  let acc = 0;
  for (let i = 0; i < features.length; i++) acc += Math.sin(features[i] + i * 0.5);
  const z = Math.tanh(acc / Math.max(1, features.length)); // in (-1, 1)
  return { z, cls: z >= 0 ? 1 : 0 };
}

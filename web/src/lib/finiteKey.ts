// THE COMPOSABLE FINITE KEY OF A BLOCK (Round 79, the user's choice). Twin of
// `src/qsh/qkd/finite_key.py`, line for line, held to it by
// `tests/test_the_satellite_block_is_a_pass.py`.
//
// The satellite card's key was the asymptotic bound times a fixed 2e7 pulses, a fifth of a second
// of Micius's clock. It is now the finite-key length of a block of link time - one pass by default,
// the block every flown experiment computed its key on - with the bound that fits the source:
//
//   - Lim, Curty, Walenta, Xu & Zbinden, Phys. Rev. A 89, 022307 (2014), for a source that sends a
//     signal, a weak decoy and the vacuum, as Micius did;
//   - Rusca, Boaron, Grunenfelder, Martin & Zbinden, Appl. Phys. Lett. 112, 171104 (2018), for a
//     signal and one decoy.
//
// Both take COUNTS - detections and errors per basis and per intensity - so a pass, whose loss
// changes by ten decibels from its top to its edge, is the sum of its seconds' counts.
//
// THE STATISTICS (Round 82, the user's choice): each count's expectation bounded by Yin et al.
// 2020's variant of the Chernoff bound (Sci. Rep. 10, 14312, Eqs. (9)-(10)) and by the published
// Hoeffding deviation on the basis total, at eps/2 each, the tighter kept ("tighter", a union bound);
// or either alone. And Rusca's (A16) with tau0 on the error term only, as the paper prints it. The
// reasons are in `finite_key.py`'s docstring.

import { qkdCfg } from "./config";
import { deadTimeFactor, eDetForBound, sourceOf, transmittance, type ChannelParams } from "./quantumSim";

/** What a bound needs of the source: (mu, p) per intensity - signal, decoy, then the vacuum for
 *  Lim et al.'s protocol - the basis bias and the security budget. */
export interface FiniteKeySource {
  intensities: [number, number][];
  pZ: number;
  epsSec: number;
  epsCor: number;
  fEC: number;
  /** True: Lim et al. 2014 (vacuum + weak decoy). False: Rusca et al. 2018 (one decoy). */
  vacuum: boolean;
  /** The statistics: "tighter" of the two, "chernoff" (Yin et al. 2020) or "hoeffding" (published). */
  deviation: Deviation;
}

export type Deviation = "tighter" | "chernoff" | "hoeffding";
export const DEVIATIONS: Deviation[] = ["tighter", "chernoff", "hoeffding"];
/** How each is named where a key is reported: `finite_key.py::DEVIATION_LABELS`. */
export const DEVIATION_LABELS: Record<Deviation, string> = {
  tighter: "the tighter of Chernoff and Hoeffding per count",
  chernoff: "Chernoff-type bounds, Yin et al. 2020",
  hoeffding: "Hoeffding, as published",
};

/** Detections `n` and errors `m` in one basis, per intensity, in the source's order. */
export interface DecoyCounts { n: number[]; m: number[] }

export interface FiniteKeyResult {
  /** The key length in bits. Non-positive: no key. -Infinity: no single-photon bound at all. */
  keyBits: number;
  nZ: number; nX: number; sZ0: number; sZ1: number; phiZ: number; lambdaEC: number; qber: number;
}

export const FINITE_KEY_BOUNDS = {
  vacuum: "Lim et al. 2014 (vacuum + weak decoy)",
  oneDecoy: "Rusca et al. 2018 (one decoy)",
} as const;

const h2 = (x: number) => (x <= 0 || x >= 1 ? 0 : -x * Math.log2(x) - (1 - x) * Math.log2(1 - x));
const delta = (x: number, eps: number) => Math.sqrt((Math.max(x, 0) / 2) * Math.log(1 / eps));
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

/** Yin et al. 2020 Eq. (9): the largest expectation consistent with an observed count. */
export function chernoffUpper(x: number, eps: number): number {
  const b = Math.log(1 / eps), v = Math.max(x, 0);
  return v + b + Math.sqrt(2 * b * v + b * b);
}
/** Eq. (10), floored at zero as the paper requires. */
export function chernoffLower(x: number, eps: number): number {
  const b = Math.log(1 / eps), v = Math.max(x, 0);
  return Math.max(0, v - b / 2 - Math.sqrt(2 * b * v + (b * b) / 4));
}

/** Lower and upper bounds on each count's expectation at `eps` each: `finite_key.py::_bounds`. */
function bounds(counts: number[], eps: number, deviation: Deviation): [number[], number[]] {
  if (deviation === "hoeffding") {
    const d = delta(sum(counts), eps);
    return [counts.map((c) => c - d), counts.map((c) => c + d)];
  }
  if (deviation === "chernoff") return [counts.map((c) => chernoffLower(c, eps)), counts.map((c) => chernoffUpper(c, eps))];
  if (deviation === "tighter") {
    const [hLo, hHi] = bounds(counts, eps / 2, "hoeffding");
    const [cLo, cHi] = bounds(counts, eps / 2, "chernoff");
    return [hLo.map((v, i) => Math.max(v, cLo[i])), hHi.map((v, i) => Math.min(v, cHi[i]))];
  }
  throw new Error(`deviation ${deviation}: one of ${DEVIATIONS.join(", ")}`);
}

/** Rusca et al.'s (A13) step: `finite_key.py::_vacuum_errors_expectation`. */
function vacuumErrorsExpectation(v0: number, nTot: number, eps: number, deviation: Deviation): number {
  if (deviation === "hoeffding") return v0 + delta(nTot, eps);
  if (deviation === "chernoff") return chernoffUpper(v0, eps);
  return Math.min(v0 + delta(nTot, eps / 2), chernoffUpper(v0, eps / 2));
}
const factorial = (n: number): number => (n <= 1 ? 1 : n * factorial(n - 1));

function tau(n: number, src: FiniteKeySource): number {
  return sum(src.intensities.map(([mu, p]) => (p * Math.exp(-mu) * mu ** n) / factorial(n)));
}

/** The sampling correction to the phase error rate, 21 squared as both papers print it. */
export function gamma(a: number, b: number, c: number, d: number): number {
  return Math.sqrt((((c + d) * (1 - b) * b) / (c * d * Math.log(2)))
                   * Math.log2(((c + d) / (c * d * (1 - b) * b)) * (21 ** 2 / a ** 2)));
}

function gainAndError(eta: number, mu: number, eDet: number, y0: number): [number, number] {
  const q = y0 + 1 - Math.exp(-eta * mu);
  if (q <= 0) return [0, 0.5];
  return [q, Math.min(0.5, (eDet * (1 - Math.exp(-eta * mu)) + 0.5 * y0) / q)];
}

/** The bound's source for a channel: its own intensities and shares (Micius's on the downlink),
 *  both parties choosing a basis at even odds as Micius's passive beam splitter did. */
export function finiteKeySourceOf(ch: ChannelParams, epsSec = 1e-9,
                                  fEC = qkdCfg().error_correction_eff,
                                  deviation: Deviation = qkdCfg().finite_key_deviation as Deviation): FiniteKeySource {
  const s = sourceOf(ch);
  if (!DEVIATIONS.includes(deviation)) throw new Error(`deviation ${deviation}: one of ${DEVIATIONS.join(", ")}`);
  const common = { pZ: 0.5, epsSec, epsCor: 1e-15, fEC, deviation };
  return s.vacuum
    ? { ...common, vacuum: true, intensities: [[s.mu, s.pSignal], [s.nu, s.pDecoy], [0, 1 - s.pSignal - s.pDecoy]] }
    : { ...common, vacuum: false, intensities: [[s.mu, s.pSignal], [s.nu, 1 - s.pSignal]] };
}

/** The expected (Z, X) counts from `nPulses` channel uses on one channel. Linear in `nPulses`. */
export function expectedCounts(nPulses: number, ch: ChannelParams, src: FiniteKeySource): [DecoyCounts, DecoyCounts] {
  const eta = transmittance(ch);
  const eDet = eDetForBound(ch), y0 = 2 * ch.darkCount;
  const gains = src.intensities.map(([mu]) => gainAndError(eta, mu, eDet, y0));
  const cDt = deadTimeFactor(ch, sum(src.intensities.map(([, p], i) => p * gains[i][0])));
  const basis = (pB: number): DecoyCounts => {
    const n = src.intensities.map(([, p], i) => nPulses * p * pB * pB * cDt * gains[i][0]);
    return { n, m: n.map((nk, i) => nk * gains[i][1]) };
  };
  return [basis(src.pZ), basis(1 - src.pZ)];
}

/** The channel uses in a fibre key's block (Round 80): the run's own pulses, that many seconds at
 *  the clock, or as many as the key basis needs to collect `paBits` detections. Twin of
 *  `finite_key.py::fibre_block_pulses`. */
export function fibreBlockPulses(ch: ChannelParams, src: FiniteKeySource,
                                 b: { kind: string; runPulses: number; seconds: number; paBits: number }): number {
  if (b.kind === "pulses") return b.runPulses;
  if (b.kind === "seconds") return b.seconds * Number(ch.sourceRateHz);
  if (b.kind === "pa_bits") {
    const perPulse = sum(expectedCounts(1, ch, src)[0].n);
    return perPulse > 0 ? b.paBits / perPulse : Infinity;
  }
  throw new Error(`block kind ${b.kind}: one of pulses, seconds, pa_bits`);
}

export const addCounts = (a: DecoyCounts, b: DecoyCounts): DecoyCounts =>
  ({ n: a.n.map((v, i) => v + b.n[i]), m: a.m.map((v, i) => v + b.m[i]) });
export const scaleCounts = (a: DecoyCounts, k: number): DecoyCounts =>
  ({ n: a.n.map((v) => k * v), m: a.m.map((v) => k * v) });

/** Rusca et al. 2018's 1-decoy bound, `finite_key.py::one_decoy_key_length`. */
export function oneDecoyKeyLength(z: DecoyCounts, x: DecoyCounts, src: FiniteKeySource): FiniteKeyResult {
  const eps1 = src.epsSec / 20, eps2 = src.epsSec / 20;
  const [[mu1, p1], [mu2]] = src.intensities;
  const p2 = 1 - p1;
  const t0 = tau(0, src), t1 = tau(1, src);
  const basis = (c: DecoyCounts) => {
    const nTot = sum(c.n), mTot = sum(c.m);
    const [nLo, nHi] = bounds(c.n, eps1, src.deviation);
    const [mLo, mHi] = bounds(c.m, eps2, src.deviation);
    const nPlus1 = (Math.exp(mu1) / p1) * nHi[0];
    const nMinus2 = (Math.exp(mu2) / p2) * nLo[1];
    const mPlus1 = (Math.exp(mu1) / p1) * mHi[0];
    const mMinus2 = (Math.exp(mu2) / p2) * mLo[1];
    const s0 = Math.max(0, (t0 / (mu1 - mu2)) * (mu1 * nMinus2 - mu2 * nPlus1));
    // (A16), the decoy's errors, tau0 on them only (Round 82: it was on delta(n_Z) too).
    const v0 = t0 * (Math.exp(mu2) / p2) * mHi[1];
    const s0Upper = 2 * vacuumErrorsExpectation(v0, nTot, eps1, src.deviation);
    const s1 = ((t1 * mu1) / (mu2 * (mu1 - mu2))) * (
      nMinus2 - ((mu2 ** 2) / (mu1 ** 2)) * nPlus1 - (((mu1 ** 2) - (mu2 ** 2)) / (mu1 ** 2)) * (s0Upper / t0));
    const v1 = (t1 / (mu1 - mu2)) * (mPlus1 - mMinus2);
    return { n: nTot, m: mTot, s0, s1, v1, qber: nTot > 0 ? mTot / nTot : 0.5 };
  };
  const Z = basis(z), X = basis(x);
  const lambdaEC = src.fEC * Z.n * h2(Z.qber);
  if (Z.s1 <= 0 || X.s1 <= 0) {
    return { keyBits: -Infinity, nZ: Z.n, nX: X.n, sZ0: Z.s0, sZ1: Z.s1, phiZ: NaN, lambdaEC, qber: Z.qber };
  }
  const b = Math.min(0.5, Math.max(0, X.v1 / X.s1));
  const phi = b <= 0 ? 0 : Math.min(0.5, b + gamma(src.epsSec, b, Z.s1, X.s1));
  const keyBits = Z.s0 + Z.s1 * (1 - h2(phi)) - lambdaEC - 6 * Math.log2(19 / src.epsSec) - Math.log2(2 / src.epsCor);
  return { keyBits, nZ: Z.n, nX: X.n, sZ0: Z.s0, sZ1: Z.s1, phiZ: phi, lambdaEC, qber: Z.qber };
}

/** Lim et al. 2014's vacuum + weak bound, `finite_key.py::vacuum_decoy_key_length`. */
export function vacuumDecoyKeyLength(z: DecoyCounts, x: DecoyCounts, src: FiniteKeySource): FiniteKeyResult {
  const eps = src.epsSec / 21;
  const [[mu1], [mu2, p2], [mu3, p3]] = src.intensities;
  const t0 = tau(0, src), t1 = tau(1, src);
  const events = (c: DecoyCounts): [number, number] => {
    const [nLo, nHi] = bounds(c.n, eps, src.deviation);
    const lo = (k: number) => (Math.exp(src.intensities[k][0]) / src.intensities[k][1]) * nLo[k];
    const hi = (k: number) => (Math.exp(src.intensities[k][0]) / src.intensities[k][1]) * nHi[k];
    const s0 = Math.max(0, (t0 * (mu2 * lo(2) - mu3 * hi(1))) / (mu2 - mu3));                   // (2)
    const s1 = (t1 * mu1 * (lo(1) - hi(2) - (((mu2 ** 2) - (mu3 ** 2)) / (mu1 ** 2)) * (hi(0) - s0 / t0)))
      / (mu1 * (mu2 - mu3) - (mu2 ** 2) + (mu3 ** 2));                                            // (3)
    return [s0, s1];
  };
  const [sZ0, sZ1] = events(z);
  const [, sX1] = events(x);
  const nZ = sum(z.n), nX = sum(x.n);
  const qber = nZ > 0 ? sum(z.m) / nZ : 0.5;
  const lambdaEC = src.fEC * nZ * h2(qber);
  if (sZ1 <= 0 || sX1 <= 0) return { keyBits: -Infinity, nZ, nX, sZ0, sZ1, phiZ: NaN, lambdaEC, qber };
  const [mLo, mHi] = bounds(x.m, eps, src.deviation);
  const vX1 = (t1 * ((Math.exp(mu2) / p2) * mHi[1] - (Math.exp(mu3) / p3) * mLo[2])) / (mu2 - mu3);  // (4)
  const b = Math.min(0.5, Math.max(0, vX1 / sX1));
  const phi = b <= 0 ? 0 : Math.min(0.5, b + gamma(src.epsSec, b, sX1, sZ1));                    // (5)
  const keyBits = sZ0 + sZ1 * (1 - h2(phi)) - lambdaEC - 6 * Math.log2(21 / src.epsSec) - Math.log2(2 / src.epsCor);
  return { keyBits, nZ, nX, sZ0, sZ1, phiZ: phi, lambdaEC, qber };
}

export function keyLength(z: DecoyCounts, x: DecoyCounts, src: FiniteKeySource): FiniteKeyResult {
  return src.vacuum ? vacuumDecoyKeyLength(z, x, src) : oneDecoyKeyLength(z, x, src);
}

/** The summed counts of (pulses, channel) segments: a pass, or part of one. */
export function blockCounts(segments: [number, ChannelParams][], src: FiniteKeySource): [DecoyCounts, DecoyCounts] {
  let acc: [DecoyCounts, DecoyCounts] | null = null;
  for (const [n, ch] of segments) {
    const [dz, dx] = expectedCounts(n, ch, src);
    acc = acc ? [addCounts(acc[0], dz), addCounts(acc[1], dx)] : [dz, dx];
  }
  if (!acc) throw new Error("no segments: a block needs at least one");
  return acc;
}

/** The fewest copies of a block, pooled into one, that carry a positive key; null past `hi`. */
export function blocksForPositiveKey(z: DecoyCounts, x: DecoyCounts, src: FiniteKeySource, hi = 10_000): number | null {
  const positive = (k: number) => keyLength(scaleCounts(z, k), scaleCounts(x, k), src).keyBits > 0;
  if (positive(1)) return 1;
  if (!positive(hi)) return null;
  let lo = 1, top = hi;
  while (top - lo > 1) {
    const mid = Math.floor((lo + top) / 2);
    if (positive(mid)) top = mid; else lo = mid;
  }
  return top;
}

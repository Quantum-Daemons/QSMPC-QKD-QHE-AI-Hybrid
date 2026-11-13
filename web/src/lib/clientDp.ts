/**
 * Differential privacy for the medical federation and the genomics counts (Round 86, the owner's choice of 2026-11-13).
 *
 * WHAT IT IS. Noise added in shares inside the secure sum, the distributed discrete Gaussian of Kairouz, Liu and
 * Steinke (ICML 2021): every party adds an integer drawn from the discrete Gaussian N_Z(0, V) to its integer vector
 * before the secure aggregation, so the sum the aggregator recovers carries the noise of all of them and nothing else is
 * revealed. Their Theorem 1 bounds the privacy of the sum: with L2 sensitivity D2, n honest parties, per-party variance
 * V and dimension d, the release is (1/2) e^2-concentrated DP with
 *
 *     e = min( sqrt(D2^2 / (n V) + 2 tau d),  D2 / sqrt(n V) + tau sqrt(d) ),
 *     tau = 10 * sum_{k=1}^{n-1} exp(-2 pi^2 V k / (k + 1)),
 *
 * T releases compose to rho = T e^2 / 2, and rho-concentrated DP gives (eps, delta)-DP with
 * eps <= rho + 2 sqrt(rho log(1/delta)) (Canonne, Kamath and Steinke, NeurIPS 2020). The unit is one person's row,
 * added or removed. The guarantee counts only parties that add their noise; the page also states it for one fewer.
 *
 * THE NOISE IS EXACT. Floating-point Laplace and Gaussian samplers leak through their low bits (Mironov, CCS 2012), so
 * the sampler is Canonne, Kamath and Steinke's Algorithms 1 to 3, in integers and BigInt rationals, fed by the platform
 * CSPRNG (csprng.ts). The seeded source below exists for the Python twin's tests only: a run on it is NOT private,
 * because its noise is a function of a public label.
 *
 * NIST SP 800-226 (2025) is the guidance read: the unit of privacy stated, epsilon under 1 as strong and up to about 10
 * as common, delta at most 1/n^2 and in practice no more than 1e-5, and no floating-point noise.
 */
import { sha256 } from "@noble/hashes/sha2.js";
import { randomWords } from "./csprng";

/** A uniform integer in [0, m) for BigInt m >= 1. */
export type RandBelow = (m: bigint) => bigint;

/** From the platform CSPRNG, by rejection: the source every page run uses. */
export const cryptoRandBelow: RandBelow = (m) => {
  if (m <= 1n) return 0n;
  const bits = (m - 1n).toString(2).length;
  const words = Math.ceil(bits / 32);
  const mask = (1n << BigInt(bits)) - 1n;
  for (;;) {
    let x = 0n;
    for (const w of randomWords(words)) x = (x << 32n) | BigInt(w);
    x &= mask;
    if (x < m) return x;
  }
};

/**
 * NOT PRIVATE: a source both languages reproduce, SHA-256 of `label:counter` read as a big-endian integer and masked,
 * by rejection. For the twin tests, which hold the page's arithmetic to Python's, and for nothing a visitor sees.
 */
export function seededRandBelow(label: string): RandBelow {
  let counter = 0;
  const enc = new TextEncoder();
  return (m) => {
    if (m <= 1n) return 0n;
    const bits = (m - 1n).toString(2).length;
    const mask = (1n << BigInt(bits)) - 1n;
    for (;;) {
      const h = sha256(enc.encode(`${label}:${counter++}`));
      let x = 0n;
      for (const b of h) x = (x << 8n) | BigInt(b);
      x &= mask;
      if (x < m) return x;
    }
  };
}

const bernoulli = (n: bigint, d: bigint, rb: RandBelow) => rb(d) < n;           // Bernoulli(n/d)

function bernoulliExpNeg(n: bigint, d: bigint, rb: RandBelow): boolean {        // Bernoulli(exp(-n/d)), Algorithm 1
  if (n <= d) {
    let k = 1n;
    while (bernoulli(n, d * k, rb)) k++;
    return k % 2n === 1n;
  }
  for (let i = 0n; i < n / d; i++) if (!bernoulliExpNeg(1n, 1n, rb)) return false;
  return bernoulliExpNeg(n % d, d, rb);
}

function discreteLaplace(t: bigint, rb: RandBelow): bigint {                    // scale t, Algorithm 2 with s = 1
  for (;;) {
    const u = rb(t);
    if (!bernoulliExpNeg(u, t, rb)) continue;
    let v = 0n;
    while (bernoulliExpNeg(1n, 1n, rb)) v++;
    const x = u + t * v;
    const neg = rb(2n) === 1n;
    if (neg && x === 0n) continue;
    return neg ? -x : x;
  }
}

/** One draw from the discrete Gaussian N_Z(0, V) for an integer variance V >= 0 (Algorithm 3); V = 0 draws 0. */
export function discreteGaussian(variance: number, rb: RandBelow = cryptoRandBelow): number {
  if (!Number.isSafeInteger(variance) || variance < 0) throw new RangeError(`variance ${variance} is not a safe integer >= 0`);
  if (variance === 0) return 0;
  const v = BigInt(variance);
  let lo = 0n, hi = 1n;                         // t = floor(sigma) + 1, from V without floating point
  while (hi * hi <= v) hi *= 2n;
  while (hi - lo > 1n) { const mid = (lo + hi) / 2n; if (mid * mid <= v) lo = mid; else hi = mid; }
  const t = lo + 1n;
  for (;;) {
    const y = discreteLaplace(t, rb);
    const a = y < 0n ? -y : y;
    const g = a * t - v;                        // (|y| - V/t)^2 / (2V) = g^2 / (2 V t^2)
    if (bernoulliExpNeg(g * g, 2n * v * t * t, rb)) return Number(y);
  }
}

/** Kairouz, Liu and Steinke's tau for n parties of per-party variance V. */
export function ddgTau(n: number, variance: number): number {
  let s = 0;
  for (let k = 1; k < n; k++) s += Math.exp(-2 * Math.PI * Math.PI * variance * k / (k + 1));
  return 10 * s;
}

/** The concentrated-DP parameter e of one release (their Theorem 1); the release is (1/2) e^2-CDP. */
export function ddgEpsilon(d2: number, n: number, variance: number, d: number): number {
  const tau = ddgTau(n, variance);
  return Math.min(Math.sqrt(d2 * d2 / (n * variance) + 2 * tau * d), d2 / Math.sqrt(n * variance) + tau * Math.sqrt(d));
}

/** rho-concentrated DP to (eps, delta)-DP: eps <= rho + 2 sqrt(rho log(1/delta)). */
export function cdpToDp(rho: number, delta: number): number {
  return rho + 2 * Math.sqrt(rho * Math.log(1 / delta));
}

/** The (eps, delta)-DP of T releases, each the sum of n parties' noise of variance V. */
export function epsilonOf(variance: number, releases: number, d2: number, n: number, d: number, delta: number): number {
  if (variance === 0) return Infinity;
  const e = ddgEpsilon(d2, n, variance, d);
  return cdpToDp(releases * e * e / 2, delta);
}

/** The least integer per-party variance whose T releases are (target, delta)-DP; 0 for an infinite target. */
export function varianceFor(target: number, releases: number, d2: number, n: number, d: number, delta: number): number {
  if (!Number.isFinite(target)) return 0;
  let hi = 1;
  while (epsilonOf(hi, releases, d2, n, d, delta) > target) {
    hi *= 2;
    if (hi > 2 ** 52) throw new RangeError(`no variance below 2^52 reaches epsilon ${target}`);
  }
  let lo = Math.floor(hi / 2);          // epsilonOf(lo) > target, or lo = 0
  while (hi - lo > 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (epsilonOf(mid, releases, d2, n, d, delta) <= target) hi = mid; else lo = mid;
  }
  return hi;
}

/**
 * The genomics page's pooled allele count, released with differential privacy (Round 86). The unit is one person, who
 * carries at most two copies of the effect allele, so one person added or removed moves the count by at most 2. Each
 * cohort contributes its count in the fixed point 1/16, plus its share of the noise, and the secure sum adds the
 * contributions; the release divided by 16 is the count with noise. The fixed point exists so that every epsilon on
 * the selector is reachable: in whole alleles the least noise this sampler draws, variance 1 per cohort, is already
 * epsilon 5.3, so 8 could not be asked for.
 */
export const GENOMICS_DP = { scale: 16, sensitivity: 2, delta: 1e-5, epsilons: [0.5, 1, 2, 4, 8], defaultEpsilon: 1 };

export interface GenomicsDp {
  /** Each cohort's integer contribution, its count times `scale` plus its noise: what it secret-shares. */
  contributions: number[];
  epsilon: number; epsilonAchieved: number; epsilonOneDishonest: number; delta: number;
  variancePerCohort: number; scale: number;
  /** The noise in the released count, in alleles. */
  sdTotal: number;
}

/** `seededLabel` is for the twin tests only: that run is NOT private. */
export function genomicsDp(counts: number[], epsilon: number, opts?: { seededLabel?: string }): GenomicsDp {
  const { scale, sensitivity, delta } = GENOMICS_DP;
  const rb = opts?.seededLabel != null ? seededRandBelow(opts.seededLabel) : cryptoRandBelow;
  const n = counts.length, d2 = sensitivity * scale;
  const V = varianceFor(epsilon, 1, d2, n, 1, delta);
  return {
    contributions: counts.map((c) => c * scale + discreteGaussian(V, rb)),
    epsilon, epsilonAchieved: epsilonOf(V, 1, d2, n, 1, delta), epsilonOneDishonest: epsilonOf(V, 1, d2, n - 1, 1, delta),
    delta, variancePerCohort: V, scale, sdTotal: Math.sqrt(n * V) / scale,
  };
}

/**
 * This tab's differential-privacy releases, composed (Round 86, the owner's choice of 2026-11-22: a ledger of the
 * releases a rerun repeats).
 *
 * WHY THIS EXISTS. Each Start draws fresh noise, so each run is its own release about the same people: the genomics
 * page's pooled allele count about the same 1000 Genomes individuals, the finance page's count of shared accounts about
 * the same two banks' accounts, the medical page's four noised federated models about the same patients. Each card
 * stated one run's epsilon, and the genomics card said in words that k runs together are k times the rho of one;
 * nothing counted the runs. This ledger records every release the tab makes and composes each population's: zero-
 * concentrated DP adds rho (Bun and Steinke, TCC 2016), and the total converts to (epsilon, delta) once, by the page's
 * own conversion (`composedEpsilon`).
 *
 * What it cannot see: other tabs, earlier visits (a reload empties it) and anyone else's runs. It holds no data, only
 * each release's label, rho and delta, and nothing of it leaves the tab.
 */
import { composedEpsilon } from "./clientDp";

/** Whose data each page's releases are about, and the unit each release protects: releases compose within a
 *  population, never across. */
export const POPULATIONS = {
  genomics: { who: "the 1000 Genomes individuals bundled with the page", unit: "person", aUnit: "a person", units: "people" },
  finance: { who: "the two banks' accounts", unit: "account", aUnit: "an account", units: "accounts" },
  "medical-heart": { who: "the four hospitals' patients", unit: "patient", aUnit: "a patient", units: "patients" },
} as const;
export type Population = keyof typeof POPULATIONS;

export interface Release {
  population: Population;
  /** What was released, as the card names it. */
  what: string;
  rho: number;
  delta: number;
  /** The release's own epsilon at that delta, as its card states it. */
  epsilon: number;
}

const releases: Release[] = [];
const listeners = new Set<() => void>();
let version = 0;

/**
 * Record one release. A release must have a finite rho above 0: a row with clipping and no noise is not a DP release,
 * and the call sites leave it out. A population's releases share one delta, the one its page states.
 */
export function recordRelease(r: Release): void {
  if (!(Number.isFinite(r.rho) && r.rho > 0)) throw new RangeError(`a DP release has a finite rho above 0, got ${r.rho}`);
  const earlier = releases.find((x) => x.population === r.population);
  if (earlier && earlier.delta !== r.delta) {
    throw new RangeError(`${r.population}: delta ${r.delta} differs from the ${earlier.delta} of its earlier releases`);
  }
  releases.push({ ...r });
  version += 1;
  for (const l of listeners) l();
}

export interface LedgerSummary {
  population: Population;
  releases: number;
  rho: number;
  delta: number;
  /** All of the population's releases in this tab, published together. */
  epsilon: number;
  last: Release;
}

export function ledgerSummary(p: Population): LedgerSummary | null {
  const rs = releases.filter((x) => x.population === p);
  if (rs.length === 0) return null;
  const rhos = rs.map((x) => x.rho);
  return {
    population: p, releases: rs.length, rho: rhos.reduce((a, b) => a + b, 0), delta: rs[0].delta,
    epsilon: composedEpsilon(rhos, rs[0].delta), last: rs[rs.length - 1],
  };
}

export function subscribeLedger(listener: () => void): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

export function ledgerVersion(): number {
  return version;
}

/** For the tests only. The page never calls it: a release, once made, cannot be unmade. */
export function forgetEveryReleaseForTests(): void {
  releases.length = 0;
  version += 1;
}

/**
 * What (epsilon, delta)-DP means for one person, exactly: any test of whether their record is in the data that says
 * "in" wrongly with probability `fpr` says it rightly with probability at most
 * min(1, e^epsilon fpr + delta, 1 - e^-epsilon (1 - delta - fpr)) (Kairouz, Oh and Viswanath, ICML 2015).
 */
export function bestTestPower(epsilon: number, delta: number, fpr: number): number {
  if (!Number.isFinite(epsilon)) return 1;
  return Math.min(1, Math.exp(epsilon) * fpr + delta, 1 - Math.exp(-epsilon) * (1 - delta - fpr));
}

/**
 * The factor by which a release can multiply any observer's odds that one person's record is in the data, when delta
 * is 0; with delta above 0 this is the reading of epsilon alone, and `bestTestPower` is the exact statement.
 */
export function oddsFactor(epsilon: number): number {
  return Math.exp(epsilon);
}

/** The largest posterior that a factor of e^epsilon on the odds allows from a prior probability `prior`. */
export function posteriorAtMost(epsilon: number, prior: number): number {
  const f = oddsFactor(epsilon);
  return (f * prior) / (f * prior + (1 - prior));
}

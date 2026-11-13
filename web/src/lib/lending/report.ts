/** What the lending page's two stages leave for its cards (Round 86), and how a pool's raw results become that. */
import type { LendingCfg } from "../config";
import type { LendingRun } from "./model";
import type { PoolRun } from "./pool";
import type { CommitteeSummary, Custody, KeyholderSetup, PrivateCheckOutcome } from "./privateCheck";

export interface PoolSummary { workers: number; ranIn: string; logicalCores: number; wallMs: number; speedUp: number; utilisation: number; busyMs: number[] }

export interface SweepRow {
  multiplier: number;
  /** Liquidated debt as a share of the opening debt, in per cent: the mean over the price paths and their 10th and 90th percentiles. */
  repaidPct: number; repaidP10: number; repaidP90: number;
  liquidationsMean: number;
  /** The share of price paths that ended with any bad debt, in per cent. */
  pathsWithBadDebtPct: number;
}

export interface LendingMarketReport {
  cfg: LendingCfg;
  run: LendingRun;
  simulateMs: number;
}

export interface LendingSweepReport { rows: SweepRow[]; seeds: number; pool: PoolSummary }

export interface LendingPrivateReport {
  custody: Custody;
  outcomes: PrivateCheckOutcome[];
  /** How many times Bob had to send anything: once under Carol's or the committee's custody, once per check under his own. */
  bobSubmissions: number;
  /** Carol's and the committee's custody. */
  setup: { submissionBytes: number; keygenMs: number; encryptMs: number; linkAgreed: boolean } | null;
  /** Carol's and the committee's custody: every signed bit checked against her public key. */
  bitSignaturesVerified: boolean | null;
  /** The committee's custody only: its parameters, once. */
  committee: CommitteeSummary | null;
  allAgree: boolean;
  flagged: number;
  /** In thousands of stable units. */
  maxAbsError: number;
  statisticalDistance: number;
  /** null when the distance is zero: the committee's mask is uniform modulo t and hides the margin perfectly. */
  statisticalDistanceLog2: number | null;
  maskBits: number;
  ckksBytes: number;
  keyBytes: number;
  gcBytes: number;
  gcMessages: number;
  andGates: number;
  signature: string;
  signatureBytes: number;
  keygenMs: number;
  ckksMs: number;
  gcMs: number;
  pqcMs: number;
  pool: PoolSummary;
}

const poolSummary = (p: PoolRun): PoolSummary => ({ workers: p.workers, ranIn: p.ranIn, logicalCores: p.logicalCores, wallMs: p.wallMs, speedUp: p.speedUp, utilisation: p.utilisation, busyMs: p.busyMs });

function quantile(sorted: number[], q: number): number {
  const at = (sorted.length - 1) * q, low = Math.floor(at), high = Math.ceil(at);
  return sorted[low] + (sorted[high] - sorted[low]) * (at - low);
}

export function summariseSweep(pool: PoolRun): LendingSweepReport {
  const rows = pool.results.map((r): SweepRow => {
    if (r.kind !== "sweep") throw new Error("a private check came back to the sweep");
    const o = r.outcome;
    const pct = o.repaid.map((v, i) => 100 * v / o.openingDebt[i]).sort((a, b) => a - b);
    const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
    return {
      multiplier: o.multiplier, repaidPct: mean(pct), repaidP10: quantile(pct, 0.1), repaidP90: quantile(pct, 0.9),
      liquidationsMean: mean(o.liquidations), pathsWithBadDebtPct: 100 * o.badDebt.filter((v) => v > 0).length / o.badDebt.length,
    };
  }).sort((a, b) => a.multiplier - b.multiplier);
  const first = pool.results[0];
  return { rows, seeds: first.kind === "sweep" ? first.outcome.liquidations.length : 0, pool: poolSummary(pool) };
}

export function summariseChecks(pool: PoolRun, signature: string, maskBits: number, custody: Custody, setup?: KeyholderSetup): LendingPrivateReport {
  const checks = pool.results.map((r) => { if (r.kind !== "check") throw new Error("a sweep level came back to the private check"); return r; });
  const outcomes = checks.map((c) => c.outcome).sort((a, b) => a.hour - b.hour);
  const sum = (f: (o: PrivateCheckOutcome) => number) => outcomes.reduce((s, o) => s + f(o), 0);
  const statisticalDistance = sum((o) => o.statisticalDistance);
  const committee = checks[0]?.committee ?? null;
  const once = custody === "keyholder" || custody === "committee";
  return {
    custody, bobSubmissions: once ? 1 : outcomes.length,
    setup: setup ? { submissionBytes: setup.submission.length, keygenMs: setup.keygenMs, encryptMs: setup.encryptMs, linkAgreed: setup.linkAgreed }
      : committee ? { submissionBytes: committee.submissionBytes, keygenMs: committee.keyGenMs, encryptMs: committee.encryptMs, linkAgreed: committee.linkAgreed } : null,
    bitSignaturesVerified: once ? outcomes.every((o) => o.bitSignatureVerified === true) : null,
    committee,
    outcomes, allAgree: outcomes.every((o) => o.agrees), flagged: outcomes.filter((o) => o.flag).length,
    maxAbsError: Math.max(...outcomes.map((o) => o.absError)), statisticalDistance, statisticalDistanceLog2: statisticalDistance > 0 ? Math.log2(statisticalDistance) : null, maskBits,
    ckksBytes: sum((o) => o.ckksBytes), keyBytes: checks[0]?.keyBytes ?? 0, gcBytes: sum((o) => o.gcBytes), gcMessages: sum((o) => o.gcMessages), andGates: sum((o) => o.andGates),
    signature, signatureBytes: checks[0]?.signatureBytes ?? 0, keygenMs: Math.max(...checks.map((c) => c.keygenMs)),
    ckksMs: sum((o) => o.encryptMs + o.evaluateMs + o.decryptMs), gcMs: sum((o) => o.gcMs), pqcMs: sum((o) => o.signMs + o.verifyMs + o.sealMs),
    pool: poolSummary(pool),
  };
}

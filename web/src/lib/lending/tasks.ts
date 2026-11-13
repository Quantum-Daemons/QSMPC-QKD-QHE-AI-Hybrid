/**
 * What the lending page runs off its own thread (Round 86). A task is either one position's private check or one volatility level of the sweep; both are independent
 * of each other, so a pool of workers can take them in any order. Kept apart from the pool so the node harness can run the same code without a Worker.
 */

import type { LendingCfg } from "../config";
import type { CheckSession, CommitteeSummary, Custody, KeyholderSetup, PrivateCheckOutcome } from "./privateCheck";
import { simulate, type LendingCheckpoint } from "./model";

export interface SweepOutcome {
  multiplier: number;
  /** One entry per seed. */
  liquidations: number[];
  repaid: number[];
  badDebt: number[];
  openingDebt: number[];
}

export type LendingTask =
  | { kind: "setup"; input: LendingCheckpoint }
  | { kind: "check"; input: LendingCheckpoint }
  | { kind: "sweep"; cfg: LendingCfg; multiplier: number; seeds: number[] };

export type LendingTaskResult =
  | { kind: "setup"; setup: KeyholderSetup }
  | { kind: "check"; outcome: PrivateCheckOutcome; keyBytes: number; signatureBytes: number; keygenMs: number; committee?: CommitteeSummary }
  | { kind: "sweep"; outcome: SweepOutcome };

/** The check's keys and the circuit's parameters. Absent for a pool that only sweeps. */
export interface TaskInit {
  sessionKey?: Uint8Array;
  sig?: string;
  cfg?: Pick<LendingCfg, "margin_scale_bits" | "margin_mask_bits">;
  custody?: Custody;
  /** The selected KEM, for the pool and Carol's handshake. */
  kem?: string;
  /** Carol's setup, for her checks. */
  keyholder?: KeyholderSetup;
  /** The committee's checks: the position Bob encrypts once, and where the threshold module is served. */
  first?: LendingCheckpoint;
  fedThreshBase?: string;
}

/** One volatility level: every seed, with the volatility scaled and no scripted shock, so that the only thing changing along the curve is volatility. */
export function sweepLevel(cfg: LendingCfg, multiplier: number, seeds: number[]): SweepOutcome {
  const level: LendingCfg = { ...cfg, annual_vol: cfg.annual_vol.map((v) => v * multiplier), shock_fraction: 0 };
  const out: SweepOutcome = { multiplier, liquidations: [], repaid: [], badDebt: [], openingDebt: [] };
  for (const seed of seeds) {
    const run = simulate(level, seed);
    out.liquidations.push(run.events.length);
    out.repaid.push(run.events.reduce((sum, e) => sum + e.repaid, 0));
    out.badDebt.push(run.badDebt);
    out.openingDebt.push(run.totalDebt[0]);
  }
  return out;
}

let session: Promise<CheckSession> | null = null;

export function closeTaskSession(): void {
  const s = session;
  session = null;
  void s?.then((x) => x.close());
}

export async function executeTask(task: LendingTask, init: TaskInit): Promise<LendingTaskResult> {
  if (task.kind === "sweep") return { kind: "sweep", outcome: sweepLevel(task.cfg, task.multiplier, task.seeds) };
  if (!init.sessionKey || !init.sig || !init.cfg) throw new Error("a private check needs the session key, the signature scheme and the circuit's parameters");
  if (task.kind === "setup") {
    const { setupKeyholder } = await import("./privateCheck");
    return { kind: "setup", setup: await setupKeyholder({ sessionKey: init.sessionKey, sig: init.sig, kem: init.kem ?? "ml-kem-768" }, task.input) };
  }
  const opts = { sessionKey: init.sessionKey, sig: init.sig, cfg: init.cfg, custody: init.custody ?? "borrower" as const, keyholder: init.keyholder,
                 kem: init.kem, first: init.first, fedThreshBase: init.fedThreshBase };
  session ??= import("./privateCheck").then((m) => m.openCheckSession(opts));
  const s = await session;
  return { kind: "check", outcome: await s.check(task.input), keyBytes: s.keyBytes, signatureBytes: s.signatureBytes, keygenMs: s.keygenMs, committee: s.committee };
}

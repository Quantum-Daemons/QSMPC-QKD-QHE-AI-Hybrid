/**
 * A run's recorded log as a timeline (Round 86, the owner's choice of 2026-11-14).
 *
 * One list of events, the one the live log already shows, drives the canvas, the log, the metrics table, the per-party
 * panel and the animation exports, and a cursor replays it: back, forward and scrub over what the run did, without
 * running it again. Replaying is folding the first `cursor` events with the same rules `usePipelineRun`'s
 * `handleEvent` applies live, so a position in the replay shows what the page showed at that moment of the run.
 * Nothing is recomputed and no cryptographic generator is re-seeded: a replay re-reads the record.
 *
 * WHAT CROSSES AND WHAT EACH PARTY HOLDS. Two event types carry the per-party view, emitted by the stages
 * (`clientCompute.ts`, `message()` and `holds()`):
 *   message  one side to another, what it is, its size where the run measured it (null where it did not), what
 *            protects it on the wire, and for public bytes the run keeps (a public key, a KEM ciphertext, a
 *            signature) their first bytes in hex;
 *   holds    what a side holds or has learned at that point, and whether it keeps it to itself.
 * Hex is only ever of bytes that cross in the clear. A key, a seed, a share or a set member never goes into an event:
 * the log is exported as JSON and copied into the results bundle verbatim. The sources the design follows and where it
 * stops: references/run-timeline-and-party-views.md.
 */
import type { PipelineEvent } from "./api";
import { USECASE_SPECS, type DiagramConfig } from "./useCaseFlows";

/** A side of the canvas ("a", "b"), the many parties of a many-party stage ("many"), "anyone" for what is published,
 *  or a party the lending page names on its own (a committee member). */
export type Side = string;

/** What protects a message on the wire. */
export type Protection =
  | "public" | "session key" | "CKKS" | "BGV" | "garbled circuit" | "additive shares" | "pairwise masks"
  | "photons (simulated)" | "classical (simulated)";

export const PROTECTION_NOTE: Record<Protection, string> = {
  public: "sent in the clear, and safe to be: a public key, a KEM ciphertext, a signature",
  "session key": "sealed with a key derived from the PQC session key",
  CKKS: "a CKKS ciphertext: only the holder of the secret key can read it",
  BGV: "a threshold BGV ciphertext or share: no single party holds the key",
  "garbled circuit": "garbled tables and oblivious-transfer messages between the two notional parties",
  "additive shares": "random-looking shares mod p: one share alone says nothing",
  "pairwise masks": "updates hidden under pairwise masks that cancel in the sum",
  "photons (simulated)": "a classical simulation of the quantum channel: no photons are sent",
  "classical (simulated)": "the protocol's classical messages, modelled, not sent byte for byte",
};

export interface MessageData {
  from: Side; to: Side; what: string;
  /** Bytes on the wire as the run measured them, summed over `count` messages; null where the run does not measure them. */
  bytes: number | null;
  protection: Protection;
  /** How many messages this line stands for, where one line groups a stage's repeated messages. */
  count?: number;
  /** The first bytes of a public value, in hex, with "..." when there are more. */
  hex?: string;
}

export interface HoldsData {
  party: Side; what: string;
  /** True for what this party keeps to itself and never sends. */
  kept: boolean;
}

/** What the page draws from the log at one position. */
export interface RunView { active: string | null; done: Set<string>; metrics: Record<string, unknown> }

/** The first `upTo` events folded the way `usePipelineRun`'s `handleEvent` folds them live. */
export function foldRun(events: PipelineEvent[], upTo: number = events.length): RunView {
  let active: string | null = null;
  const done = new Set<string>();
  const metrics: Record<string, unknown> = {};
  for (const e of events.slice(0, Math.max(0, upTo))) {
    if (e.type === "stage_start") active = e.stage;
    if (e.type === "stage_end") { active = null; done.add(e.stage); }
    if (e.type === "metric") metrics[String(e.data?.name)] = e.data?.value;
  }
  return { active, done, metrics };
}

/** The positions a step lands on: before anything, just after each stage starts and just after each stage ends, and
 *  the end of the log. A position is a count of events applied. */
export function stageMarks(events: PipelineEvent[]): number[] {
  const out = [0];
  events.forEach((e, i) => { if (e.type === "stage_start" || e.type === "stage_end") out.push(i + 1); });
  out.push(events.length);
  return [...new Set(out)].sort((x, y) => x - y);
}
export const prevMark = (marks: number[], at: number): number => [...marks].reverse().find((m) => m < at) ?? 0;
export const nextMark = (marks: number[], at: number): number => marks.find((m) => m > at) ?? (marks[marks.length - 1] ?? 0);

/** The stage running or last finished at a position, for the timeline's caption. */
export function stageAt(events: PipelineEvent[], upTo: number): string | null {
  for (let i = Math.min(upTo, events.length) - 1; i >= 0; i--) {
    if (events[i].type === "stage_start" || events[i].type === "stage_end") return events[i].stage;
  }
  return null;
}

export interface LedgerMessage extends MessageData { stage: string; t: number }
export interface LedgerHolding extends HoldsData { stage: string; t: number }

/** Every message sent, and everything each side holds, up to a position. */
export function ledgerAt(events: PipelineEvent[], upTo: number = events.length): {
  messages: LedgerMessage[]; holdings: Record<Side, LedgerHolding[]>;
} {
  const messages: LedgerMessage[] = [];
  const holdings: Record<Side, LedgerHolding[]> = {};
  for (const e of events.slice(0, Math.max(0, upTo))) {
    if (e.type === "message") messages.push({ ...(e.data as unknown as MessageData), stage: e.stage, t: e.t });
    if (e.type === "holds") {
      const h = e.data as unknown as HoldsData;
      (holdings[h.party] ??= []).push({ ...h, stage: e.stage, t: e.t });
    }
  }
  return { messages, holdings };
}

/** The first `n` bytes as hex, with "..." when there are more: enough to tell two values apart. Only for bytes that
 *  cross in the clear. */
export function hexPrefix(u: Uint8Array, n = 8): string {
  return [...u.slice(0, n)].map((b) => b.toString(16).padStart(2, "0")).join("") + (u.length > n ? "..." : "");
}

/** Who the many parties of a page's many-party stage are, one at a time. */
const MANY: Record<string, string> = {
  medical_fl: "each hospital", genomics: "each cohort", finance_aml: "each client", demo: "each party",
};

/** A side as the canvas heads its column, so the panel and the log name the parties the diagram draws. */
export function sideName(usecase: string, side: Side, c: DiagramConfig = {}): string {
  const spec = USECASE_SPECS[usecase];
  const heads = spec?.parties?.(c) ?? (spec ? { a: spec.a, b: spec.b } : null);
  if (side === "a") return heads?.a.name ?? "Party A";
  if (side === "b") return heads?.b.name ?? "Party B";
  if (side === "many") return MANY[usecase] ?? "each party";
  return side;
}

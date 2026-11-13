// REAL in-browser 2-party MPC for the non-linear decision layer (argmax over the FHE logits), on
// Polytune (sine-fdn, MIT): WRK17b multi-party authenticated garbling, secure against malicious
// adversaries controlling all but one party - the protocol's property, in a pre-release library whose
// open upstream issues docs/threat-model.md assesses (Rounds 68 and 72; at the pinned commit nothing
// here claims malicious security between hosts). BOTH parties run inside this tab over
// an in-memory channel, so there is ZERO server / ZERO VPS - yet the protocol, its messages and its
// wall-clock are GENUINELY the real garbled-circuit computation (not a hand-rolled reconstruction, not
// a cost model).
// Each quantised logit is additively secret-shared modulo 2^32 (q_k = a_k + b_k mod 2^32), with a_k
// uniform over 32 bits from the browser's CSPRNG, so each share on its own is uniform whatever the
// logit. The circuit privately reconstructs and compares the values and outputs ONLY the argmax index.
//
// Honest scope: in one browser the two "parties" necessarily share a JS heap and a thread, so the
// party separation is notional (it must be, with zero VPS) - what is genuinely real is the
// cryptographic protocol, the measured time and the real communication. The full deployment's MPyC
// tier runs three Shamir parties as separate processes, on one host over loopback TCP.
//
// REPLACED IN ROUND 65. This file ran mpc-framework + emp-wasm-engine + summon-ts until then: all three
// archived upstream, and mpc-framework pulled the zod release carrying GHSA-mrhx-c7xc-m3w5 with no fix
// coming. Polytune was prototyped (Round 63) and measured between two tabs over netem-shaped TCP
// (Round 64): on this 2-class argmax it was faster on every link measured, though it sends about five
// times the bytes. The module is built from wasm/qsh-mpc by the web image's Rust stage, where its
// `cargo test` runs on every build, or locally by scripts/build_wasm.sh.
//
// THE MASKS, ROUND 66. Until then a_k was drawn from the run's seeded generator (mulberry32, keyed on a
// seed the page shows) and bounded to 2^16, with b_k = q_k - a_k + 2^16. Anyone with the seed could
// recompute every mask, and a bounded mask hides a value only statistically: b_k for two logits up to
// QMAX apart differ in distribution by up to QMAX / 2^16 = 1/64. The masks now come from
// csprng.ts and the sharing is modulo 2^32, which also removed a subtraction per logit from the circuit
// (367 AND gates at 2 classes, from 400).

import { randomWords } from "./csprng";

export interface MPCResult {
  out: number[];        // [argmax index]
  compares: number;     // comparisons in the argmax (N-1)
  commBytes: number;    // REAL measured garbled-circuit communication, both directions
  mpcMs: number;        // REAL measured wall-clock of the protocol (module init cached via warmMPC)
  messages?: number;    // messages exchanged - over a network each one would cost a latency
  /** The index the same computation gives IN THE CLEAR, on the same quantised values, with the same
   *  tie rule (the first maximum). A self-test, available only because both parties are in this tab:
   *  the caller compares it with `out` and reports a disagreement rather than trusting either one. */
  clearOut: number[];
}

// Fixed-point parameters. OFFSET makes the quantised value non-negative, and the clamp keeps it at most
// QMAX, so the reconstructed values compare correctly as unsigned 32-bit integers. Round 86 (review
// PI-4): these were 64 and 8, the emp-wasm version's, so a logit gap under 1/64 tied and went to the
// lower index, and logits beyond +-8 clamped together; 22 of medical's 171 held-out rows exceed 8. The
// shares have been modulo 2^32 since Round 66, so the range is free: 2^-20 steps over +-512, at most
// 2^30, and the circuit, which reconstructs in u64 and casts back to u32, is unchanged.
export const SCALE = 2 ** 20; // fixed-point scale for the float logits
export const OFFSET = 512;   // logits assumed within [-OFFSET, OFFSET] -> quantised value is positive
const QMAX = 2 * OFFSET * SCALE; // 2^30, the max quantised value (logits clamped to it)

type QshMpc = typeof import("../wasm/qsh_mpc/qsh_mpc.js");

let _mod: Promise<QshMpc> | null = null;

// Dynamic import, so the wasm stays out of the entry bundle and loads on first use.
function getModule(): Promise<QshMpc> {
  return (_mod ??= (async () => {
    const m = await import("../wasm/qsh_mpc/qsh_mpc.js");
    await m.default();
    return m;
  })());
}

/** Load and instantiate the wasm so the first real inference is not also the first download. */
export async function warmMPC(): Promise<void> {
  await getModule();
}

/**
 * Genuine 2-party secure argmax over logits shared modulo 2^32 - both parties run Polytune's
 * authenticated-garbling protocol inside this tab. Returns the argmax index plus the REAL measured
 * protocol time and communication. It takes no generator: the masks are the CSPRNG's.
 */
export async function secureArgmax(logits: number[]): Promise<MPCResult> {
  const n = logits.length;
  if (n < 2) return { out: [0], compares: 0, commBytes: 0, mpcMs: 0, messages: 0, clearOut: [0] };
  const m = await getModule();

  // quantise + share each logit: q_k = clamp(round((logit_k+OFFSET)*SCALE), 0, QMAX); a_k uniform
  // over 32 bits; b_k = q_k - a_k mod 2^32. The circuit computes a_k + b_k mod 2^32 = q_k privately.
  const a = randomWords(n);
  const b = new Uint32Array(n);
  // `clear` is the same argmax in the clear, for the caller's self-test. Strictly greater, so the first
  // maximum wins, as in the circuit.
  let clear = 0, best = -1;
  for (let k = 0; k < n; k++) {
    const q = Math.min(QMAX, Math.max(0, Math.round((logits[k] + OFFSET) * SCALE)));
    b[k] = (q - a[k]) >>> 0;
    if (q > best) { best = q; clear = k; }
  }

  const r = await m.secure_argmax(a, b) as { arg: number; mpcMs: number; bytes: number; messages: number };
  return { out: [r.arg], compares: n - 1, commBytes: r.bytes, mpcMs: r.mpcMs, messages: r.messages,
           clearOut: [clear] };
}

/** One batch of ReLUs under Polytune, as `clientHybridCnn` calls it. `out` is what party 0 receives: each ReLU plus party 1's mask. */
export interface ReluBatchResult { out: number[]; compileMs: number; mpcMs: number; bytes: number; messages: number; andOps: number; }

export async function secureReluBatch(y: Uint32Array, r: Uint32Array, s: BigUint64Array, ringBits: number, outBits: number): Promise<ReluBatchResult> {
  const m = await getModule();
  const o = await m.secure_relu(y, r, s, ringBits, outBits) as Omit<ReluBatchResult, "out"> & { out: BigUint64Array };
  return { ...o, out: Array.from(o.out, Number) };
}

/** One n-party PSI under Polytune, as `clientMultiPsi` calls it: for each element of the first set, that element if every other set holds it, else 0. */
export interface MultiPartyPsiOutcome { ids: number[]; compileMs: number; mpcMs: number; bytes: number; messages: number; andOps: number; }

export async function secureMultiPartyPsi(sets: Uint32Array[]): Promise<MultiPartyPsiOutcome> {
  const m = await getModule();
  const k = sets[0].length;
  const flat = new Uint32Array(sets.length * k);
  sets.forEach((s, i) => flat.set(s, i * k));
  const o = await m.secure_psi(flat, sets.length, k) as Omit<MultiPartyPsiOutcome, "ids"> & { ids: Uint32Array };
  return { ...o, ids: Array.from(o.ids) };
}

/** The sign of a value shared modulo an odd t below 2^53 (the lending committee's): true when (a + b) mod t, read in (-t/2, t/2), is negative. */
export async function secureSignMod(a: number, b: number, t: number): Promise<{ negative: boolean; compileMs: number; mpcMs: number; bytes: number; messages: number; andOps: number }> {
  for (const v of [a, b, t]) if (!Number.isSafeInteger(v) || v < 0) throw new Error(`secureSignMod: ${v} is not a non-negative integer below 2^53`);
  const m = await getModule();
  return await m.secure_sign_mod(BigInt(a), BigInt(b), BigInt(t)) as { negative: boolean; compileMs: number; mpcMs: number; bytes: number; messages: number; andOps: number };
}

/** The secure argmax over shares the caller already holds: q_k = a_k + b_k mod 2^32. */
export async function secureArgmaxShares(a: Uint32Array, b: Uint32Array): Promise<{ arg: number; mpcMs: number; bytes: number; messages: number; andOps: number }> {
  const m = await getModule();
  return await m.secure_argmax(a, b) as { arg: number; mpcMs: number; bytes: number; messages: number; andOps: number };
}

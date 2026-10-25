// REAL in-browser 2-party MPC for the non-linear decision layer (argmax over the FHE logits), on
// Polytune (sine-fdn, MIT): WRK17b multi-party authenticated garbling, secure against malicious
// adversaries controlling all but one party. BOTH parties run inside this tab over an in-memory
// channel, so there is ZERO server / ZERO VPS - yet the protocol, its messages and its wall-clock are
// GENUINELY the real garbled-circuit computation (not a hand-rolled reconstruction, not a cost model).
// Each logit is additively secret-shared across the two parties (logit_k = a_k + b_k - BIG); the
// circuit privately reconstructs and compares the values and outputs ONLY the argmax index, so
// neither party's circuit input reveals a logit.
//
// Honest scope: in one browser the two "parties" necessarily share a JS heap and a thread, so the
// party separation is notional (it must be, with zero VPS) - what is genuinely real is the
// cryptographic protocol, the measured time and the real communication. The full deployment runs MPyC
// across separate hosts.
//
// REPLACED IN ROUND 65. This file ran mpc-framework + emp-wasm-engine + summon-ts until then: all three
// archived upstream, and mpc-framework pulled the zod release carrying GHSA-mrhx-c7xc-m3w5 with no fix
// coming. Polytune was prototyped (Round 63) and measured between two tabs over netem-shaped TCP
// (Round 64): on this 2-class argmax it was faster on every link measured, though it sends about five
// times the bytes. The module is built from wasm/qsh-mpc by the web image's Rust stage, where its
// `cargo test` runs on every build, or locally by scripts/build_wasm.sh.

export interface MPCResult {
  out: number[];        // [argmax index]
  compares: number;     // comparisons in the argmax (N-1)
  commBytes: number;    // REAL measured garbled-circuit communication, both directions
  mpcMs: number;        // REAL measured wall-clock of the protocol (module init cached via warmMPC)
  messages?: number;    // messages exchanged - over a network each one would cost a latency
}

// Fixed-point + additive-share parameters, unchanged from the emp-wasm version so results stay
// comparable. Logits from the small demo model are O(1); OFFSET makes the quantised value
// non-negative, BIG masks it (>> the value), and every intermediate stays < 2^17, well inside the
// circuit's 32-bit integers.
const SCALE = 64;            // fixed-point scale for the float logits
const OFFSET = 8;            // logits assumed within [-OFFSET, OFFSET] -> quantised value is positive
const QMAX = 2 * OFFSET * SCALE; // 1024, the max quantised value (logits clamped to it)
const BIG = 1 << 16;         // additive-share mask range (65536 >> QMAX, hides the value)

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
 * Genuine 2-party secure argmax over additively-shared logits - both parties run Polytune's
 * authenticated-garbling protocol inside this tab. Returns the argmax index plus the REAL measured
 * protocol time and communication.
 */
export async function secureArgmax(logits: number[], rnd: () => number): Promise<MPCResult> {
  const n = logits.length;
  if (n < 2) return { out: [0], compares: 0, commBytes: 0, mpcMs: 0, messages: 0 };
  const m = await getModule();

  // quantise + additively share each logit: q_k = clamp(round((logit_k+OFFSET)*SCALE), 0, QMAX);
  // a_k uniform in [0,BIG); b_k = q_k - a_k + BIG (>=0). The circuit computes a_k+b_k-BIG = q_k privately.
  const a = new Int32Array(n);
  const b = new Int32Array(n);
  for (let k = 0; k < n; k++) {
    const q = Math.min(QMAX, Math.max(0, Math.round((logits[k] + OFFSET) * SCALE)));
    a[k] = Math.floor(rnd() * BIG);
    b[k] = q - a[k] + BIG;
  }

  const r = await m.secure_argmax(a, b) as { arg: number; mpcMs: number; bytes: number; messages: number };
  return { out: [r.arg], compares: n - 1, commBytes: r.bytes, mpcMs: r.mpcMs, messages: r.messages };
}

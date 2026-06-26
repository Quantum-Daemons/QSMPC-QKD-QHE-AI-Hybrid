// REAL in-browser 2-party MPC for the non-linear decision layer (argmax over the FHE logits), via
// mpc-framework + emp-wasm-engine (both MIT) — authenticated-garbling secure two-party computation.
// BOTH parties run loopback inside this single browser tab over an in-memory transport, so there is
// ZERO server / ZERO VPS — yet the protocol, its on-wire messages and its wall-clock are GENUINELY the
// real garbled-circuit 2PC (not a hand-rolled reconstruction, not a cost model). Each logit is
// additively secret-shared across the two parties (logit_k = a_k + b_k − BIG); the garbled circuit
// privately reconstructs + compares the values and outputs ONLY the argmax index, so neither party's
// circuit input reveals a logit.
//
// Honest scope: in one browser the two "parties" necessarily share a JS heap, so the party separation
// is notional (it must be, with zero VPS) — what is genuinely real is the cryptographic protocol, the
// measured time and the real communication bytes. The full deployment runs MPyC across separate hosts.
// summon-ts compiles the argmax circuit at runtime; both libraries inline their WASM as base64, so they
// load cleanly under Vite (no separate-wasm fetch) and are dynamic-imported (kept out of the entry bundle).

export interface MPCResult {
  out: number[];        // [argmax index]
  compares: number;     // comparisons in the argmax (N-1)
  commBytes: number;    // REAL measured 2-party garbled-circuit communication
  mpcMs: number;        // REAL measured wall-clock of the 2PC protocol (compile cached via warmMPC)
}

// Fixed-point + additive-share parameters. Logits from the small demo model are O(1); OFFSET makes the
// quantised value non-negative, BIG masks it (>> the value), and every intermediate stays < 2^17 so the
// boolify WIDTH has ample headroom and no signed/overflow subtlety arises.
const SCALE = 64;            // fixed-point scale for the float logits
const OFFSET = 8;            // logits assumed within [-OFFSET, OFFSET] -> quantised value is positive
const QMAX = 2 * OFFSET * SCALE; // 1024, the max quantised value (logits clamped to it)
const BIG = 1 << 16;         // additive-share mask range (65536 >> QMAX, hides the value)
const WIDTH = 24;            // boolify bit-width (a_k+b_k = q_k+BIG < 2^17 << 2^24)

interface MpcLibs {
  Protocol: any;
  EmpWasmEngine: any;
  summon: any;
}

let _libs: Promise<MpcLibs> | null = null;
const _circuits = new Map<number, any>(); // compiled argmax circuit, cached by class count N

function getLibs(): Promise<MpcLibs> {
  return (_libs ??= (async () => {
    const [mpc, eng, summon] = await Promise.all([
      import("mpc-framework"),
      import("emp-wasm-engine"),
      import("summon-ts"),
    ]);
    await summon.init();
    return { Protocol: mpc.Protocol, EmpWasmEngine: eng.EmpWasmEngine, summon };
  })());
}

// Build the argmax-of-N circuit source (summon TS-like): alice supplies a_0..a_{N-1}, bob supplies
// b_0..b_{N-1}; q_k = a_k + b_k − BIG is the privately-reconstructed logit; output the index of the max.
function argmaxCircuitSrc(n: number): string {
  const aIn = Array.from({ length: n }, (_, k) => `  const a${k} = io.input('alice', 'a${k}', summon.number());`).join("\n");
  const bIn = Array.from({ length: n }, (_, k) => `  const b${k} = io.input('bob', 'b${k}', summon.number());`).join("\n");
  const q = Array.from({ length: n }, (_, k) => `  const q${k} = a${k} + b${k} - ${BIG};`).join("\n");
  let cmp = "  let arg = 0;\n  let best = q0;\n";
  for (let k = 1; k < n; k++) cmp += `  if (q${k} > best) { best = q${k}; arg = ${k}; }\n`;
  return `export default (io: Summon.IO) => {\n${aIn}\n${bIn}\n${q}\n${cmp}  io.outputPublic('arg', arg);\n}`;
}

async function getCircuit(n: number): Promise<any> {
  if (!_circuits.has(n)) {
    const { summon } = await getLibs();
    const { circuit } = summon.compile({
      path: "circuit/main.ts",
      boolifyWidth: WIDTH,
      files: { "circuit/main.ts": argmaxCircuitSrc(n) },
    });
    _circuits.set(n, circuit);
  }
  return _circuits.get(n);
}

/** Warm the WASM init + compile the common 2-class circuit so the first real inference is fast. */
export async function warmMPC(): Promise<void> {
  await getCircuit(2);
}

/**
 * Genuine 2-party secure argmax over additively-shared logits — both parties run the real
 * emp-wasm authenticated-garbling protocol loopback in this browser. Returns the argmax index plus the
 * REAL measured protocol time + communication bytes.
 */
export async function secureArgmax(logits: number[], rnd: () => number): Promise<MPCResult> {
  const n = logits.length;
  if (n < 2) return { out: [0], compares: 0, commBytes: 0, mpcMs: 0 };
  const { Protocol, EmpWasmEngine } = await getLibs();
  const circuit = await getCircuit(n);

  // quantise + additively share each logit: q_k = clamp(round((logit_k+OFFSET)*SCALE), 0, QMAX);
  // a_k uniform in [0,BIG); b_k = q_k - a_k + BIG (>=0). The circuit computes a_k+b_k-BIG = q_k privately.
  const aInputs: Record<string, number> = {};
  const bInputs: Record<string, number> = {};
  for (let k = 0; k < n; k++) {
    const q = Math.min(QMAX, Math.max(0, Math.round((logits[k] + OFFSET) * SCALE)));
    const a = Math.floor(rnd() * BIG);
    aInputs[`a${k}`] = a;
    bInputs[`b${k}`] = q - a + BIG;
  }

  const protocol = new Protocol(circuit, new EmpWasmEngine());
  const sessions: Record<string, any> = {};
  let commBytes = 0;
  const mkSend = (self: string) => (to: string, msg: Uint8Array) => {
    commBytes += msg.length;
    queueMicrotask(() => sessions[to].handleMessage(self, msg));
  };

  const t0 = performance.now();
  const alice = protocol.join("alice", aInputs, mkSend("alice"));
  const bob = protocol.join("bob", bInputs, mkSend("bob"));
  sessions.alice = alice;
  sessions.bob = bob;
  const [oa] = await Promise.all([alice.output(), bob.output()]);
  const mpcMs = performance.now() - t0;

  const arg = typeof oa?.arg === "number" ? oa.arg : 0;
  return { out: [arg], compares: n - 1, commBytes, mpcMs };
}

/**
 * Private set intersection among three to five banks, under Polytune, in this tab (Round 86).
 *
 * The finance page's PSI between two banks is RFC 9497's OPRF (`oprfPsi.ts`), which is two-party by construction. Among more banks this module uses the
 * general tool instead: one garbled circuit, every bank a party to Polytune's WRK17b protocol, whose only output is the accounts all of them hold. Each bank
 * hashes its accounts to 32 bits (an ID of 0 is reserved for "no match" and cannot occur after the hash's mapping below) and shuffles them, because the output
 * is positional in the first bank's list. The circuit compares every pair, so it stays small only for small lists: a run of eight accounts each
 * takes seconds, and dedicated protocols go much further (Kolesnikov et al., CCS 2017). Every bank runs in this tab, so the separation is notional, and that
 * is the only reason the page can compare the result with the intersection in the clear.
 */

import { randomWords } from "./csprng";
import type { MultiPartyPsiOutcome } from "./clientMPC";

export interface MultiPsiResult {
  parties: number;
  setSize: number;
  /** How many accounts the circuit reported as held by every bank. */
  intersection: number;
  /** A self-test, available only because every bank is in this tab: the same set computed in the clear. */
  matchesClear: boolean;
  clearIntersection: number;
  andGates: number;
  bytes: number;
  messages: number;
  compileMs: number;
  mpcMs: number;
  totalMs: number;
  /** log2 of the union bound on two different accounts sharing a 32-bit hash in this run's comparisons. */
  falseMatchBoundLog2: number;
}

const encoder = new TextEncoder();

async function hash32(id: string): Promise<number> {
  const digest = new DataView(await crypto.subtle.digest("SHA-256", encoder.encode(id)));
  return digest.getUint32(0, false) || 1;
}

/** A uniform shuffle from the CSPRNG (Fisher-Yates), with rejection so `% (i + 1)` carries no bias. */
function shuffled(a: Uint32Array): Uint32Array {
  const out = a.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const bound = i + 1;
    const limit = Math.floor(2 ** 32 / bound) * bound;
    let w = randomWords(1)[0];
    while (w >= limit) w = randomWords(1)[0];
    const j = w % bound;
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

export async function multiPartyPsi(lists: string[][]): Promise<MultiPsiResult> {
  const t0 = performance.now();
  const parties = lists.length, setSize = lists[0]?.length ?? 0;
  const hashed = await Promise.all(lists.map((l) => Promise.all(l.map(hash32))));
  const sets = hashed.map((h) => shuffled(Uint32Array.from(h)));
  const mpc: { secureMultiPartyPsi(s: Uint32Array[]): Promise<MultiPartyPsiOutcome> } = await import("./clientMPC");
  const o = await mpc.secureMultiPartyPsi(sets);

  const inAll = lists[0].filter((id) => lists.every((l) => l.includes(id)));
  const want = new Set(await Promise.all(inAll.map(hash32)));
  const got = o.ids.filter((x) => x !== 0);
  const matchesClear = got.length === want.size && got.every((x) => want.has(x));
  const comparisons = setSize * setSize * (parties - 1);
  return {
    parties, setSize, intersection: got.length, matchesClear, clearIntersection: want.size,
    andGates: o.andOps, bytes: o.bytes, messages: o.messages, compileMs: o.compileMs, mpcMs: o.mpcMs,
    totalMs: performance.now() - t0, falseMatchBoundLog2: Math.log2(comparisons) - 32,
  };
}

/** The same, off the page's thread: five banks are tens of seconds of garbling. */
export async function multiPartyPsiOffThread(lists: string[][]): Promise<MultiPsiResult & { ranIn: string }> {
  if (typeof window === "undefined" || typeof Worker === "undefined") {
    return { ...(await multiPartyPsi(lists)), ranIn: "the calling thread" };
  }
  const worker = new Worker(new URL("./clientMultiPsi.worker.ts", import.meta.url), { type: "module" });
  try {
    return await new Promise((resolve, reject) => {
      worker.onmessage = (e: MessageEvent<{ ok?: boolean; result?: MultiPsiResult; error?: string }>) => {
        if (e.data.ok && e.data.result) resolve({ ...e.data.result, ranIn: "a Web Worker" });
        else reject(new Error(`in its worker: ${e.data.error ?? "no result"}`));
      };
      worker.onerror = (e) => reject(new Error(`its worker failed: ${e.message || "the worker could not be loaded"}`));
      worker.onmessageerror = () => reject(new Error("its worker's result could not be read back"));
      worker.postMessage({ lists });
    });
  } finally {
    worker.terminate();
  }
}

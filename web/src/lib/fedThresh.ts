/**
 * The threshold tier's wasm (Lattigo v6.2.0, compiled from Go: wasm/qsh-fedthresh) as the pages load it. Kept apart from clientFedAgg so the lending page's worker can
 * load the module without the medical page's federation code.
 */

/** globalThis.qshFedThresh, as wasm/qsh-fedthresh/cmd/wasm installs it. */
export interface FedThreshApi {
  settings(): string;
  open(kind: string, parties: number, crs: Uint8Array): { id?: number; setting?: string; keyShareBytes?: number;
    keyGenMs?: number; error?: string };
  round(id: number, flat: Uint8Array, parties: number, n: number): { aggregate?: Uint8Array; stats?: string;
    error?: string };
  close(id: number): void;
  /** The lending page's committee (agg/committee.go): the members' keys and Carol's own key pair, in this tab. */
  committeeOpen(members: number, threshold: number, crs: Uint8Array, maxWeights: number): { id?: number; setting?: string;
    keyGenMs?: number; publicKeyBytes?: number; carolPublicKeyBytes?: number; carolKeyGenMs?: number; error?: string };
  /** The borrower's side: each balance (float64 integers, little-endian) in its own ciphertext, framed for the wire. */
  committeeEncrypt(id: number, balances: Uint8Array): { ciphertexts?: Uint8Array; encryptMs?: number; error?: string };
  /** The pool's side, once: the borrower's ciphertexts as it opened them, checked for shape. */
  committeeStore(id: number, ciphertexts: Uint8Array, count: number): { count?: number; error?: string };
  /** The pool evaluates and masks, the active members re-encrypt to Carol, and the pool combines their shares. */
  committeeCheck(id: number, weights: Uint8Array, settled: Uint8Array, active: Int32Array): { switched?: Uint8Array;
    poolShare?: number; evaluateMs?: number; evaluatedBytes?: number; reencrypt?: string; error?: string };
  /** Carol's side: slot 0 of what she received, M + r mod t, and whether every other slot is zero. */
  committeeDecrypt(id: number, switched: Uint8Array): { held?: number; restZero?: boolean; decryptMs?: number; error?: string };
  committeeClose(id: number): void;
}

/** Loads Go's wasm_exec.js and the module from `base` (a URL prefix ending in "/") into this realm. */
export async function loadFedThresh(base: string): Promise<FedThreshApi> {
  const g = globalThis as unknown as { Go?: new () => { importObject: WebAssembly.Imports; run(i: WebAssembly.Instance): void };
                                       qshFedThresh?: FedThreshApi };
  if (!g.qshFedThresh) {
    if (!g.Go) await import(/* @vite-ignore */ `${base}wasm_exec.js`);
    const go = new g.Go!();
    const res = await fetch(`${base}qsh_fedthresh.wasm`);
    if (!res.ok) throw new Error(`qsh_fedthresh.wasm: HTTP ${res.status}`);
    const { instance } = await WebAssembly.instantiate(await res.arrayBuffer(), go.importObject);
    void go.run(instance);   // installs globalThis.qshFedThresh, then waits on its exported functions
  }
  if (!g.qshFedThresh) throw new Error("qsh_fedthresh.wasm did not install qshFedThresh");
  return g.qshFedThresh;
}

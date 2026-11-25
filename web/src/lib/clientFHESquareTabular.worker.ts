/**
 * The finance square network under CKKS, off the page's thread (Round 86, F7b; the owner's choice of 2026-11-25 that it
 * replaces the linear student on the finance page).
 *
 * The same module as on the calling thread (`clientFHESquareTabular.ts`), with its own node-seal context per parameter
 * set. node-seal 7.0.0 ships one single-threaded wasm build, so a block of rows is one worker's work; a pool runs blocks
 * side by side. Raw rows arrive by structured clone and are scaled and clipped here, as the client would before it
 * encrypts them; the decrypted and plaintext logits return the same way.
 */

import { runSquareTabular, scaleRow, type SquareTabRun, type SquareTabWeights } from "./clientFHESquareTabular";

export interface SquareTabWorkerRequest { id: number; rows: number[][]; weights: SquareTabWeights; set: string }

// The project compiles against the DOM lib, where `self` is a Window; a worker's own scope needs only these two.
const scope = self as unknown as {
  onmessage: ((e: MessageEvent<SquareTabWorkerRequest>) => void) | null;
  postMessage(message: unknown): void;
};

scope.onmessage = async (e) => {
  const { id, rows, weights, set } = e.data;
  try {
    const z = new Float64Array(rows.length * weights.n_in);
    for (let i = 0; i < rows.length; i++) z.set(scaleRow(weights, rows[i]), i * weights.n_in);
    const t0 = performance.now();
    const run: SquareTabRun = await runSquareTabular(weights, z, rows.length, set, () => {});
    scope.postMessage({ id, ok: true, run, totalMs: performance.now() - t0 });
  } catch (err) {
    scope.postMessage({ id, ok: false, error: (err as Error)?.message ?? String(err) });
  }
};

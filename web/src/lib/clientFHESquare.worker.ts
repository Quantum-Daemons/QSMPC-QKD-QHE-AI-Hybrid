/**
 * The fully encrypted medical student, off the page's thread (Round 86, 2026-11-19).
 *
 * The same module as on the calling thread (`clientFHESquare.ts`): its own node-seal context per parameter set, so
 * `clientFHE.ts`, which the poster cites, is not involved. node-seal 7.0.0 ships one single-threaded wasm build, so this
 * is one worker beside the page. The images and weights arrive by structured clone, a progress message follows each
 * image, and the runs return the same way; no SharedArrayBuffer, so no cross-origin isolation is needed.
 */

import { runSquareStudent, type SquareRun, type SquareWeights } from "./clientFHESquare";

export interface SquareWorkerRequest { id: number; rows: number[][]; weights: SquareWeights; set: string }

// The project compiles against the DOM lib, where `self` is a Window; a worker's own scope needs only these two.
const scope = self as unknown as {
  onmessage: ((e: MessageEvent<SquareWorkerRequest>) => void) | null;
  postMessage(message: unknown): void;
};

scope.onmessage = async (e) => {
  const { id, rows, weights, set } = e.data;
  try {
    const runs: SquareRun[] = [];
    for (const x of rows) {
      runs.push(await runSquareStudent(x, weights, set));
      scope.postMessage({ id, done: runs.length });
    }
    scope.postMessage({ id, ok: true, runs });
  } catch (err) {
    scope.postMessage({ id, ok: false, error: (err as Error)?.message ?? String(err) });
  }
};

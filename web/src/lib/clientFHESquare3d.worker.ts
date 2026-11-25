/**
 * The 3D nodule student's encrypted pass, off the page's thread (batch 4).
 *
 * The same module as on the calling thread (`clientFHESquare3d.ts`): its own node-seal context per parameter set, so
 * `clientFHE.ts` is not involved. node-seal 7.0.0 ships one single-threaded wasm build, so this is one worker beside
 * the page. The crop and weights arrive by structured clone and the run returns the same way; no SharedArrayBuffer,
 * so no cross-origin isolation is needed.
 */

import { runSquare3d, type Square3dWeights } from "./clientFHESquare3d";

export interface Square3dWorkerRequest { id: number; crop: number[]; weights: Square3dWeights; set: string }

// The project compiles against the DOM lib, where `self` is a Window; a worker's own scope needs only these two.
const scope = self as unknown as {
  onmessage: ((e: MessageEvent<Square3dWorkerRequest>) => void) | null;
  postMessage(message: unknown): void;
};

scope.onmessage = async (e) => {
  const { id, crop, weights, set } = e.data;
  try {
    scope.postMessage({ id, ok: true, run: await runSquare3d(crop, weights, set) });
  } catch (err) {
    scope.postMessage({ id, ok: false, error: (err as Error)?.message ?? String(err) });
  }
};

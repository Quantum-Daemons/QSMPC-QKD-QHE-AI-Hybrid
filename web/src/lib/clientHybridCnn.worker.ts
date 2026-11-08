/**
 * The medical CNN student's CKKS + garbled-circuit run, off the page's thread (Round 86).
 *
 * The run is tens of seconds of CKKS and Polytune, and on the page's own thread input would lag for all of it, as the satellite student's did. The same
 * function runs here, in a dedicated worker. The weights and the image arrive by structured clone, the progress and the result go back the same way, and no
 * SharedArrayBuffer is used.
 */

import { hybridStudentInference } from "./clientHybridCnn";
import type { CnnStudentWeights } from "./clientFHE";

// The project compiles against the DOM lib, where `self` is a Window; a worker's own scope needs only these two.
const scope = self as unknown as {
  onmessage: ((e: MessageEvent<{ weights: CnnStudentWeights; image: number[][] }>) => void) | null;
  postMessage(message: unknown): void;
};

scope.onmessage = async (e) => {
  try {
    const result = await hybridStudentInference(e.data.weights, e.data.image, { onProgress: (progress) => scope.postMessage({ progress }) });
    scope.postMessage({ ok: true, result });
  } catch (err) {
    scope.postMessage({ ok: false, error: (err as Error)?.message ?? String(err) });
  }
};

/** A published tabular MLP's CKKS + garbled-circuit run, off the page's thread, as `clientHybridCnn.worker.ts` does for the CNN (Round 86). */

import { hybridMlpInference, type MlpWeights } from "./clientHybridMlp";

// The project compiles against the DOM lib, where `self` is a Window; a worker's own scope needs only these two.
const scope = self as unknown as {
  onmessage: ((e: MessageEvent<{ weights: MlpWeights; row: number[] }>) => void) | null;
  postMessage(message: unknown): void;
};

scope.onmessage = async (e) => {
  try {
    const result = await hybridMlpInference(e.data.weights, e.data.row, { onProgress: (progress) => scope.postMessage({ progress }) });
    scope.postMessage({ ok: true, result });
  } catch (err) {
    scope.postMessage({ ok: false, error: (err as Error)?.message ?? String(err) });
  }
};

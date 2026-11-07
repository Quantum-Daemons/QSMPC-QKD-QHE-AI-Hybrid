/**
 * The satellite student's evaluation, off the page's thread (Round 86).
 *
 * Under CKKS the student's 8,640 forecasts take about 21 s in a tab: 135 ciphertexts of 64 rows, each one encrypted,
 * evaluated and decrypted by node-seal. On the page's own thread that was 138 tasks of about 160 ms, and input lagged
 * for the whole run. Here the same module runs in a dedicated worker, so the page stays responsive.
 *
 * Nothing about the computation changes. This imports `evaluateForecastStudent` itself, so `clientFHE.ts`, which
 * the poster cites, is the same code either way. node-seal 7.0.0 ships one single-threaded wasm build, so this is one
 * worker beside the page, not a parallel CKKS. The sample and weights arrive by structured clone, and the result
 * returns the same way. No SharedArrayBuffer is used, so the worker needs no cross-origin isolation.
 */

import { evaluateForecastStudent, type ForecastEventsFile, type ForecasterWeights } from "./forecastStudent";

export interface StudentWorkerRequest {
  id: number; file: ForecastEventsFile; weights: ForecasterWeights; ckks: "none" | "spot" | "all";
}

// The project compiles against the DOM lib, where `self` is a Window; a worker's own scope needs only these two.
const scope = self as unknown as {
  onmessage: ((e: MessageEvent<StudentWorkerRequest>) => void) | null;
  postMessage(message: unknown): void;
};

scope.onmessage = async (e) => {
  const { id, file, weights, ckks } = e.data;
  try {
    const ev = await evaluateForecastStudent(file, weights, { ckks });
    scope.postMessage({ id, ok: true, ev });
  } catch (err) {
    scope.postMessage({ id, ok: false, error: (err as Error)?.message ?? String(err) });
  }
};

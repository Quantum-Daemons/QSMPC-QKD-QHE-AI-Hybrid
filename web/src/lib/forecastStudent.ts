/**
 * The satellite CKKS student, scored end to end in this tab (Round 86): `forecastEvents.ts` reproduces the recorded
 * forecast, flags and event scores without encryption, and this module adds the same dot products under CKKS
 * (`encryptedLinearBatch`, unchanged) and thresholds the decrypted forecast, so the page shows what encryption does
 * to the events, not only to the numbers.
 */

import { encryptedLinearBatch } from "./clientFHE";
import {
  featureRows, flagsDigest, type ForecastEventsFile, type ForecasterWeights, plaintextForecast, scoreStretch,
  type StretchScore,
} from "./forecastEvents";

export type { ForecastEventsFile, ForecasterWeights, StretchScore } from "./forecastEvents";

export interface StudentEvaluation {
  /** Plaintext forecast, thresholding and scoring, against what the generator recorded. */
  mae: number;
  persistenceMae: number;
  plain: StretchScore;
  flagsDigest: string;
  flagsDigestMatches: boolean;
  scoreMatches: boolean;
  /** Under CKKS, when asked: the decrypted forecast against the plaintext one, and the events it gives. */
  ckks: null | {
    rows: number; ciphertexts: number; fheMs: number; maxAbsError: number;
    flagsChanged: number; encrypted: StretchScore;
  };
  /** Where it ran: the page's own runs use a dedicated Web Worker; node, which has none, the calling thread. */
  ranIn?: "a Web Worker" | "the calling thread";
}

/** Evaluate the student on its published stretch; `ckks` "all" also runs every forecast under CKKS. */
export async function evaluateForecastStudent(file: ForecastEventsFile, w: ForecasterWeights,
                                              opts: { ckks?: "none" | "spot" | "all" } = {}): Promise<StudentEvaluation> {
  if (w.kind !== "linear-forecaster") throw new Error(`the weights are a ${w.kind}, not a linear forecaster`);
  const x = featureRows(file);
  const forecast = plaintextForecast(w, x);
  const k = file.stretch.history_rows;
  let err = 0, pers = 0, count = 0;
  for (let t = 0; t < forecast.length; t++) {
    for (let c = 0; c < w.b.length; c++) {
      err += Math.abs(forecast[t][c] - file.rows[t + k][c]);
      pers += Math.abs(file.rows[t + k - 1][c] - file.rows[t + k][c]);
      count++;
    }
  }
  const plain = scoreStretch(file, w, forecast);
  const digest = await flagsDigest(plain.flags);
  const exp = file.expected;
  const result: StudentEvaluation = {
    mae: err / count, persistenceMae: pers / count, plain, flagsDigest: digest,
    flagsDigestMatches: digest === exp.flags_sha256,
    scoreMatches: Math.abs(plain.anomalies.f05 - exp.anomalies.f_0_5) <= 1e-9
      && plain.anomalies.tp === exp.anomalies.tp && plain.anomalies.fp === exp.anomalies.fp
      && plain.anomalies.fn === exp.anomalies.fn,
    ckks: null,
  };
  const mode = opts.ckks ?? "none";
  if (mode === "none") return result;
  // 48 features take a 64-slot block, so a ciphertext of this ring carries 64 rows; "spot" encrypts one of them.
  const rows = mode === "all" ? x : x.slice(0, 64);
  const enc = await encryptedLinearBatch(w.W, w.b, rows, { maxCiphertexts: Math.ceil(rows.length / 64) + 1 });
  if (enc.rows !== rows.length) {
    throw new Error(`CKKS scored ${enc.rows} of the ${rows.length} rows offered; the cap would hide the rest`);
  }
  let maxAbs = 0;
  for (let r = 0; r < enc.rows; r++) {
    for (let c = 0; c < w.b.length; c++) maxAbs = Math.max(maxAbs, Math.abs(enc.logits[r][c] - forecast[r][c]));
  }
  // The events of the decrypted forecast, over the whole stretch only when every row was encrypted.
  const decrypted = mode === "all" ? enc.logits : null;
  const encrypted = decrypted ? scoreStretch(file, w, decrypted) : plain;
  let changed = 0;
  if (decrypted) for (let i = 0; i < plain.flags.length; i++) if (plain.flags[i] !== encrypted.flags[i]) changed++;
  result.ckks = { rows: enc.rows, ciphertexts: enc.ciphertexts, fheMs: enc.fheMs, maxAbsError: maxAbs,
                  flagsChanged: changed, encrypted };
  return result;
}

/**
 * The same evaluation, off the page's thread. Under CKKS every row takes about 21 s in a tab, as 138 tasks of about
 * 160 ms that left input lagging for the whole run, so in a browser this always runs `evaluateForecastStudent` in a
 * dedicated worker (`forecastStudent.worker.ts`). A worker that fails is reported as the failure it is and is never
 * retried on the page's thread, which would hide it behind 21 s of blocked input. Outside a browser (node, the
 * tests) there is no Worker, and it runs on the calling thread. The result says which.
 */
export async function evaluateForecastStudentOffThread(file: ForecastEventsFile, w: ForecasterWeights,
                                                       opts: { ckks?: "none" | "spot" | "all" } = {}): Promise<StudentEvaluation> {
  if (typeof window === "undefined" || typeof Worker === "undefined") {
    return { ...(await evaluateForecastStudent(file, w, opts)), ranIn: "the calling thread" };
  }
  // Vite bundles the worker from exactly this form: `new URL(...)` written inside `new Worker(...)`.
  const worker = new Worker(new URL("./forecastStudent.worker.ts", import.meta.url), { type: "module" });
  try {
    return await new Promise<StudentEvaluation>((resolve, reject) => {
      worker.onmessage = (e: MessageEvent<{ id: number; ok: boolean; ev?: StudentEvaluation; error?: string }>) => {
        const d = e.data;
        if (d.ok && d.ev) resolve({ ...d.ev, ranIn: "a Web Worker" });
        else reject(new Error(`in its worker: ${d.error ?? "no result"}`));
      };
      worker.onerror = (e) => reject(new Error(`its worker failed: ${e.message || "the worker could not be loaded"}`));
      worker.onmessageerror = () => reject(new Error("its worker's result could not be read back"));
      worker.postMessage({ id: 1, file, weights: w, ckks: opts.ckks ?? "none" });
    });
  } finally {
    worker.terminate();
  }
}

/** The published weights, beside the evaluation sample at the same pinned revision. */
export async function fetchForecasterWeights(modelUrl: string): Promise<ForecasterWeights> {
  if (!/\/onnx\/[^/]+$/.test(modelUrl)) throw new Error(`cannot derive weights.json from "${modelUrl}"`);
  const res = await fetch(modelUrl.replace(/\/onnx\/[^/]+$/, "/weights.json"), { mode: "cors" });
  if (!res.ok) throw new Error(`the Hugging Face Hub answered HTTP ${res.status} for weights.json`);
  return (await res.json()) as ForecasterWeights;
}

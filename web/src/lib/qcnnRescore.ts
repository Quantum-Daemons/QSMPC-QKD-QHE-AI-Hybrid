// The trained QCNN re-scored in the tab, over the rows of the trainer's own test split (Round 86).
//
// Until this, every QCNN figure a page printed - accuracy, macro F1, minority-class F1 - came from the weights file, labelled
// "recorded at training time, not re-scored in this tab". `scripts/data/build_qcnn_heldout.py` ships the test rows in the units
// `qcnnInferTrained` takes and writes them only after the trainer's own circuit reproduces every recorded metric from them, so a
// re-score here that differs from the record is a difference between this tab's circuit and the trainer's, not between two splits.
//
// The metrics follow scikit-learn's `f1_score` with `zero_division=0`, which is what the trainer calls: macro F1 averages over the
// labels present in the truth or the predictions, and a binary model's F1 is class 1's.

import { qcnnInferTrained, type QcnnWeights } from "./quantumSim";

export interface QcnnHeldOut {
  usecase: string;
  feature_names: string[];
  input_space: string;
  class_names?: string[] | null;
  split: string;
  rows: number[][];
  labels: number[];
  licence: string;
  licence_url?: string;
  attribution?: string;
}

export interface QcnnRescore {
  rows: number;
  /** The metric the artifact headlines, its value here and the value its weights file records. */
  label: string;
  headline: number;
  recorded: number | null;
  /** Equal to six decimals, the precision the trainer records. */
  matchesRecord: boolean;
  accuracy: number;
  macroF1: number | null;
  minorityF1: number | null;
  ms: number;
  split: string;
}

/** F1 of one class, scikit-learn's convention: 0 when the class is neither predicted nor present. */
function f1Of(y: number[], p: number[], c: number): number {
  let tp = 0, fp = 0, fn = 0;
  for (let i = 0; i < y.length; i++) {
    if (p[i] === c && y[i] === c) tp++;
    else if (p[i] === c) fp++;
    else if (y[i] === c) fn++;
  }
  return tp === 0 ? 0 : (2 * tp) / (2 * tp + fp + fn);
}

/** `f1_score(average="macro", zero_division=0)`: the mean over the labels present in the truth or the predictions. */
export function macroF1(y: number[], p: number[]): number {
  const labels = [...new Set([...y, ...p])].sort((a, b) => a - b);
  return labels.reduce((s, c) => s + f1Of(y, p, c), 0) / labels.length;
}

export function minorityF1(y: number[], p: number[]): number {
  return f1Of(y, p, 1);
}

const round6 = (v: number) => Math.round(v * 1e6) / 1e6;

export function rescoreQcnn(w: QcnnWeights, held: QcnnHeldOut): QcnnRescore {
  if (held.rows.length !== held.labels.length || held.rows.length === 0) {
    throw new Error(`QCNN held-out file has ${held.rows.length} rows and ${held.labels.length} labels`);
  }
  if (held.input_space !== w.input_space) {
    throw new Error(`QCNN held-out rows are in "${held.input_space}" units and the circuit takes "${w.input_space}"`);
  }
  const names = w.feature_names ?? [];
  if (names.length && held.feature_names.some((n, i) => n !== names[i])) {
    throw new Error(`QCNN held-out columns [${held.feature_names}] are not the circuit's [${names}]`);
  }
  const t0 = performance.now();
  const pred = held.rows.map((r) => qcnnInferTrained(r, w).cls);
  const ms = performance.now() - t0;
  const y = held.labels;
  const accuracy = pred.reduce((s, v, i) => s + (v === y[i] ? 1 : 0), 0) / y.length;
  const multiclass = Boolean(w.readout?.readout_qubits);
  const macro = multiclass ? macroF1(y, pred) : null, minority = multiclass ? null : minorityF1(y, pred);
  const head = w.headline_metric;
  const [label, headline, recorded] = head === "macro_f1" ? ["macro F1", macro ?? 0, w.test_macro_f1 ?? null]
    : head === "minority_class_f1" ? ["minority-class F1", minority ?? 0, w.test_minority_f1 ?? null]
      : ["accuracy", accuracy, w.test_accuracy ?? null];
  return {
    rows: y.length, label, headline, recorded, matchesRecord: recorded != null && round6(headline) === round6(recorded),
    accuracy, macroF1: macro, minorityF1: minority, ms, split: held.split,
  };
}

const _heldCache: Record<string, QcnnHeldOut | null> = {};

/** The served rows for a use case's data directory, or null. Content-type is the test, as for the weights: the host answers a missing
 *  file with the SPA's index.html and HTTP 200. */
export async function loadQcnnHeldOut(dataDir: string): Promise<QcnnHeldOut | null> {
  if (dataDir in _heldCache) return _heldCache[dataDir];
  try {
    const r = await fetch(`${import.meta.env.BASE_URL}data/${dataDir}/qcnn_heldout.json`);
    const json = (r.headers.get("content-type") ?? "").includes("application/json");
    _heldCache[dataDir] = r.ok && json ? ((await r.json()) as QcnnHeldOut) : null;
  } catch {
    _heldCache[dataDir] = null;
  }
  return _heldCache[dataDir];
}

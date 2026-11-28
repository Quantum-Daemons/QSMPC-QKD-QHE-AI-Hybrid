/**
 * A tabular square network's arithmetic without node-seal (Round 86, F7b; k outputs since batch 12): its file's layout,
 * the client's scaling and clip, and the network in the clear. Kept apart from `clientFHESquareTabular.ts`, which imports
 * the SEAL wasm, so that `realData.ts` and node harnesses that bundle it alone need no wasm. The finance network has two
 * outputs; the module takes any number from two, one a class in `classes`' order.
 */

export interface SquareTabParams { N: number; coeff_mod_bits: number[]; scale_bits: number }

/** `weights.json` of a tabular square network, as scripts/train/finance_square_student.py writes the finance one. */
export interface SquareTabWeights {
  kind: "dense-bn-square-dense";
  n_in: number; width: number;
  /** The fit window's standardisation and the clip, applied by the client before encryption. */
  centre: number[]; scale: number[]; clip: number;
  /** width x n_in and width, batch norm folded. */
  w1: number[][]; b1: number[];
  /** k x width and k, one row a class in `classes`' order (finance: two, its validation threshold folded into b2). */
  w2: number[][]; b2: number[];
  classes: string[];
  /** Each parameter set that met the pre-registered fidelity criterion, by name. */
  ckks: Record<string, SquareTabParams>;
}

/** One raw row, standardised by the fit window's moments and clipped, as the client encrypts it. */
export function scaleRow(w: SquareTabWeights, x: ArrayLike<number>): Float64Array {
  const z = new Float64Array(w.n_in);
  for (let j = 0; j < w.n_in; j++) {
    const v = (Number(x[j]) - w.centre[j]) / (w.scale[j] || 1);
    z[j] = Math.max(-w.clip, Math.min(w.clip, v));
  }
  return z;
}

/** The network in the clear, float64, on one scaled row starting at `at` in a flat array: one logit a class. */
export function plainLogits(w: SquareTabWeights, z: ArrayLike<number>, at = 0): number[] {
  const nOut = w.b2.length;
  const out = w.b2.slice(0, nOut);
  for (let k = 0; k < w.width; k++) {
    let h = w.b1[k];
    const row = w.w1[k];
    for (let j = 0; j < w.n_in; j++) h += row[j] * z[at + j];
    const s = h * h;
    for (let c = 0; c < nOut; c++) out[c] += w.w2[c][k] * s;
  }
  return out;
}

/** The decision: the first class with the largest logit, so two outputs decide class 1 only when its logit is larger. */
export function argmaxAt(logits: ArrayLike<number>, at: number, nOut: number): number {
  let best = 0;
  for (let c = 1; c < nOut; c++) if (logits[at + c] > logits[at + best]) best = c;
  return best;
}


/**
 * The finance square network's arithmetic without node-seal (Round 86, F7b): its file's layout, the client's scaling
 * and clip, and the network in the clear. Kept apart from `clientFHESquareTabular.ts`, which imports the SEAL wasm, so
 * that `realData.ts` and node harnesses that bundle it alone need no wasm.
 */

export interface SquareTabParams { N: number; coeff_mod_bits: number[]; scale_bits: number }

/** `weights.json` of the finance square student, as scripts/train/finance_square_student.py writes it. */
export interface SquareTabWeights {
  kind: "dense-bn-square-dense";
  n_in: number; width: number;
  /** The fit window's standardisation and the clip, applied by the client before encryption. */
  centre: number[]; scale: number[]; clip: number;
  /** width x n_in and width, batch norm folded. */
  w1: number[][]; b1: number[];
  /** 2 x width and 2, the validation threshold folded into b2. */
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

/** The network in the clear, float64, on one scaled row starting at `at` in a flat array. */
export function plainLogits(w: SquareTabWeights, z: ArrayLike<number>, at = 0): [number, number] {
  const out: [number, number] = [w.b2[0], w.b2[1]];
  for (let k = 0; k < w.width; k++) {
    let h = w.b1[k];
    const row = w.w1[k];
    for (let j = 0; j < w.n_in; j++) h += row[j] * z[at + j];
    const s = h * h;
    out[0] += w.w2[0][k] * s;
    out[1] += w.w2[1][k] * s;
  }
  return out;
}


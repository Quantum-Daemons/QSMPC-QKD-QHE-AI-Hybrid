/**
 * Split inference on a chest X-ray (Round 86, the owner's fourth choice of 2026-11-14): the pure part.
 *
 * The published network up to its 512 pooled features (`onnx/features_int8.onnx`) runs in the visitor's tab on one of the
 * model's 16 published films; the network's own last layer, a 512-weight linear map to the effusion logit z
 * (`/data/medical/cxr_head.json`, written and checked by scripts/export_cxr_head.py), is evaluated on the encrypted
 * features under CKKS; the decision is the model's operating point, p = sigmoid(z) at or above the published threshold.
 * These functions hold the arithmetic the card and its test share.
 */

export interface CxrHead {
  model: string; hf_repo: string; hf_revision: string; features_graph: string; features: number;
  weights: number[]; bias: number; decision_threshold: number; operating_point_logit: number;
  licence: string; licence_url: string; attribution: string;
  check: { films: number; max_abs_logit_difference_float32: number; decisions_agreeing_int8: number };
}

export const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));

/** z = w . features + b, in float64. */
export function headLogit(head: CxrHead, features: ArrayLike<number>): number {
  if (features.length !== head.weights.length) {
    throw new Error(`the head reads ${head.weights.length} features and was given ${features.length}`);
  }
  let z = head.bias;
  for (let i = 0; i < head.weights.length; i++) z += head.weights[i] * features[i];
  return z;
}

/** The model's decision at its published operating point. */
export function decide(head: CxrHead, z: number): { p: number; effusion: boolean } {
  const p = sigmoid(z);
  return { p, effusion: p >= head.decision_threshold };
}

/** A film's 224 x 224 uint8 pixels as the graph's input: one grey channel scaled to [0, 1] (the graph normalises). */
export function filmInput(pixels: readonly number[]): Float32Array {
  const out = new Float32Array(pixels.length);
  for (let i = 0; i < pixels.length; i++) out[i] = pixels[i] / 255;
  return out;
}

/** The same pixels as RGBA, to draw on a canvas. */
export function filmRgba(pixels: readonly number[]): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(new ArrayBuffer(pixels.length * 4));
  for (let i = 0; i < pixels.length; i++) {
    const v = pixels[i];
    out[4 * i] = v; out[4 * i + 1] = v; out[4 * i + 2] = v; out[4 * i + 3] = 255;
  }
  return out;
}

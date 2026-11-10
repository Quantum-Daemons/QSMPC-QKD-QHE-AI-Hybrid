// What the published CNN student responds to, in the clear (Round 86): occlusion sensitivity on the held-out image the page has just run under MPC.
//
// The forward pass is the plaintext reference of `encryptedStudentInference` in `clientFHE.ts`, in the order the weights file states. Occlusion (Zeiler and Fergus, ECCV 2014) needs no gradients and no extra model: hide a patch of the image, re-score, and see how far the predicted class's
// margin falls. On a 28x28 image and this network that is a few dozen forward passes.

import type { CnnStudentWeights } from "./clientFHE";

const relu = (v: number) => (v > 0 ? v : 0);

/** The student's logits on one image, in the clear. */
export function cnnLogits(w: CnnStudentWeights, image: number[][]): number[] {
  const p = w.conv.padding, k = w.conv.kernel, H = image.length + 2 * p, W = image[0].length + 2 * p;
  const pad = Array.from({ length: H }, (_, y) => Array.from({ length: W }, (_, x) => (y >= p && y < image.length + p && x >= p && x < image[0].length + p ? image[y - p][x - p] : 0)));
  const outH = H - k + 1, outW = W - k + 1;
  const biases = w.conv.bias ?? new Array(w.conv.out_ch).fill(0);
  const act = (v: number, kind: string) => (kind === "relu" ? relu(v) : v);
  const maps = w.conv.weight.map((oc, c) => Array.from({ length: outH }, (_, y) => Array.from({ length: outW }, (_, x) => {
    let a = biases[c];
    for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) a += pad[y + ky][x + kx] * oc[0][ky][kx];
    return act(a, w.conv.activation);
  })));
  const s = w.pool.stride, ph = Math.floor(outH / s), pw = Math.floor(outW / s);
  let v: number[] = [];
  for (const m of maps) {
    for (let y = 0; y < ph; y++) {
      for (let x = 0; x < pw; x++) {
        let a = 0;
        for (let dy = 0; dy < s; dy++) for (let dx = 0; dx < s; dx++) a += m[y * s + dy][x * s + dx];
        v.push(a / (s * s));
      }
    }
  }
  for (const d of w.dense) v = d.weight.map((row, j) => act(row.reduce((acc, wt, i) => acc + wt * v[i], 0) + (d.bias?.[j] ?? 0), d.activation));
  return v;
}

export interface OcclusionMap {
  /** The image as the model reads it. */
  image: number[][];
  /** Per pixel: how far the predicted class's margin falls when a patch covering it is hidden, averaged over the patches that cover it. */
  map: number[][];
  predicted: number;
  logits: number[];
  patch: number;
}

const margin = (l: number[], c: number) => l[c] - Math.max(...l.filter((_, i) => i !== c));

export function occlusionMap(w: CnnStudentWeights, image: number[][], patch = 6, stride = 2): OcclusionMap {
  const logits = cnnLogits(w, image);
  const predicted = logits.indexOf(Math.max(...logits));
  const base = margin(logits, predicted);
  const H = image.length, W = image[0].length;
  const fill = image.flat().reduce((a, b) => a + b, 0) / (H * W);
  const sum = Array.from({ length: H }, () => new Array(W).fill(0)), count = Array.from({ length: H }, () => new Array(W).fill(0));
  for (let y = 0; y + patch <= H; y += stride) {
    for (let x = 0; x + patch <= W; x += stride) {
      const hidden = image.map((row, ry) => row.map((v, rx) => (ry >= y && ry < y + patch && rx >= x && rx < x + patch ? fill : v)));
      const drop = base - margin(cnnLogits(w, hidden), predicted);
      for (let ry = y; ry < y + patch; ry++) for (let rx = x; rx < x + patch; rx++) { sum[ry][rx] += drop; count[ry][rx]++; }
    }
  }
  return { image, map: sum.map((row, y) => row.map((v, x) => (count[y][x] ? v / count[y][x] : 0))), predicted, logits, patch };
}

/**
 * A 3D chest-CT nodule student end to end under CKKS (batch 4; the owner's choices of 2026-11-23/24: 28 x 28 x 28
 * encrypted, LIDC-IDRI, a canvas viewer with an exact contribution map).
 *
 * The student is the 2D square student's shape in three dimensions: a 3 x 3 x 3 convolution from one channel to four
 * (padding 1, batch norm folded), a square, a 2 x 2 x 2 average pool and a dense layer to two logits: multiplicative
 * depth 3, so it runs in its own context, as `clientFHESquare.ts` does.
 *
 * LAYOUT. A crop is 28^3 = 21,952 voxels. Repeating each convolution tap for the four channels, as the 2D student does,
 * would take 27 x 22 ciphertexts at N = 8192; here each tap is split into six blocks of N/2 slots instead (162
 * ciphertexts), and the evaluator forms each channel by multiplying the tap blocks by that channel's kernel tap, a
 * plaintext scalar, and summing: 4 channels x 6 blocks. After the square, each class's dense weights, averaged over each
 * pooling window and laid out in the same blocks, multiply the 24 squared blocks, which are summed into one ciphertext,
 * and a rotate-and-add over all N/2 slots puts the logit in every slot: the client decrypts no partial sum. Slots past
 * the 21,952nd voxel of the last block are zero in every input and weight.
 *
 * WHAT THE DATA OWNER SEES IN THE CLEAR. The contribution map: each convolution output's share of the logit difference,
 * sum over channels of (dense weight of class 1 minus class 0, averaged over its pooling window) times the squared
 * convolution output. Over all 21,952 positions it sums exactly to the logit difference minus the difference of the dense
 * biases, which `contributionSum` checks; it needs the model's weights and the volume, so the owner computes it, not the
 * evaluator.
 */
// The SEAL wasm by its hashed asset URL, exactly as clientFHE.ts loads it.
import sealWasmUrl from "/node_modules/node-seal/dist/seal_throws.wasm?url";

export const SIDE3 = 28;
export const VOXELS = SIDE3 * SIDE3 * SIDE3;          // 21,952
export const CHANNELS3 = 4;
const POOLED = SIDE3 / 2;                               // 14
const POOLED_VOXELS = POOLED * POOLED * POOLED;         // 2,744

/** `weights.json` of the 3D square student, as its trainer writes it. */
export interface Square3dWeights {
  kind: "cnn3d-1conv-square-1dense";
  conv: { in_ch: 1; out_ch: number; kernel: 3; padding: 1; activation: "square"; batch_norm: "folded";
          /** (out_ch, 1, 3, 3, 3) */
          weight: number[][][][][]; bias: number[] };
  pool: { type: "avg"; stride: 2 };
  dense: { in: number; out: 2; weight: number[][]; bias: number[]; activation: "none" }[];
  classes: string[];
  /** Each parameter set that passed the pre-registered fidelity criterion, by name. */
  ckks: Record<string, { N: number; coeff_mod_bits: number[]; scale_bits: number }>;
}

export interface Square3dRun {
  set: string;
  logits: [number, number];
  plain: [number, number];
  maxAbsError: number;
  decisionAgrees: boolean;
  predicted: number;
  ciphertexts: { input: number; output: number };
  bytes: { input: number; output: number };
  ms: { keys: number; encrypt: number; evaluate: number; decrypt: number };
}

function check(w: Square3dWeights): void {
  if (w.kind !== "cnn3d-1conv-square-1dense" || w.conv.out_ch !== CHANNELS3 || w.conv.kernel !== 3 || w.conv.padding !== 1
      || w.pool.stride !== 2 || w.dense.length !== 1 || w.dense[0].in !== CHANNELS3 * POOLED_VOXELS || w.dense[0].out !== 2) {
    throw new Error("these weights are not the 3D square student's layout");
  }
}

const idx = (z: number, y: number, x: number) => (z * SIDE3 + y) * SIDE3 + x;

/** The 27 taps of a 3 x 3 x 3 convolution with padding 1, as Conv3d correlates: tap (dz, dy, dx) is the zero-padded
 *  volume shifted by (dz - 1, dy - 1, dx - 1), over the 28^3 output in z, y, x order. */
export function taps3d(v: ArrayLike<number>): Float64Array[] {
  if (v.length !== VOXELS) throw new Error(`the 3D student reads ${VOXELS} voxels, not ${v.length}`);
  const taps: Float64Array[] = [];
  for (let dz = 0; dz < 3; dz++) for (let dy = 0; dy < 3; dy++) for (let dx = 0; dx < 3; dx++) {
    const t = new Float64Array(VOXELS);
    for (let z = 0; z < SIDE3; z++) {
      const zz = z + dz - 1;
      if (zz < 0 || zz >= SIDE3) continue;
      for (let y = 0; y < SIDE3; y++) {
        const yy = y + dy - 1;
        if (yy < 0 || yy >= SIDE3) continue;
        for (let x = 0; x < SIDE3; x++) {
          const xx = x + dx - 1;
          if (xx >= 0 && xx < SIDE3) t[idx(z, y, x)] = Number(v[idx(zz, yy, xx)]);
        }
      }
    }
    taps.push(t);
  }
  return taps;
}

/** Kernel tap k (0..26, dz-major) of channel c. */
const tapWeight = (w: Square3dWeights, c: number, k: number) =>
  w.conv.weight[c][0][Math.floor(k / 9)][Math.floor(k / 3) % 3][k % 3];

/** Class j's dense weights over the layout c * VOXELS + v, each pooled weight divided over its 8 voxels. */
export function pooledDense3d(w: Square3dWeights, j: number): Float64Array {
  check(w);
  const out = new Float64Array(CHANNELS3 * VOXELS);
  for (let c = 0; c < CHANNELS3; c++) {
    for (let z = 0; z < SIDE3; z++) for (let y = 0; y < SIDE3; y++) for (let x = 0; x < SIDE3; x++) {
      const p = ((z >> 1) * POOLED + (y >> 1)) * POOLED + (x >> 1);
      out[c * VOXELS + idx(z, y, x)] = w.dense[0].weight[j][c * POOLED_VOXELS + p] / 8;
    }
  }
  return out;
}

/** The squared convolution outputs, c * VOXELS + v, in float64. */
function squaredConv(v: ArrayLike<number>, w: Square3dWeights): Float64Array {
  const taps = taps3d(v);
  const sq = new Float64Array(CHANNELS3 * VOXELS);
  for (let c = 0; c < CHANNELS3; c++) {
    for (let p = 0; p < VOXELS; p++) {
      let s = w.conv.bias[c];
      for (let k = 0; k < 27; k++) s += tapWeight(w, c, k) * taps[k][p];
      sq[c * VOXELS + p] = s * s;
    }
  }
  return sq;
}

/** The same network in float64: the reference every decrypted logit is compared with. */
export function plainSquare3dLogits(v: ArrayLike<number>, w: Square3dWeights): [number, number] {
  check(w);
  const sq = squaredConv(v, w);
  const out: [number, number] = [0, 0];
  for (let j = 0; j < 2; j++) {
    const d = pooledDense3d(w, j);
    let s = w.dense[0].bias[j];
    for (let i = 0; i < d.length; i++) s += d[i] * sq[i];
    out[j] = s;
  }
  return out;
}

/** Each of the 21,952 positions' share of logit 1 minus logit 0, summed over the four channels. */
export function contributionMap(v: ArrayLike<number>, w: Square3dWeights): Float64Array {
  check(w);
  const sq = squaredConv(v, w);
  const d0 = pooledDense3d(w, 0), d1 = pooledDense3d(w, 1);
  const map = new Float64Array(VOXELS);
  for (let c = 0; c < CHANNELS3; c++) for (let p = 0; p < VOXELS; p++) map[p] += (d1[c * VOXELS + p] - d0[c * VOXELS + p]) * sq[c * VOXELS + p];
  return map;
}

/** The map's total and what it must equal: logit 1 minus logit 0, minus the dense biases' difference. */
export function contributionSum(v: ArrayLike<number>, w: Square3dWeights): { total: number; expected: number } {
  const m = contributionMap(v, w);
  let total = 0;
  for (let i = 0; i < m.length; i++) total += m[i];
  const [l0, l1] = plainSquare3dLogits(v, w);
  return { total, expected: l1 - l0 - (w.dense[0].bias[1] - w.dense[0].bias[0]) };
}

let _seal: Promise<any> | null = null;
function sealModule(): Promise<any> {
  if (!_seal) {
    _seal = (async () => {
      const SEAL = (await import("node-seal/throws")).default;
      return SEAL({ locateFile: (path: string) => (path.endsWith(".wasm") ? sealWasmUrl : path) });
    })();
  }
  return _seal;
}

/** One crop, end to end under CKKS with the named parameter set: the keys are made for this run and discarded. */
export async function runSquare3d(v: ArrayLike<number>, w: Square3dWeights, set: string): Promise<Square3dRun> {
  check(w);
  const p = w.ckks[set];
  if (!p) throw new Error(`the weights offer no parameter set "${set}" (they offer ${Object.keys(w.ckks).join(", ")})`);
  const seal = await sealModule();
  const slots = p.N / 2, blocks = Math.ceil(VOXELS / slots), scale = Math.pow(2, p.scale_bits);
  const steps = Array.from({ length: Math.log2(slots) }, (_, i) => 1 << i);
  const live: { delete(): void }[] = [];
  const keep = <T extends { delete(): void }>(o: T) => { live.push(o); return o; };
  try {
    const tk = performance.now();
    const parms = keep(new seal.EncryptionParameters(seal.SchemeType.ckks));
    parms.setPolyModulusDegree(p.N);
    parms.setCoeffModulus(keep(seal.CoeffModulus.Create(p.N, Int32Array.from(p.coeff_mod_bits))));
    const context = keep(new seal.SEALContext(parms, true, seal.SecLevelType.tc128));
    if (!context.parametersSet()) throw new Error(`SEAL rejected N = ${p.N} with ${p.coeff_mod_bits.join("+")} bits`);
    const keyGen = keep(new seal.KeyGenerator(context));
    const pk = keep(keyGen.createPublicKey()), sk = keep(keyGen.secretKey());
    const encoder = keep(new seal.CKKSEncoder(context));
    const encryptor = keep(new seal.Encryptor(context, pk));
    const decryptor = keep(new seal.Decryptor(context, sk));
    const evaluator = keep(new seal.Evaluator(context));
    const relin = keep(keyGen.createRelinKeys());
    const galois = keep(keyGen.createGaloisKeysWithSteps(Int32Array.from(steps)));
    const msKeys = performance.now() - tk;

    const pt = () => keep(new seal.Plaintext());
    const ct = () => keep(new seal.Ciphertext());
    const block = (values: ArrayLike<number>, b: number, offset = 0) => {
      const out = new Float64Array(slots);
      const start = b * slots;
      for (let i = 0; i < slots && start + i < VOXELS; i++) out[i] = Number(values[offset + start + i]);
      return out;
    };
    const encodeAt = (values: Float64Array, s: number, like: any | null) => {
      const out = pt();
      encoder.encode(values, s, out);
      if (like) evaluator.plainModSwitchToInplace(out, like.parmsId());
      return out;
    };

    // The client: 27 taps x `blocks` blocks, encrypted under the public key.
    const t0 = performance.now();
    const taps = taps3d(v);
    const enc: any[][] = taps.map((tap) => Array.from({ length: blocks }, (_, b) => {
      const out = ct();
      encryptor.encrypt(encodeAt(block(tap, b), scale, null), out);
      return out;
    }));
    let inputBytes = 0;
    for (const row of enc) for (const e of row) inputBytes += e.saveToArray(seal.ComprModeType.none).length;
    const t1 = performance.now();

    // The evaluator: each channel's convolution from the tap blocks, its bias, the square, the pooled dense layer.
    // Each kernel tap is encoded once, as a constant at the inputs' level, and one product ciphertext is reused: a
    // product kept per tap, channel and block would hold 648 ciphertexts, about 0.4 GB at N = 8192.
    const kernel = Array.from({ length: CHANNELS3 }, (_, c) => Array.from({ length: 27 }, (_, k) =>
      encodeAt(new Float64Array(slots).fill(tapWeight(w, c, k)), scale, null)));
    const tmp = ct();
    const squared: any[][] = [];
    for (let c = 0; c < CHANNELS3; c++) {
      squared.push(Array.from({ length: blocks }, (_, b) => {
        const acc = ct();
        evaluator.multiplyPlain(enc[0][b], kernel[c][0], acc);
        for (let k = 1; k < 27; k++) {
          evaluator.multiplyPlain(enc[k][b], kernel[c][k], tmp);
          evaluator.add(acc, tmp, acc);
        }
        evaluator.rescaleToNextInplace(acc);
        // The bias only where a voxel is: the last block's tail stays zero, so the square adds nothing there.
        const bias = new Float64Array(slots);
        for (let i = 0; i < slots && b * slots + i < VOXELS; i++) bias[i] = w.conv.bias[c];
        evaluator.addPlain(acc, encodeAt(bias, acc.scale(), acc), acc);
        const sq = ct();
        evaluator.square(acc, sq);
        evaluator.relinearizeInplace(sq, relin);
        evaluator.rescaleToNextInplace(sq);
        return sq;
      }));
    }
    const outs = [0, 1].map((j) => {
      const dense = pooledDense3d(w, j);
      const sum = ct();
      for (let c = 0; c < CHANNELS3; c++) {
        for (let b = 0; b < blocks; b++) {
          const weights = encodeAt(block(dense, b, c * VOXELS), scale, squared[c][b]);
          if (c === 0 && b === 0) evaluator.multiplyPlain(squared[c][b], weights, sum);
          else {
            evaluator.multiplyPlain(squared[c][b], weights, tmp);
            evaluator.add(sum, tmp, sum);
          }
        }
      }
      evaluator.rescaleToNextInplace(sum);
      const r = ct();
      for (const s of steps) {
        evaluator.rotateVector(sum, s, galois, r);
        evaluator.add(sum, r, sum);
      }
      evaluator.addPlain(sum, encodeAt(new Float64Array(slots).fill(w.dense[0].bias[j]), sum.scale(), sum), sum);
      return sum;
    });
    const outputBytes = outs.reduce((n, o) => n + o.saveToArray(seal.ComprModeType.none).length, 0);
    const t2 = performance.now();

    // The client: two decryptions; every slot holds its logit, and slot 0 is read.
    const logits = outs.map((o) => {
      const plain = pt();
      decryptor.decrypt(o, plain);
      return Number(encoder.decodeFloat64(plain)[0]);
    }) as [number, number];
    const t3 = performance.now();

    const ref = plainSquare3dLogits(v, w);
    const predicted = logits[1] > logits[0] ? 1 : 0;
    return {
      set, logits, plain: ref,
      maxAbsError: Math.max(Math.abs(logits[0] - ref[0]), Math.abs(logits[1] - ref[1])),
      decisionAgrees: predicted === (ref[1] > ref[0] ? 1 : 0), predicted,
      ciphertexts: { input: 27 * blocks, output: 2 },
      bytes: { input: inputBytes, output: outputBytes },
      ms: { keys: msKeys, encrypt: t1 - t0, evaluate: t2 - t1, decrypt: t3 - t2 },
    };
  } finally {
    for (const o of live.reverse()) o.delete();
  }
}

/** The same run off the page's thread. One crop holds about 85 MB of ciphertexts and takes about two seconds under
 *  either set in node, so on the page's thread it would hold input for seconds; like the 2D student, this runs in a
 *  dedicated worker in a browser and on the calling thread elsewhere, and a worker that fails is reported as the
 *  failure it is, never retried on the page's thread. Every ciphertext is freed inside `runSquare3d`, and the worker
 *  is terminated after the run, which frees its wasm heap. */
export async function runSquare3dOffThread(v: ArrayLike<number>, w: Square3dWeights, set: string):
    Promise<{ run: Square3dRun; ranIn: string }> {
  if (typeof window === "undefined" || typeof Worker === "undefined") {
    return { run: await runSquare3d(v, w, set), ranIn: "the calling thread" };
  }
  // Vite bundles the worker from exactly this form: `new URL(...)` written inside `new Worker(...)`.
  const worker = new Worker(new URL("./clientFHESquare3d.worker.ts", import.meta.url), { type: "module" });
  try {
    return await new Promise((resolve, reject) => {
      worker.onmessage = (e: MessageEvent<{ id: number; ok: boolean; run?: Square3dRun; error?: string }>) => {
        const d = e.data;
        if (d.ok && d.run) resolve({ run: d.run, ranIn: "a Web Worker" });
        else reject(new Error(`in its worker: ${d.error ?? "no result"}`));
      };
      worker.onerror = (e) => reject(new Error(`its worker failed: ${e.message || "the worker could not be loaded"}`));
      worker.onmessageerror = () => reject(new Error("its worker's result could not be read back"));
      worker.postMessage({ id: 1, crop: Array.from(v), weights: w, set });
    });
  } finally {
    worker.terminate();
  }
}

/** What one run holds at its peak, measured in node (about 0.6 GB resident under either set), against what the
 *  device reports. `navigator.deviceMemory` is Chromium's, rounded and capped at 8 GB; elsewhere there is no figure,
 *  and nothing is said. */
export function memoryNote(): string | null {
  const gb = typeof navigator === "undefined" ? undefined : (navigator as { deviceMemory?: number }).deviceMemory;
  if (gb === undefined || gb >= 4) return null;
  return `This device reports about ${gb} GB of memory, and one run holds about 0.6 GB while it works: it may be slow, ` +
    "or the tab may be stopped.";
}

// REAL in-browser CKKS homomorphic encryption via node-seal (MIT — Microsoft SEAL → WebAssembly).
// This replaces the demo's old representative FHE cost model with a GENUINE encrypted linear layer
// computed entirely client-side: the feature vector is CKKS-encrypted, the linear layer (W·x + b) runs
// homomorphically (encrypted multiply-plain + slot rotate-and-add for the dot product), and only the
// decrypted logits leave encryption — so the demo's FHE-linear numbers are REAL MEASURED wall-clock,
// with ZERO server/VPS compute. node-seal is lazy-loaded (dynamic import → async chunk; the ~few-MB
// WASM is browser-cached). No commercial-OK WebGPU FHE exists in 2026, so CKKS runs on WASM-CPU.
//
// node-seal v7 API note: raw SEAL class names with `new`, `SEALContext`/`SecLevelType`, and in-place
// encode/encrypt (encode(arr, scale, destPlaintext); encrypt(pt, destCt); decodeFloat64(pt)).

// The Vite-emitted, content-hashed URL of node-seal's wasm. node-seal's default loader resolves the
// binary via `new URL("seal_throws.wasm", import.meta.url)` — the UN-hashed name — which 404s under
// Vite's hashed assets (it renames the file to seal_throws-<hash>.wasm) and the SPA fallback then
// returns index.html, so the wasm "magic word" check fails. Importing the wasm with Vite's `?url`
// query yields the real hashed asset URL, which we feed to `locateFile` below. This keeps node-seal
// itself lazy (the heavy glue + wasm load only when `init()` dynamic-imports the module).
// Root-absolute path (Vite root = web/) deliberately bypasses node-seal's "exports" map, which only
// exposes ./throws + ./allows and blocks the deep ./dist/*.wasm specifier; this is a direct filesystem
// asset reference, so Vite emits + hashes it normally and returns the served URL.
import sealWasmUrl from "/node_modules/node-seal/dist/seal_throws.wasm?url";

let _ctx: Promise<SealCtx> | null = null;

interface SealCtx {
  seal: any;
  encoder: any;
  encryptor: any;
  decryptor: any;
  evaluator: any;
  galois: any;
  slots: number;
  scale: number;
}

const POLY_MODULUS_DEGREE = 8192;          // CKKS ring degree (NIST-128 security at this coeff-modulus)
const COEFF_MOD_BITS = [60, 40, 40, 60];   // one multiplication of depth (enough for a linear layer)

//: Power-of-two rotation steps 1 .. 2^(GALOIS_STEP_BITS-1) are pre-generated. 2^10 = 1024 slots is
//: far above any feature width this demo evaluates (the widest published model takes 32), and a
//: request beyond it throws in SEAL rather than degrading silently -- which is the intent.
const GALOIS_STEP_BITS = 11;

async function init(): Promise<SealCtx> {
  const SEAL = (await import("node-seal/throws")).default;
  // Override the default (broken-under-Vite) wasm locator with the real hashed asset URL.
  const seal = await SEAL({
    locateFile: (path: string) => (path.endsWith(".wasm") ? sealWasmUrl : path),
  });
  const parms = new seal.EncryptionParameters(seal.SchemeType.ckks);
  parms.setPolyModulusDegree(POLY_MODULUS_DEGREE);
  parms.setCoeffModulus(seal.CoeffModulus.Create(POLY_MODULUS_DEGREE, Int32Array.from(COEFF_MOD_BITS)));
  const context = new seal.SEALContext(parms, true, seal.SecLevelType.tc128);
  const keyGen = new seal.KeyGenerator(context);
  const publicKey = keyGen.createPublicKey();
  const secretKey = keyGen.secretKey();
  // Only the steps the rotate-and-add slot sum below actually uses. `createGaloisKeys()` with no
  // argument generates the full default set for a degree-8192 ring -- key generation is the single
  // most expensive thing in `warmFHE()`, and it runs synchronously on the main thread at first Run,
  // which is what made the first click feel like a hang. The loop at the bottom of this file steps
  // 1, 2, 4, ... < sumWidth, so those powers of two are sufficient and anything else would be a
  // rotation the code never asks for.
  const rotationSteps = Int32Array.from(
    Array.from({ length: GALOIS_STEP_BITS }, (_, i) => 1 << i));
  // node-seal exposes the step-limited variant under its own name; createGaloisKeys() takes none.
  const galois = keyGen.createGaloisKeysWithSteps(rotationSteps);
  return {
    seal,
    encoder: new seal.CKKSEncoder(context),
    encryptor: new seal.Encryptor(context, publicKey),
    decryptor: new seal.Decryptor(context, secretKey),
    evaluator: new seal.Evaluator(context),
    galois,
    slots: POLY_MODULUS_DEGREE / 2,
    scale: Math.pow(2, 40),
  };
}

function getCtx(): Promise<SealCtx> {
  return (_ctx ??= init());
}

/** Warm the CKKS context (keygen) so the first inference doesn't pay it. */
export async function warmFHE(): Promise<void> {
  await getCtx();
}

export interface FHEResult {
  logits: number[];        // decrypted W·x + b
  fheMs: number;           // REAL measured wall-clock of the encrypted computation
  ctBytes: number;         // serialized ciphertext size (for a genuine ciphertext/plaintext ratio)
  maxAbsError: number;     // CKKS approximation error vs the plaintext reference
  /** Homomorphic rotations performed, counted. Each triggers a key-switch, and key-switching is
   *  what dominates CKKS cost - so this is the number that makes a dense layer and a convolution
   *  comparable on the axis that actually matters, rather than on wall-clock alone. */
  rotations: number;
  multiplies: number;
}

/**
 * Genuinely encrypt ``x`` and evaluate the linear layer ``W·x + b`` under CKKS, returning the decrypted
 * logits + the REAL measured time + the ciphertext size. ``W`` is ``(out, in)``. The dot product is a
 * homomorphic multiply-plain followed by a rotate-and-add slot sum (the standard packed-CKKS inner
 * product). Run on a small model — CKKS on WASM-CPU is ~hundreds of ms.
 */
export async function encryptedLinear(W: number[][], b: number[], x: number[]): Promise<FHEResult> {
  const c = await getCtx();
  const { seal, encoder, encryptor, decryptor, evaluator, galois, slots, scale } = c;
  const n = x.length;
  const sumWidth = 1 << Math.ceil(Math.log2(Math.max(n, 2)));   // power-of-2 ≥ n for the slot reduction

  const t0 = performance.now();
  const xPad = Float64Array.from({ length: slots }, (_, i) => (i < n ? x[i] : 0));
  const ptX = seal.PlainText ? seal.PlainText() : new seal.Plaintext();
  encoder.encode(xPad, scale, ptX);
  const ctX = new seal.Ciphertext();
  encryptor.encrypt(ptX, ctX);
  // serialized ciphertext size (node-seal v7: saveToArray(ComprModeType)) → genuine ct/pt ratio
  let ctBytes = 0;
  try { ctBytes = ctX.saveToArray(seal.ComprModeType.none).length; } catch { ctBytes = 0; }

  const logits: number[] = [];
  let rotations = 0, multiplies = 0;
  for (let j = 0; j < W.length; j++) {
    // an all-zero weight row (our 2-class "class 0" convention) → logit is just the bias; skip the
    // homomorphic eval (CKKS would yield a "transparent" ciphertext, which the safe build rejects).
    if (W[j].every((v) => v === 0)) {
      logits.push(b[j] ?? 0);
      continue;
    }
    const wPad = Float64Array.from({ length: slots }, (_, i) => (i < n ? (W[j][i] ?? 0) : 0));
    const ptW = new seal.Plaintext();
    encoder.encode(wPad, scale, ptW);
    let acc = new seal.Ciphertext();
    evaluator.multiplyPlain(ctX, ptW, acc);                     // encrypted element-wise W_j ⊙ x
    multiplies++;
    for (let s = 1; s < sumWidth; s <<= 1) {                    // rotate-and-add → Σ slots in slot 0
      const rot = new seal.Ciphertext();
      evaluator.rotateVector(acc, s, galois, rot);
      const nxt = new seal.Ciphertext();
      evaluator.add(acc, rot, nxt);
      acc = nxt;
      rotations++;
    }
    const out = new seal.Plaintext();
    decryptor.decrypt(acc, out);
    logits.push(encoder.decodeFloat64(out)[0] + (b[j] ?? 0));
  }
  const fheMs = performance.now() - t0;

  // approximation error vs the plaintext reference (CKKS is approximate)
  const plain = W.map((row, j) => row.reduce((a, w, i) => a + w * (x[i] ?? 0), 0) + (b[j] ?? 0));
  const maxAbsError = Math.max(...logits.map((v, j) => Math.abs(v - plain[j])), 0);
  return { logits, fheMs, ctBytes, maxAbsError, rotations, multiplies };
}

export function argmax(a: number[]): number {
  return a.reduce((bi, v, i, arr) => (v > arr[bi] ? i : bi), 0);
}

export interface ConvResult {
  /** Decrypted feature map, (out_ch, Hout, Wout) flattened per channel. */
  maps: number[][];
  outH: number;
  outW: number;
  fheMs: number;
  ctBytes: number;
  maxAbsError: number;
  /** Homomorphic rotations performed. Each one triggers a key-switch, which is what dominates. */
  rotations: number;
  /** Ciphertext-plaintext multiplications performed. */
  multiplies: number;
  /** How many im2col patches shared one ciphertext. 1 would mean no packing at all. */
  patchesPerCiphertext: number;
}

/**
 * A genuine encrypted 2-D convolution, mirroring ``src/qsh/fhe/conv.py``.
 *
 * Until this existed the browser's "CNN workload" ran the same 8-feature dot product as the linear
 * one and multiplied the measured latency by a cost model. The page reported a convolution that
 * never happened. This performs one.
 *
 * PACKING IS THE POINT, and it is why the cost model was misleading in both directions. The naive
 * form encrypts one patch at a time: 12x12 output positions x 4 channels is 576 dot products, each
 * needing its own rotate-and-add, so ~2,300 rotations and several seconds in a tab. Instead every
 * im2col patch goes into ONE ciphertext, each padded to a power-of-two block, so a single
 * multiply-plain against the tiled kernel plus log2(block) rotations produces every output position
 * for that channel at once. That is the technique the 2026 CKKS-inference literature is largely
 * about, and it turns the rotation count from a function of the output size into a function of the
 * kernel size.
 *
 * The returned ``rotations`` and ``multiplies`` are counted, not estimated, because rotations and
 * the key-switch each one triggers are what actually drive CKKS cost - so the page can report the
 * quantity the literature identifies instead of a latency multiplier.
 *
 * CHOICE OF PACKING, and its known cost. im2col lowering is one of two standard families; the other
 * packs the whole image into a single ciphertext and moves a rotating kernel over it. Published
 * comparisons favour the rotation family on both peak memory and speed, because lowering's memory
 * grows with kernel size while the rotation form stays roughly constant. im2col is used here anyway
 * and deliberately: it is the construction ``src/qsh/fhe/conv.py`` implements, and a browser mirror
 * that computed a different convolution than the backend would make the two incomparable, which is
 * the whole point of having both. At 14x14 with a 3x3 kernel the memory difference is immaterial -
 * 2,304 of 4,096 slots - so the fidelity is worth more than the saving. Rotation-free formulations
 * also exist (arXiv:2409.05205) and are not used.
 */
export async function encryptedConv2d(
  image: number[][], kernels: number[][][], biases: number[], padding = 0,
): Promise<ConvResult> {
  const c = await getCtx();
  const { seal, encoder, encryptor, decryptor, evaluator, galois, slots, scale } = c;

  // Zero-pad first, so a trained model's `padding` is honoured rather than silently ignored. The
  // student uses pad 1, which keeps 14x14 through the conv; running it as a valid convolution
  // would give 12x12 and a flatten of the wrong length for its dense layer.
  const src = padding > 0
    ? Array.from({ length: image.length + 2 * padding }, (_, y) =>
        Array.from({ length: image[0].length + 2 * padding }, (_, x) =>
          (y >= padding && y < image.length + padding && x >= padding && x < image[0].length + padding)
            ? image[y - padding][x - padding] : 0))
    : image;
  const H = src.length, W = src[0].length;
  const kH = kernels[0].length, kW = kernels[0][0].length;
  const outH = H - kH + 1, outW = W - kW + 1;
  const patchLen = kH * kW;
  // Power-of-two block per patch so the rotate-and-add reduction lands the sum in the block's
  // first slot without bleeding into its neighbour.
  const block = 1 << Math.ceil(Math.log2(Math.max(patchLen, 2)));
  const nPatches = outH * outW;
  if (nPatches * block > slots) {
    throw new Error(`conv needs ${nPatches * block} slots, ring provides ${slots} - `
      + "reduce the input or the kernel rather than silently tiling, which would change the result");
  }

  const t0 = performance.now();

  // im2col, packed: patch p occupies slots [p*block, p*block + patchLen).
  const packed = new Float64Array(slots);
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      const base = (oy * outW + ox) * block;
      for (let ky = 0; ky < kH; ky++) {
        for (let kx = 0; kx < kW; kx++) packed[base + ky * kW + kx] = src[oy + ky][ox + kx];
      }
    }
  }
  const ptX = new seal.Plaintext();
  encoder.encode(packed, scale, ptX);
  const ctX = new seal.Ciphertext();
  encryptor.encrypt(ptX, ctX);
  let ctBytes = 0;
  try { ctBytes = ctX.saveToArray(seal.ComprModeType.none).length; } catch { ctBytes = 0; }

  let rotations = 0, multiplies = 0;
  const maps: number[][] = [];
  for (let oc = 0; oc < kernels.length; oc++) {
    // The same kernel tiled into every patch block, so one multiply-plain covers all positions.
    const tiled = new Float64Array(slots);
    for (let p = 0; p < nPatches; p++) {
      for (let ky = 0; ky < kH; ky++) {
        for (let kx = 0; kx < kW; kx++) tiled[p * block + ky * kW + kx] = kernels[oc][ky][kx];
      }
    }
    const ptW = new seal.Plaintext();
    encoder.encode(tiled, scale, ptW);
    let acc = new seal.Ciphertext();
    evaluator.multiplyPlain(ctX, ptW, acc);
    multiplies++;
    // Reduce WITHIN each block. Only log2(block) rotations, independent of how many patches ride
    // along - which is exactly what the packing buys.
    for (let s = 1; s < block; s <<= 1) {
      const rot = new seal.Ciphertext();
      evaluator.rotateVector(acc, s, galois, rot);
      const nxt = new seal.Ciphertext();
      evaluator.add(acc, rot, nxt);
      acc = nxt;
      rotations++;
    }
    const out = new seal.Plaintext();
    decryptor.decrypt(acc, out);
    const dec = encoder.decodeFloat64(out);
    const map = new Array<number>(nPatches);
    for (let p = 0; p < nPatches; p++) map[p] = dec[p * block] + (biases[oc] ?? 0);
    maps.push(map);
  }
  const fheMs = performance.now() - t0;

  // Plaintext reference, so the approximation error is measured rather than assumed.
  let maxAbsError = 0;
  for (let oc = 0; oc < kernels.length; oc++) {
    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        let s = 0;
        for (let ky = 0; ky < kH; ky++) {
          for (let kx = 0; kx < kW; kx++) s += src[oy + ky][ox + kx] * kernels[oc][ky][kx];
        }
        maxAbsError = Math.max(maxAbsError, Math.abs(maps[oc][oy * outW + ox] - (s + (biases[oc] ?? 0))));
      }
    }
  }
  return { maps, outH, outW, fheMs, ctBytes, maxAbsError,
           rotations, multiplies, patchesPerCiphertext: nPatches };
}

/** The published student's structure, as written by `distil.save_cnn_student_weights`. */
export interface CnnStudentWeights {
  kind: string;
  conv: { in_ch: number; out_ch: number; kernel: number; padding: number;
          activation: string; weight: number[][][][]; bias: number[] | null };
  pool: { type: string; stride: number };
  dense: { in: number; out: number; weight: number[][]; bias: number[] | null; activation: string }[];
  input: { shape: number[]; preprocessing: string[] };
  n_parameters: number;
}

export interface EncryptedStudentResult extends ConvResult {
  /** Decrypted logits of the FULL student, not just the conv block. */
  logits: number[];
  prediction: number;
  /** Rotations across every CKKS layer, conv and dense. */
  totalRotations: number;
  totalMultiplies: number;
  /** Max |encrypted - plaintext| over the logits, so the approximation cost is measured. */
  logitMaxAbsError: number;
}

/**
 * Evaluate the PUBLISHED medical student end to end under CKKS.
 *
 * Until this existed the encrypted convolution ran on the student's SHAPE with a synthetic kernel
 * pattern. It was a genuine homomorphic convolution and an honest cost model, but it was not this
 * model's inference - so `crypto_fidelity`, the project's headline metric, stayed a number measured
 * offline that a visitor could not check. Feeding the real weights makes the encrypted path compute
 * the published prediction.
 *
 * The layer order comes from the weights file rather than from an assumption here: conv, the conv's
 * own activation, average pool, flatten, then each dense with its recorded activation. Omitting the
 * conv's ReLU still reproduces every argmax on the test split while shifting the logits by ~0.15,
 * which is exactly why the file states each activation instead of leaving it to be inferred.
 *
 * ReLU is applied in the clear here. That is the project's stated hybrid - linear layers under FHE,
 * non-linear under MPC - and the argmax genuinely runs under garbled circuits in `clientMPC`; the
 * intermediate ReLU is not yet routed through it, and that gap is reported rather than papered over.
 */
export async function encryptedStudentInference(
  w: CnnStudentWeights, image: number[][],
): Promise<EncryptedStudentResult> {
  const kernels = w.conv.weight.map((oc) => oc[0]);          // in_ch is 1 for this student
  const biases = w.conv.bias ?? new Array(w.conv.out_ch).fill(0);
  const conv = await encryptedConv2d(image, kernels, biases, w.conv.padding);

  const relu = (v: number) => (v > 0 ? v : 0);
  const act = (v: number, kind: string) => (kind === "relu" ? relu(v) : v);

  // conv activation, then average pool, then flatten channel-major - the order the file states.
  const s = w.pool.stride;
  const ph = Math.floor(conv.outH / s), pw = Math.floor(conv.outW / s);
  const flat: number[] = [];
  for (let c = 0; c < kernels.length; c++) {
    for (let y = 0; y < ph; y++) {
      for (let x = 0; x < pw; x++) {
        let acc = 0;
        for (let dy = 0; dy < s; dy++) {
          for (let dx = 0; dx < s; dx++) {
            acc += act(conv.maps[c][(y * s + dy) * conv.outW + (x * s + dx)], w.conv.activation);
          }
        }
        flat.push(acc / (s * s));
      }
    }
  }

  let v = flat;
  let rotations = conv.rotations, multiplies = conv.multiplies, fheMs = conv.fheMs, ctBytes = conv.ctBytes;
  for (const d of w.dense) {
    if (v.length !== d.in) {
      throw new Error(`dense layer expects ${d.in} inputs, got ${v.length} - the published weights `
        + "and this evaluation disagree about the network, which must not be silently reshaped");
    }
    const r = await encryptedLinear(d.weight, d.bias ?? new Array(d.out).fill(0), v);
    rotations += r.rotations; multiplies += r.multiplies; fheMs += r.fheMs;
    ctBytes = Math.max(ctBytes, r.ctBytes);
    v = r.logits.map((x) => act(x, d.activation));
  }

  // Plaintext reference over the same weights, so the CKKS error is measured rather than assumed.
  const ref = (() => {
    const p = w.conv.padding, k = w.conv.kernel;
    const H = image.length + 2 * p;
    const pad = Array.from({ length: H }, (_, y) =>
      Array.from({ length: H }, (_, x) =>
        (y >= p && y < image.length + p && x >= p && x < image[0].length + p) ? image[y - p][x - p] : 0));
    const oh = H - k + 1;
    const fm: number[][][] = kernels.map((kern, oc) =>
      Array.from({ length: oh }, (_, y) => Array.from({ length: oh }, (_, x) => {
        let a = biases[oc];
        for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) a += pad[y + ky][x + kx] * kern[ky][kx];
        return act(a, w.conv.activation);
      })));
    const fl: number[] = [];
    for (let c = 0; c < fm.length; c++)
      for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++) {
        let a = 0;
        for (let dy = 0; dy < s; dy++) for (let dx = 0; dx < s; dx++) a += fm[c][y * s + dy][x * s + dx];
        fl.push(a / (s * s));
      }
    let u = fl;
    for (const d of w.dense) {
      u = d.weight.map((row, j) => act(row.reduce((a, wt, i) => a + wt * u[i], 0) + (d.bias?.[j] ?? 0), d.activation));
    }
    return u;
  })();

  return {
    ...conv, fheMs, ctBytes,
    logits: v, prediction: argmax(v),
    totalRotations: rotations, totalMultiplies: multiplies,
    logitMaxAbsError: Math.max(...v.map((x, i) => Math.abs(x - ref[i])), 0),
  };
}

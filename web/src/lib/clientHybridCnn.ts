/**
 * The published medical CNN student under MPC: CKKS for its linear layers, garbled circuits for its ReLUs and its argmax.
 *
 * The protocol's shape is Gazelle's (USENIX Security 2018). A client holds the image and the CKKS secret key, a server holds the
 * weights, and between layers they hold additive shares of the activations:
 *
 *   1. the client encrypts its input; the server evaluates a linear layer under CKKS and adds an encrypted mask r to every slot;
 *   2. the client decrypts x + r, so the parties hold shares of x, and neither sees x;
 *   3. a garbled circuit reconstructs x, applies the ReLU, and adds a fresh mask s of the server's, and only the client gets the result;
 *   4. average pooling is linear, so each party applies it to its own share;
 *   5. the client encrypts its share of the next layer's input, and the server removes its own under encryption.
 *
 * The last layer's shares go into the existing 2PC argmax, so neither party learns the logits and both learn the class.
 *
 * WHY IT IS A SEPARATE MODULE. `clientFHE.ts` is cited by a submitted poster and stays as it is until the notification. It decrypts each layer's
 * outputs in the clear; this module is the path that does not.
 *
 * WHAT THIS IS NOT, and the page says so. Both parties run in one tab, as the page's other 2PC does, so the separation is notional while
 * the protocol, its messages and its costs are real. The protocol is semi-honest: Polytune's garbling is designed against malicious parties, but
 * the CKKS half and the shares are unauthenticated, and Lehmkuhl et al. (MUSE, USENIX Security 2021) extract weights from protocols of this shape with
 * a malicious client. And the masks are bounded reals, so they hide a value only statistically:
 *
 *     for r uniform on [-R, R), SD(x + r, x' + r) = |x - x'| / (2R).
 *
 * Gazelle, Delphi and the 2025-26 CKKS hybrids (BLB, EncFormer, FESC) mask uniformly over the plaintext ring, which is perfect hiding, and their
 * security theorems do not cover a mask like this one. Those conversions need about two million AND gates per ciphertext in Polytune, so
 * they cannot run in a tab. `statisticalDistance` below is this project's own bound for this run, summed over every masked value. It does
 * not reach 2^-40 at double precision: the decode error is about 6e-16 * R, which caps R.
 *
 * WHAT THE MASKS COVER. A CKKS key holder sees every slot's real and imaginary parts, and the noise there depends on the weights. Masks encoded from
 * real values leave the imaginary half unmasked, and node-seal 7.0.0 encodes only real vectors. So the server adds the conjugate of the ciphertext
 * first, which leaves 2 Re(slot) and cancels the imaginary parts, and the client halves what it decodes.
 *
 * Why not retrain to x^2 + x, as Flash does (arXiv:2401.16732): its activation squares a masked integer, so it needs exact BFV arithmetic. Over
 * CKKS reals with a 2^30 mask the square is about 2^60, and the decode error about 700.
 */

import sealWasmUrl from "/node_modules/node-seal/dist/seal_throws.wasm?url";
import type { CnnStudentWeights } from "./clientFHE";
import type { ReluBatchResult } from "./clientMPC";
import { fheCfg, hybridCnnCfg } from "./config";
import { randomWords } from "./csprng";

export interface HybridCnnProgress { phase: "keys" | "conv" | "relu" | "dense" | "argmax"; done: number; total: number }

export interface HybridCnnResult {
  /** The class from the 2PC argmax. */
  prediction: number;
  /** The same network in the clear, on the same image: a self-test, available only because both parties are in this tab. */
  clearPrediction: number;
  matchesClear: boolean;
  /** Both parties' shares of the logits added together, and the largest error against the clear network. Neither party learns these in a real deployment. */
  logits: number[];
  logitMaxAbsError: number;
  /** ReLUs in the garbled circuits, across the conv and the dense layers. */
  reluCount: number;
  andGates: number;
  batches: number;
  /** Garbled-circuit traffic between the parties, the argmax's included. */
  gcBytes: number;
  gcMessages: number;
  gcCompileMs: number;
  gcMs: number;
  /** Ciphertexts actually serialized across the boundary, and the keys the server received. */
  ckksBytes: number;
  keyBytes: number;
  ckksMs: number;
  rotations: number;
  conjugations: number;
  totalMs: number;
  /** log2 of the mask half-width per layer, and this run's bound on the statistical distance of everything masked, summed. */
  maskBits: { conv: number; dense: number; logit: number };
  statisticalDistance: number;
  statisticalDistanceLog2: number;
  /** The largest |reconstructed - clear| over every layer's outputs: CKKS noise, double-precision decoding and fixed-point rounding, accumulated. Measured, not a proven bound. */
  maxLayerError: number;
  /** The median |value| the client decoded across every layer's outputs: about half the mask's width, so what it saw was masked. */
  clientViewMedianAbs: number;
}

export interface Sealed { seal: any; context: any; slots: number; scale: number }

let _sealed: Promise<Sealed> | null = null;

export function sealed(): Promise<Sealed> {
  return (_sealed ??= (async () => {
    const SEAL = (await import("node-seal/throws")).default;
    const seal = await SEAL({ locateFile: (path: string) => (path.endsWith(".wasm") ? sealWasmUrl : path) });
    const N = fheCfg().poly_modulus_degree;
    const parms = new seal.EncryptionParameters(seal.SchemeType.ckks);
    parms.setPolyModulusDegree(N);
    parms.setCoeffModulus(seal.CoeffModulus.Create(N, Int32Array.from(fheCfg().coeff_mod_bit_sizes)));
    return { seal, context: new seal.SEALContext(parms, true, seal.SecLevelType.tc128), slots: N / 2, scale: 2 ** fheCfg().global_scale_bits };
  })().catch((e) => { _sealed = null; throw e; }));
}

/** Frees every SEAL object it was handed, including on a throw. */
export class Arena {
  private readonly live: { delete(): void }[] = [];
  hold<T extends { delete(): void }>(o: T): T { this.live.push(o); return o; }
  close(): void { for (const o of this.live) o.delete(); this.live.length = 0; }
}

/** Independent reals uniform on [-2^bits, 2^bits), from the CSPRNG. */
function uniformReal(n: number, bits: number): Float64Array {
  const w = randomWords(2 * n);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = (2 * (w[2 * i] * 2 ** 21 + (w[2 * i + 1] >>> 11)) / 2 ** 53 - 1) * 2 ** bits;
  return out;
}

/** Independent integers uniform on [0, 2^bits), bits at most 52, from the CSPRNG. */
function uniformBits(n: number, bits: number): Float64Array {
  if (bits <= 32 || bits > 52) throw new Error(`an output mask of ${bits} bits is outside (32, 52]`);
  const w = randomWords(2 * n);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = (w[2 * i] >>> (64 - bits)) * 2 ** 32 + w[2 * i + 1];
  return out;
}

/** round(v) reduced to [0, modulus), for |v| below 2^53, where a double still holds the integer. */
export function toRing(v: number, modulus: number): number {
  const m = Math.round(v) % modulus;
  return m < 0 ? m + modulus : m;
}

// Dynamic, so this module costs nothing until the run asks for it, and the node harnesses that stand in a plaintext argmax for `clientMPC` still bundle.
type Mpc = typeof import("./clientMPC");

/** The party with the secret key and the image. */
export class Client {
  private readonly arena = new Arena();
  private readonly encoder: any;
  private readonly encryptor: any;
  private readonly decryptor: any;
  private readonly pt: any;
  private readonly secretKey: any;
  readonly publicKey: Uint8Array;
  readonly galoisKeys: Uint8Array;

  constructor(private readonly s: Sealed, steps: number[]) {
    const { seal, context } = s;
    const gen = this.arena.hold(new seal.KeyGenerator(context));
    const pk = this.arena.hold(gen.createPublicKey());
    this.publicKey = pk.saveToArray(seal.ComprModeType.none);
    this.galoisKeys = this.arena.hold(gen.createGaloisKeysWithSteps(Int32Array.from(steps))).saveToArray(seal.ComprModeType.none);
    this.encoder = this.arena.hold(new seal.CKKSEncoder(context));
    this.encryptor = this.arena.hold(new seal.Encryptor(context, pk));
    this.secretKey = this.arena.hold(gen.secretKey());
    this.decryptor = this.arena.hold(new seal.Decryptor(context, this.secretKey));
    this.pt = this.arena.hold(new seal.Plaintext());
  }

  /** Her key as bytes, so a key holder's other threads can load it (lending/privateCheck.ts). Every party is in this tab. */
  exportSecretKey(): Uint8Array {
    return this.secretKey.saveToArray(this.s.seal.ComprModeType.none);
  }

  encrypt(values: Float64Array): Uint8Array {
    const ct = new this.s.seal.Ciphertext();
    try {
      this.encoder.encode(values, this.s.scale, this.pt);
      this.encryptor.encrypt(this.pt, ct);
      return ct.saveToArray(this.s.seal.ComprModeType.none);
    } finally { ct.delete(); }
  }

  /** Every slot's real part. The server doubled it when it cancelled the imaginary half, so this halves it. */
  decrypt(bytes: Uint8Array): Float64Array {
    const ct = new this.s.seal.Ciphertext();
    try {
      ct.loadFromArray(this.s.context, bytes);
      this.decryptor.decrypt(ct, this.pt);
      return Float64Array.from(this.encoder.decodeFloat64(this.pt) as ArrayLike<number>, (v) => v / 2);
    } finally { ct.delete(); }
  }

  close(): void { this.arena.close(); }
}

/** The party with the weights. It holds a public key and rotation keys, never the secret key. */
export class Server {
  private readonly arena = new Arena();
  private readonly encoder: any;
  private readonly encryptor: any;
  private readonly evaluator: any;
  private readonly galois: any;
  rotations = 0;
  conjugations = 0;

  constructor(private readonly s: Sealed, publicKey: Uint8Array, galoisKeys: Uint8Array) {
    const { seal, context } = s;
    const pk = this.arena.hold(new seal.PublicKey());
    pk.loadFromArray(context, publicKey);
    this.galois = this.arena.hold(new seal.GaloisKeys());
    this.galois.loadFromArray(context, galoisKeys);
    this.encoder = this.arena.hold(new seal.CKKSEncoder(context));
    this.encryptor = this.arena.hold(new seal.Encryptor(context, pk));
    this.evaluator = this.arena.hold(new seal.Evaluator(context));
  }

  /**
   * One encrypted block: multiply by the tiled weights, sum within each `block`, add the bias at the output slots, cancel the imaginary half, and add
   * an encrypted mask to every slot. Returns the ciphertext the client decrypts and the mask the server keeps.
   */
  evaluate(ctX: any, tile: Float64Array, block: number, outputs: number, bias: (i: number) => number, maskBits: number): { bytes: Uint8Array; mask: Float64Array } {
    const { seal, slots, scale } = this.s;
    const a = new Arena();
    try {
      const ptW = a.hold(new seal.Plaintext());
      this.encoder.encode(tile, scale, ptW);
      let acc = a.hold(new seal.Ciphertext());
      this.evaluator.multiplyPlain(ctX, ptW, acc);
      for (let step = 1; step < block; step <<= 1) {
        const rot = a.hold(new seal.Ciphertext());
        this.evaluator.rotateVector(acc, step, this.galois, rot);
        const next = a.hold(new seal.Ciphertext());
        this.evaluator.add(acc, rot, next);
        acc = next;
        this.rotations++;
      }
      const biasSlots = new Float64Array(slots);
      for (let i = 0; i < outputs; i++) biasSlots[i * block] = bias(i);
      const ptB = a.hold(new seal.Plaintext());
      this.encoder.encode(biasSlots, scale * scale, ptB);
      const withBias = a.hold(new seal.Ciphertext());
      this.evaluator.addPlain(acc, ptB, withBias);
      // Conjugate and add: 2 Re(slot), every imaginary part gone.
      const conj = a.hold(new seal.Ciphertext());
      this.evaluator.complexConjugate(withBias, this.galois, conj);
      this.conjugations++;
      const real = a.hold(new seal.Ciphertext());
      this.evaluator.add(withBias, conj, real);
      // The mask, under the public key, so the addition also re-randomises the ciphertext's mask term (Gazelle, Appendix B).
      const mask = uniformReal(slots, maskBits);
      const ptM = a.hold(new seal.Plaintext());
      this.encoder.encode(mask.map((m) => 2 * m), scale * scale, ptM);
      const ctM = a.hold(new seal.Ciphertext());
      this.encryptor.encrypt(ptM, ctM);
      const masked = a.hold(new seal.Ciphertext());
      this.evaluator.add(real, ctM, masked);
      return { bytes: masked.saveToArray(seal.ComprModeType.none), mask };
    } finally { a.close(); }
  }

  receive(bytes: Uint8Array, arena: Arena): any {
    const ct = arena.hold(new this.s.seal.Ciphertext());
    ct.loadFromArray(this.s.context, bytes);
    return ct;
  }

  /** The client's share is under encryption; this adds the server's own, at the input scale, so the ciphertext encrypts the activation. */
  addShare(ct: any, share: Float64Array, arena: Arena): any {
    const pt = arena.hold(new this.s.seal.Plaintext());
    this.encoder.encode(share, this.s.scale, pt);
    const out = arena.hold(new this.s.seal.Ciphertext());
    this.evaluator.addPlain(ct, pt, out);
    return out;
  }

  close(): void { this.arena.close(); }
}

/** The garbled circuits' running totals. */
interface GcTotals { and: number; batches: number; bytes: number; messages: number; compileMs: number; mpcMs: number }

function addBatch(t: GcTotals, r: Pick<ReluBatchResult, "compileMs" | "mpcMs" | "bytes" | "messages" | "andOps">): void {
  t.and += r.andOps; t.batches++; t.bytes += r.bytes; t.messages += r.messages; t.compileMs += r.compileMs; t.mpcMs += r.mpcMs;
}

/**
 * A layer's ReLUs, in batches. The client holds `y` = x + r and the server `r`; afterwards the client holds relu(x) + s and the server -s, as reals.
 * The output mask is `frac + maskBits` bits of fixed point, which stays below 2^53, so both shares are exact doubles.
 */
async function reluLayer(mpc: Mpc, y: ArrayLike<number>, r: ArrayLike<number>, maskBits: number, batch: number, gc: GcTotals,
                         onBatch: () => void): Promise<{ client: Float64Array; server: Float64Array }> {
  const { frac_bits: frac, relu_ring_bits: ringBits } = hybridCnnCfg();
  const unit = 2 ** frac, ring = 2 ** ringBits, outBits = frac + maskBits, n = y.length;
  const Y = Uint32Array.from({ length: n }, (_, i) => toRing(y[i] * unit, ring));
  const R = Uint32Array.from({ length: n }, (_, i) => toRing(-r[i] * unit, ring));
  const S = uniformBits(n, outBits);
  const client = new Float64Array(n), server = new Float64Array(n);
  for (let at = 0; at < n; at += batch) {
    const end = Math.min(n, at + batch);
    const res = await mpc.secureReluBatch(Y.subarray(at, end), R.subarray(at, end),
      BigUint64Array.from(S.subarray(at, end), (v) => BigInt(v)), ringBits, outBits);
    addBatch(gc, res);
    res.out.forEach((o, i) => { client[at + i] = o / unit; server[at + i] = -S[at + i] / unit; });
    onBatch();
  }
  return { client, server };
}

/** The im2col packing `clientFHE.encryptedConv2d` uses, so the two paths compute the same convolution. */
function convLayout(image: number[][], w: CnnStudentWeights) {
  const p = w.conv.padding, kH = w.conv.kernel;
  const H = image.length + 2 * p, W = image[0].length + 2 * p;
  const src = Array.from({ length: H }, (_, y) => Array.from({ length: W }, (_, x) =>
    (y >= p && y < image.length + p && x >= p && x < image[0].length + p) ? image[y - p][x - p] : 0));
  const outH = H - kH + 1, outW = W - kH + 1, patch = kH * kH;
  return { src, outH, outW, kH, patch, block: 1 << Math.ceil(Math.log2(Math.max(patch, 2))), nPatches: outH * outW };
}

/** Sum over each block, in the clear: what the ciphertext's slots hold, for the statistical bound and the self-test. */
function reduceBlocks(v: Float64Array<ArrayBufferLike>, block: number): Float64Array<ArrayBufferLike> {
  let acc = v;
  for (let step = 1; step < block; step <<= 1) {
    const next = new Float64Array(acc.length);
    for (let i = 0; i < acc.length; i++) next[i] = acc[i] + acc[(i + step) % acc.length];
    acc = next;
  }
  return acc;
}

const sumAbs = (v: ArrayLike<number>): number => { let s = 0; for (let i = 0; i < v.length; i++) s += Math.abs(v[i]); return s; };
const relu = (v: number) => (v > 0 ? v : 0);

export async function hybridStudentInference(w: CnnStudentWeights, image: number[][],
                                             opts: { onProgress?: (p: HybridCnnProgress) => void } = {}): Promise<HybridCnnResult> {
  const t0 = performance.now();
  const s = await sealed();
  const mpc: Mpc = await import("./clientMPC");
  await mpc.warmMPC();
  const { slots } = s;
  const cfg = hybridCnnCfg();
  const say = (phase: HybridCnnProgress["phase"], done: number, total: number) => opts.onProgress?.({ phase, done, total });
  if (w.conv.in_ch !== 1) throw new Error(`the hybrid path takes one input channel, and this student has ${w.conv.in_ch}`);
  const kernels = w.conv.weight.map((oc) => oc[0]);
  const biases = w.conv.bias ?? new Array(w.conv.out_ch).fill(0);
  const { src, outH, outW, kH, block, nPatches } = convLayout(image, w);
  if (nPatches * block > slots) throw new Error(`the convolution needs ${nPatches * block} slots and the ring has ${slots}`);
  const widest = Math.max(...w.dense.map((d) => 1 << Math.ceil(Math.log2(Math.max(d.in, 2)))));
  const steps = [0];                                // 0 is the conjugation
  for (let step = 1; step < Math.max(block, widest); step <<= 1) steps.push(step);

  say("keys", 0, 1);
  const client = new Client(s, steps);
  const server = new Server(s, client.publicKey, client.galoisKeys);
  const gc: GcTotals = { and: 0, batches: 0, bytes: 0, messages: 0, compileMs: 0, mpcMs: 0 };
  let ckksBytes = 0, ckksMs = 0, statistical = 0, maxLayerError = 0, reluCount = 0;
  const clientView: number[] = [];
  const arena = new Arena();
  try {
    // ---- The convolution, under CKKS
    let t = performance.now();
    const packed = new Float64Array(slots);
    for (let oy = 0; oy < outH; oy++) for (let ox = 0; ox < outW; ox++) {
      const base = (oy * outW + ox) * block;
      for (let ky = 0; ky < kH; ky++) for (let kx = 0; kx < kH; kx++) packed[base + ky * kH + kx] = src[oy + ky][ox + kx];
    }
    const sent = client.encrypt(packed);
    ckksBytes += sent.length;
    const ctX = server.receive(sent, arena);
    const clearConv: Float64Array[] = [];
    const yConv: number[] = [], rConv: number[] = [];
    for (let c = 0; c < kernels.length; c++) {
      const tile = new Float64Array(slots);
      for (let i = 0; i < nPatches; i++) for (let ky = 0; ky < kH; ky++) for (let kx = 0; kx < kH; kx++) tile[i * block + ky * kH + kx] = kernels[c][ky][kx];
      const { bytes, mask } = server.evaluate(ctX, tile, block, nPatches, (i) => (i < nPatches ? biases[c] : 0), cfg.conv_mask_bits);
      ckksBytes += bytes.length;
      const decoded = client.decrypt(bytes);
      // The clear values, for the bound and the self-test only.
      const clear = reduceBlocks(packed.map((v, i) => v * tile[i]), block);
      for (let i = 0; i < nPatches; i++) clear[i * block] += biases[c];
      statistical += sumAbs(clear) / 2 ** (cfg.conv_mask_bits + 1);
      for (let i = 0; i < nPatches; i++) {
        yConv.push(decoded[i * block]); rConv.push(mask[i * block]); clientView.push(Math.abs(decoded[i * block]));
        maxLayerError = Math.max(maxLayerError, Math.abs(decoded[i * block] - mask[i * block] - clear[i * block]));
      }
      clearConv.push(clear);
      say("conv", c + 1, kernels.length);
    }
    ckksMs += performance.now() - t;

    // ---- Its ReLUs, in the garbled circuit
    if (w.conv.activation !== "relu") throw new Error(`the hybrid path expects a ReLU after the convolution, and the file says ${w.conv.activation}`);
    let total = yConv.length;
    reluCount += total;
    let done = 0;
    const conv = await reluLayer(mpc, yConv, rConv, cfg.conv_mask_bits, cfg.conv_relu_batch, gc, () => say("relu", (done += cfg.conv_relu_batch), total));
    for (let i = 0; i < total; i++) statistical += relu(clearConv[Math.floor(i / nPatches)][(i % nPatches) * block]) / 2 ** cfg.conv_mask_bits;

    // ---- The pool, on each party's own share
    const ps = w.pool.stride;
    const ph = Math.floor(outH / ps), pw = Math.floor(outW / ps);
    const pool = (share: Float64Array) => {
      const out: number[] = [];
      for (let c = 0; c < kernels.length; c++) for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++) {
        let a = 0;
        for (let dy = 0; dy < ps; dy++) for (let dx = 0; dx < ps; dx++) a += share[c * nPatches + (y * ps + dy) * outW + (x * ps + dx)];
        out.push(a / (ps * ps));
      }
      return Float64Array.from(out);
    };
    let clientShare: Float64Array<ArrayBufferLike> = pool(conv.client), serverShare: Float64Array<ArrayBufferLike> = pool(conv.server);
    let clearVec: Float64Array<ArrayBufferLike> = (() => {
      const flatRelu = new Float64Array(total);
      for (let i = 0; i < total; i++) flatRelu[i] = relu(clearConv[Math.floor(i / nPatches)][(i % nPatches) * block]);
      return pool(flatRelu);
    })();

    // ---- The dense layers
    let yLast: Float64Array<ArrayBufferLike> = new Float64Array(0), rLast: Float64Array<ArrayBufferLike> = new Float64Array(0);
    for (let k = 0; k < w.dense.length; k++) {
      const d = w.dense[k];
      if (clientShare.length !== d.in) throw new Error(`dense layer ${k} takes ${d.in} inputs and has ${clientShare.length}`);
      t = performance.now();
      const b = 1 << Math.ceil(Math.log2(Math.max(d.in, 2)));
      if (b * d.out > slots) throw new Error(`dense layer ${k} needs ${b * d.out} slots and the ring has ${slots}`);
      const lay = (v: ArrayLike<number>) => {
        const out = new Float64Array(slots);
        for (let j = 0; j < d.out; j++) for (let i = 0; i < d.in; i++) out[j * b + i] = v[i];
        return out;
      };
      const tile = new Float64Array(slots);
      for (let j = 0; j < d.out; j++) for (let i = 0; i < d.in; i++) tile[j * b + i] = d.weight[j][i] ?? 0;
      const up = client.encrypt(lay(clientShare));
      ckksBytes += up.length;
      const ct = server.addShare(server.receive(up, arena), lay(serverShare), arena);
      const last = k === w.dense.length - 1;
      const maskBits = last ? cfg.logit_mask_bits : cfg.dense_mask_bits;
      const { bytes, mask } = server.evaluate(ct, tile, b, d.out, (j) => d.bias?.[j] ?? 0, maskBits);
      ckksBytes += bytes.length;
      const decoded = client.decrypt(bytes);
      ckksMs += performance.now() - t;
      const clear = reduceBlocks(lay(clearVec).map((v, i) => v * tile[i]), b);
      for (let j = 0; j < d.out; j++) clear[j * b] += d.bias?.[j] ?? 0;
      statistical += sumAbs(clear) / 2 ** (maskBits + 1);
      const y = new Float64Array(d.out), r = new Float64Array(d.out);
      const out = new Float64Array(d.out);
      for (let j = 0; j < d.out; j++) {
        y[j] = decoded[j * b]; r[j] = mask[j * b]; out[j] = clear[j * b]; clientView.push(Math.abs(y[j]));
        maxLayerError = Math.max(maxLayerError, Math.abs(y[j] - r[j] - out[j]));
      }
      say("dense", k + 1, w.dense.length);
      if (last) { yLast = y; rLast = r; clearVec = out; break; }
      if (d.activation !== "relu") throw new Error(`the hybrid path expects a ReLU after dense layer ${k}, and the file says ${d.activation}`);
      reluCount += d.out;
      done = 0; total = d.out;
      const dense = await reluLayer(mpc, y, r, cfg.dense_mask_bits, d.out, gc, () => say("relu", (done += d.out), total));
      for (let j = 0; j < d.out; j++) statistical += relu(out[j]) / 2 ** cfg.dense_mask_bits;
      clientShare = dense.client; serverShare = dense.server;
      clearVec = out.map(relu);
    }

    // ---- The class: the existing argmax, on shares of the logits
    const n = yLast.length, ring = 2 ** 32;
    const a = Uint32Array.from({ length: n }, (_, k) => toRing((yLast[k] + mpc.OFFSET) * mpc.SCALE, ring));
    const b = Uint32Array.from({ length: n }, (_, k) => toRing(-rLast[k] * mpc.SCALE, ring));
    const arg = await mpc.secureArgmaxShares(a, b);
    gc.bytes += arg.bytes; gc.messages += arg.messages; gc.mpcMs += arg.mpcMs; gc.and += arg.andOps;
    say("argmax", 1, 1);

    const logits = Array.from(yLast, (v, k) => v - rLast[k]);
    let clearPrediction = 0;
    for (let k = 1; k < n; k++) if (clearVec[k] > clearVec[clearPrediction]) clearPrediction = k;
    return {
      prediction: arg.arg, clearPrediction, matchesClear: arg.arg === clearPrediction,
      logits, logitMaxAbsError: Math.max(...logits.map((v, k) => Math.abs(v - clearVec[k]))),
      reluCount, andGates: gc.and, batches: gc.batches, gcBytes: gc.bytes, gcMessages: gc.messages,
      gcCompileMs: gc.compileMs, gcMs: gc.mpcMs, ckksBytes, keyBytes: client.publicKey.length + client.galoisKeys.length, ckksMs,
      rotations: server.rotations, conjugations: server.conjugations, totalMs: performance.now() - t0,
      maskBits: { conv: cfg.conv_mask_bits, dense: cfg.dense_mask_bits, logit: cfg.logit_mask_bits },
      statisticalDistance: statistical, statisticalDistanceLog2: Math.log2(statistical), maxLayerError,
      clientViewMedianAbs: clientView.sort((x, y) => x - y)[clientView.length >> 1],
    };
  } finally {
    arena.close(); server.close(); client.close();
  }
}

/** In a browser, off the page's thread, as the satellite student's CKKS runs: a worker that fails is reported, never retried on the page's thread. */
export async function hybridStudentInferenceOffThread(w: CnnStudentWeights, image: number[][],
                                                      opts: { onProgress?: (p: HybridCnnProgress) => void } = {}): Promise<HybridCnnResult & { ranIn: string }> {
  if (typeof window === "undefined" || typeof Worker === "undefined") {
    return { ...(await hybridStudentInference(w, image, opts)), ranIn: "the calling thread" };
  }
  const worker = new Worker(new URL("./clientHybridCnn.worker.ts", import.meta.url), { type: "module" });
  try {
    return await new Promise((resolve, reject) => {
      worker.onmessage = (e: MessageEvent<{ progress?: HybridCnnProgress; ok?: boolean; result?: HybridCnnResult; error?: string }>) => {
        const d = e.data;
        if (d.progress) { opts.onProgress?.(d.progress); return; }
        if (d.ok && d.result) resolve({ ...d.result, ranIn: "a Web Worker" });
        else reject(new Error(`in its worker: ${d.error ?? "no result"}`));
      };
      worker.onerror = (e) => reject(new Error(`its worker failed: ${e.message || "the worker could not be loaded"}`));
      worker.onmessageerror = () => reject(new Error("its worker's result could not be read back"));
      worker.postMessage({ weights: w, image });
    });
  } finally {
    worker.terminate();
  }
}

/**
 * The fully encrypted medical student under CKKS, in its own context (Round 86, 2026-11-19).
 *
 * `medical_fl.square_student` is a convolution, a square, an average pool and a dense layer: multiplicative depth 3. The
 * page's own context (`clientFHE.ts`, 60+40+40+60 bits) allows two rescales, and the FHE.org poster describes that module,
 * so it is left as it is until the poster's notice on 2026-12-18. This module builds its own node-seal context for each
 * parameter set that passed the pre-registered fidelity criterion (`references/medical-square-student-preregistration.md`
 * and its addendum for A2), read from the published `weights.json`, which lists exactly those: A2, N = 8192 with
 * 48+34+34+34+52 bits at scale 2^34, and B, N = 16384 with 60+40+40+40+60 at 2^40. The first N = 8192 set (50+30+30+30+50
 * at 2^30) missed the 1e-3 limit on every test image and is not published as a set.
 *
 * The pass is the one the Python run measured. The client encrypts the image's nine convolution taps, each repeated for
 * the four channels (784 slots); the evaluator multiplies each by its kernel tap and sums them, adds the bias, squares and
 * relinearises, multiplies by each class's dense weights averaged over the pooling windows, sums the slots by rotation and
 * adds the class's bias; the client decrypts two logits. No layer is decrypted in between.
 *
 * What the client decrypts. Every vector here is tiled with period 1,024 across all N/2 slots (784 used, 240 zero), and
 * 1,024 divides N/2 in both sets. A rotate-and-add over steps 1 to 512 then sums exactly one period into EVERY slot, so
 * the decrypted vector holds the logit in every slot and no partial sum anywhere. A partial sum is what `clientFHE.ts`
 * masks and floods before decrypting (Round 84), because slot k minus slot k + 1 gives a product w_k x_k to anyone who
 * knows x; here the depth-3 chain has no level left for a mask, and the tiling removes the partial sums instead. What it
 * does not hide is each logit's own CKKS error, as on the rest of the page.
 */
// The SEAL wasm by its hashed asset URL, exactly as clientFHE.ts loads it (see the note there): Vite renames the file.
import sealWasmUrl from "/node_modules/node-seal/dist/seal_throws.wasm?url";

/** `weights.json` of `medical_fl.square_student`, as `scripts/train/medical_square_student.py` writes it. */
export interface SquareWeights {
  kind: "cnn-1conv-square-1dense";
  conv: { in_ch: number; out_ch: number; kernel: number; padding: number; activation: "square"; batch_norm: "folded";
          weight: number[][][][]; bias: number[] };
  pool: { type: "avg"; stride: number };
  dense: { in: number; out: number; weight: number[][]; bias: number[]; activation: "none" }[];
  classes: string[];
  n_parameters: number;
  /** The parameter sets that passed the pre-registered fidelity criterion, by name. */
  ckks: Record<string, { N: number; coeff_mod_bits: number[]; scale_bits: number }>;
}

export interface SquareRun {
  set: string;
  N: number;
  coeffModBits: number[];
  scaleBits: number;
  /** Decrypted logits, then the same network in float64 on the same input. */
  logits: [number, number];
  plain: [number, number];
  maxAbsError: number;
  decisionAgrees: boolean;
  /** The class index of the decrypted logits' argmax. */
  predicted: number;
  ms: { encrypt: number; evaluate: number; decrypt: number };
  bytes: { input: number; output: number };
}

const SIDE = 14;
const PIXELS = SIDE * SIDE;          // 196
const CHANNELS = 4;
const USED = CHANNELS * PIXELS;      // 784
const PERIOD = 1024;                 // a power of two >= USED that divides N/2 in every set
const ROTATIONS = Array.from({ length: Math.log2(PERIOD) }, (_, i) => 1 << i);   // 1, 2, ..., 512

/** The nine taps of a 3x3 convolution with padding 1, as Conv2d correlates: tap (di, dj) is the zero-padded image
 *  shifted by (di, dj), row-major over the 14 x 14 output. */
export function squareTaps(x: ArrayLike<number>): number[][] {
  if (x.length !== PIXELS) throw new Error(`the square student reads ${PIXELS} pixels, not ${x.length}`);
  const taps: number[][] = [];
  for (let di = 0; di < 3; di++) {
    for (let dj = 0; dj < 3; dj++) {
      const t = new Array<number>(PIXELS);
      for (let i = 0; i < SIDE; i++) {
        for (let j = 0; j < SIDE; j++) {
          const r = i + di - 1, c = j + dj - 1;
          t[i * SIDE + j] = r >= 0 && r < SIDE && c >= 0 && c < SIDE ? Number(x[r * SIDE + c]) : 0;
        }
      }
      taps.push(t);
    }
  }
  return taps;
}

function check(w: SquareWeights): void {
  if (w.kind !== "cnn-1conv-square-1dense" || w.conv.out_ch !== CHANNELS || w.conv.kernel !== 3 || w.conv.padding !== 1
      || w.pool.stride !== 2 || w.dense.length !== 1 || w.dense[0].in !== CHANNELS * 49 || w.dense[0].out !== 2) {
    throw new Error("these weights are not the square student's layout");
  }
}

/** The kernel tap k of channel c, read from the (4, 1, 3, 3) weight. */
const tapWeight = (w: SquareWeights, c: number, k: number) => w.conv.weight[c][0][Math.floor(k / 3)][k % 3];

/** The dense weights of class j over the slot layout c * 196 + p, averaged over each 2 x 2 pooling window. */
export function pooledDense(w: SquareWeights, j: number): number[] {
  check(w);
  const v = new Array<number>(USED).fill(0);
  for (let c = 0; c < CHANNELS; c++) {
    for (let pi = 0; pi < SIDE; pi++) {
      for (let pj = 0; pj < SIDE; pj++) {
        v[c * PIXELS + pi * SIDE + pj] = w.dense[0].weight[j][c * 49 + Math.floor(pi / 2) * 7 + Math.floor(pj / 2)] / 4;
      }
    }
  }
  return v;
}

/** The same network in float64, the reference every decrypted logit is compared with. */
export function plainSquareLogits(x: ArrayLike<number>, w: SquareWeights): [number, number] {
  check(w);
  const taps = squareTaps(x);
  const sq = new Array<number>(USED);
  for (let c = 0; c < CHANNELS; c++) {
    for (let p = 0; p < PIXELS; p++) {
      let s = w.conv.bias[c];
      for (let k = 0; k < 9; k++) s += tapWeight(w, c, k) * taps[k][p];
      sq[c * PIXELS + p] = s * s;
    }
  }
  const out: [number, number] = [0, 0];
  for (let j = 0; j < 2; j++) {
    const v = pooledDense(w, j);
    let s = w.dense[0].bias[j];
    for (let i = 0; i < USED; i++) s += v[i] * sq[i];
    out[j] = s;
  }
  return out;
}

/** `values` (at most PERIOD long) tiled with period PERIOD across `slots` slots. */
function tiled(values: ArrayLike<number>, slots: number): Float64Array {
  const out = new Float64Array(slots);
  for (let start = 0; start < slots; start += PERIOD) {
    for (let i = 0; i < values.length; i++) out[start + i] = Number(values[i]);
  }
  return out;
}

interface SquareCtx {
  seal: any; encoder: any; encryptor: any; decryptor: any; evaluator: any; relin: any; galois: any;
  slots: number; scale: number;
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

const _ctx = new Map<string, Promise<SquareCtx>>();
function squareContext(N: number, bits: number[], scaleBits: number): Promise<SquareCtx> {
  const key = `${N}:${bits.join("+")}:${scaleBits}`;
  let hit = _ctx.get(key);
  if (!hit) {
    hit = (async () => {
      const seal = await sealModule();
      if ((N / 2) % PERIOD !== 0) throw new Error(`N = ${N} has no whole number of ${PERIOD}-slot periods`);
      const parms = new seal.EncryptionParameters(seal.SchemeType.ckks);
      parms.setPolyModulusDegree(N);
      parms.setCoeffModulus(seal.CoeffModulus.Create(N, Int32Array.from(bits)));
      // tc128: SEAL refuses a modulus above its classical 128-bit bound; both sets are also within the post-quantum one.
      const context = new seal.SEALContext(parms, true, seal.SecLevelType.tc128);
      if (!context.parametersSet()) throw new Error(`SEAL rejected N = ${N} with ${bits.join("+")} bits`);
      const keyGen = new seal.KeyGenerator(context);
      return {
        seal,
        encoder: new seal.CKKSEncoder(context),
        encryptor: new seal.Encryptor(context, keyGen.createPublicKey()),
        decryptor: new seal.Decryptor(context, keyGen.secretKey()),
        evaluator: new seal.Evaluator(context),
        relin: keyGen.createRelinKeys(),
        galois: keyGen.createGaloisKeysWithSteps(Int32Array.from(ROTATIONS)),
        slots: N / 2,
        scale: Math.pow(2, scaleBits),
      };
    })();
    _ctx.set(key, hit);
  }
  return hit;
}

/** One image, end to end under CKKS with the named parameter set from the weights. */
export async function runSquareStudent(x: ArrayLike<number>, w: SquareWeights, set: string): Promise<SquareRun> {
  check(w);
  const p = w.ckks[set];
  if (!p) throw new Error(`the published weights offer no parameter set "${set}" (they offer ${Object.keys(w.ckks).join(", ")})`);
  const c = await squareContext(p.N, p.coeff_mod_bits, p.scale_bits);
  const { seal, encoder, encryptor, decryptor, evaluator, relin, galois, slots, scale } = c;
  const live: { delete(): void }[] = [];
  const pt = () => { const o = new seal.Plaintext(); live.push(o); return o; };
  const ct = () => { const o = new seal.Ciphertext(); live.push(o); return o; };
  const encodeAt = (values: Float64Array, s: number, like: any | null) => {
    const out = pt();
    encoder.encode(values, s, out);
    if (like) evaluator.plainModSwitchToInplace(out, like.parmsId());
    return out;
  };
  try {
    // The client: nine taps, each repeated for the four channels, tiled, encrypted under the public key.
    const t0 = performance.now();
    const taps = squareTaps(x);
    const enc = taps.map((tap) => {
      const rep = new Array<number>(USED);
      for (let ch = 0; ch < CHANNELS; ch++) for (let i = 0; i < PIXELS; i++) rep[ch * PIXELS + i] = tap[i];
      const out = ct();
      encryptor.encrypt(encodeAt(tiled(rep, slots), scale, null), out);
      return out;
    });
    const inputBytes = enc.reduce((n, e) => n + e.saveToArray(seal.ComprModeType.none).length, 0);
    const t1 = performance.now();

    // The evaluator: the convolution, the bias, the square, the pooled dense layer and its bias.
    let acc: any = null;
    for (let k = 0; k < 9; k++) {
      const wk = new Array<number>(USED);
      for (let ch = 0; ch < CHANNELS; ch++) for (let i = 0; i < PIXELS; i++) wk[ch * PIXELS + i] = tapWeight(w, ch, k);
      const prod = ct();
      evaluator.multiplyPlain(enc[k], encodeAt(tiled(wk, slots), scale, null), prod);
      if (acc === null) acc = prod;
      else evaluator.add(acc, prod, acc);
    }
    evaluator.rescaleToNextInplace(acc);
    const bias = new Array<number>(USED);
    for (let ch = 0; ch < CHANNELS; ch++) for (let i = 0; i < PIXELS; i++) bias[ch * PIXELS + i] = w.conv.bias[ch];
    evaluator.addPlain(acc, encodeAt(tiled(bias, slots), acc.scale(), acc), acc);
    const sq = ct();
    evaluator.square(acc, sq);
    evaluator.relinearizeInplace(sq, relin);
    evaluator.rescaleToNextInplace(sq);
    const outs = [0, 1].map((j) => {
      const t = ct();
      evaluator.multiplyPlain(sq, encodeAt(tiled(pooledDense(w, j), slots), scale, sq), t);
      evaluator.rescaleToNextInplace(t);
      const r = ct();
      for (const s of ROTATIONS) {
        evaluator.rotateVector(t, s, galois, r);
        evaluator.add(t, r, t);
      }
      evaluator.addPlain(t, encodeAt(new Float64Array(slots).fill(w.dense[0].bias[j]), t.scale(), t), t);
      return t;
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

    const ref = plainSquareLogits(x, w);
    const predicted = logits[1] > logits[0] ? 1 : 0;
    return {
      set, N: p.N, coeffModBits: p.coeff_mod_bits, scaleBits: p.scale_bits, logits, plain: ref,
      maxAbsError: Math.max(Math.abs(logits[0] - ref[0]), Math.abs(logits[1] - ref[1])),
      decisionAgrees: predicted === (ref[1] > ref[0] ? 1 : 0), predicted,
      ms: { encrypt: t1 - t0, evaluate: t2 - t1, decrypt: t3 - t2 },
      bytes: { input: inputBytes, output: outputBytes },
    };
  } finally {
    for (const o of live) o.delete();
  }
}

/** A run over several images, as the worker reports it back. */
export interface SquareBatch { runs: SquareRun[]; ranIn: string }

/**
 * The same runs off the page's thread. One image costs tens of milliseconds of CKKS under A2 and about twice that under B,
 * and the first one builds the keys, so a batch on the page's thread would hold input for seconds; like the satellite
 * student (`forecastStudent.ts`), this runs in a dedicated worker in a browser and on the calling thread elsewhere, and a
 * worker that fails is reported as the failure it is, never retried on the page's thread.
 */
export async function runSquareStudentOffThread(rows: number[][], w: SquareWeights, set: string,
                                                onProgress?: (done: number) => void): Promise<SquareBatch> {
  if (typeof window === "undefined" || typeof Worker === "undefined") {
    const runs: SquareRun[] = [];
    for (const x of rows) { runs.push(await runSquareStudent(x, w, set)); onProgress?.(runs.length); }
    return { runs, ranIn: "the calling thread" };
  }
  // Vite bundles the worker from exactly this form: `new URL(...)` written inside `new Worker(...)`.
  const worker = new Worker(new URL("./clientFHESquare.worker.ts", import.meta.url), { type: "module" });
  try {
    return await new Promise<SquareBatch>((resolve, reject) => {
      worker.onmessage = (e: MessageEvent<{ id: number; done?: number; ok?: boolean; runs?: SquareRun[]; error?: string }>) => {
        const d = e.data;
        if (d.done != null && d.ok == null) { onProgress?.(d.done); return; }
        if (d.ok && d.runs) resolve({ runs: d.runs, ranIn: "a Web Worker" });
        else reject(new Error(`in its worker: ${d.error ?? "no result"}`));
      };
      worker.onerror = (e) => reject(new Error(`its worker failed: ${e.message || "the worker could not be loaded"}`));
      worker.onmessageerror = () => reject(new Error("its worker's result could not be read back"));
      worker.postMessage({ id: 1, rows, weights: w, set });
    });
  } finally {
    worker.terminate();
  }
}

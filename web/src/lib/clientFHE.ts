// REAL in-browser CKKS homomorphic encryption via node-seal (MIT - Microsoft SEAL compiled to WebAssembly).
// This replaces the demo's old representative FHE cost model with a GENUINE encrypted linear layer
// computed entirely client-side: the feature vector is CKKS-encrypted, W.x runs homomorphically
// (multiply-plain, then a slot rotate-and-add for each dot product), the bias b is added to the
// ciphertext as a plaintext, and every slot that is not a layer output is zeroed by a 0/1 mask and
// flooded before the client decrypts (`decryptOutputs`), so the decryption yields W.x + b and nothing
// else. The demo's FHE-linear numbers are REAL MEASURED wall-clock, with ZERO server/VPS compute.
// node-seal is lazy-loaded (dynamic import, an async chunk; the WASM of a few MB is browser-cached).
// No commercial-OK WebGPU FHE exists in 2026, so CKKS runs on WASM-CPU.
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
import { fheCfg } from "./config";
import { randomWords } from "./csprng";
// Types only, erased at build: the figure's own schema, so the producer below is checked against the
// component that draws it rather than against a copy that could drift.
import type { EncryptedAgreementPlot } from "../components/VerificationChart";
import type { HeldOutCorpus } from "./realData";

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
  /** The 0/1 output masks `outputMask` has encoded, by layout. A layout is fixed per layer, so
   *  encoding it once per tab is enough; the flood beside it is drawn fresh for every decryption. */
  masks: Map<string, any>;
}

// The CKKS parameter set, from the config the backend also reads.
//
// These were two literals here, matching two more in `src/qsh/config/defaults.py`, a third pair of
// default arguments in `src/qsh/fhe/backends.py`, and a fourth copy written into
// `report/build_data.py`'s crypto table. `FHEConfig.poly_modulus_degree` was read by NONE of them,
// so the config field that names this parameter governed nothing and the four live copies agreed
// only because no one had changed one. Changing the ring degree is exactly the operation that turns
// that into a published PDF describing a parameter set the code does not run.
const POLY_MODULUS_DEGREE = fheCfg().poly_modulus_degree;
const COEFF_MOD_BITS = fheCfg().coeff_mod_bit_sizes;   // depth for one multiplication per dot product

//: Power-of-two rotation steps 1 .. 2^(GALOIS_STEP_BITS-1) are pre-generated. 2^10 = 1024 slots is
//: above any reduction width this demo evaluates (the widest is the medical student's first dense
//: layer, 196 inputs in a 256-slot block), and a request beyond it throws in SEAL rather than
//: degrading silently -- which is the intent.
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
  // argument generates the full default set for the configured ring -- key generation is the single
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
    scale: Math.pow(2, fheCfg().global_scale_bits),
    masks: new Map(),
  };
}

function getCtx(): Promise<SealCtx> {
  // A rejected init is not kept (Round 86, review PI-17): the promise was cached whether it resolved or
  // not, so one transient WASM-load failure disabled CKKS for the rest of the tab's life and every
  // later run reported `failed: "fhe"`. The next call retries.
  return (_ctx ??= init().catch((e) => { _ctx = null; throw e; }));
}

/**
 * Every value and the scale, finite, before they reach the encoder (Round 86, review PI-3). NaN encoded
 * to a silently wrong result that no precision check flagged (every such check compared with `>=`,
 * which NaN fails), and +-Infinity trapped node-seal's WASM. A non-finite input is refused here, by name.
 */
function encodeFinite(encoder: any, values: ArrayLike<number>, scale: number, pt: any, what: string): void {
  if (!Number.isFinite(scale)) throw new Error(`CKKS ${what}: scale ${scale} is not a finite number`);
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      throw new Error(`CKKS ${what}: value ${i} is ${values[i]}, not a finite number; CKKS cannot encode it`);
    }
  }
  encoder.encode(values, scale, pt);
}

/** Warm the CKKS context (keygen) so the first inference doesn't pay it. */
export async function warmFHE(): Promise<void> {
  await getCtx();
}

export interface FHEResult {
  logits: number[];        // decrypted W·x + b
  fheMs: number;           // REAL measured wall-clock of the encrypted computation
  ctBytes: number | null;   // null = the size could not be read; NEVER 0, which is impossible
  maxAbsError: number;     // CKKS approximation error vs the plaintext reference
  /** The plaintext reference itself, which this function already computes to get `maxAbsError`.
   *  Exposed in Round 34 so every caller can report the TOP-1 MISMATCH count - whether the encrypted
   *  argmax equals the plaintext one - which is the published convention for encrypted-vs-plaintext
   *  agreement (arXiv:2605.22237 Table IX reports it as an integer expected to be zero). Without it
   *  each caller would have to recompute W.x + b and could get it subtly wrong. */
  plainLogits: number[];
  /** Homomorphic rotations performed, counted. Each triggers a key-switch, so this is the number
   *  that makes a dense layer and a convolution comparable on a shared axis rather than on
   *  wall-clock alone.
   *
   *  Measured in Round 38: rotations are 57% of a small dense layer's wall-clock, 70% of the
   *  published student's convolution and 86% of its dense layers, at a per-rotation cost of
   *  3.10-4.39 ms across every stage measured. So key-switching does dominate, and the per-rotation figure is the
   *  comparable unit for the same reason.
   *
   *  An earlier revision of this comment said 10% for the convolution and that im2col encoding
   *  dominated it. That share divided the conv block's rotation time by the WHOLE student's
   *  wall-clock - a numerator and denominator of different scopes - and the claim is withdrawn. */
  rotations: number;
  multiplies: number;
  /** Wall-clock spent INSIDE `rotateVector`, nothing else. The page's `fhe_linear_ms` is this
   *  call's total multiplied by a cost-model scale for the selected model size, so dividing it by
   *  `rotations` does not give a per-rotation cost - it gave 158 ms against a true ~3 ms. This is
   *  the measured quantity, and it is the unit published CKKS primitive benchmarks report. */
  rotateMs: number;
}

export interface FHEBatchResult {
  /** `logits[r][j]` — decrypted score for row `r`, class `j`. */
  logits: number[][];
  plainLogits: number[][];
  rows: number;
  /** Rows the caller handed over. Larger than `rows` means the cap dropped the difference. */
  rowsOffered: number;
  /** Rows that fit in one ciphertext at this feature width. The batch is capped at
   *  `rowsPerCiphertext * maxCiphertexts`. */
  rowsPerCiphertext: number;
  /** Ciphertexts actually filled: 1 unless the caller raised `maxCiphertexts`, 0 for no rows. */
  ciphertexts: number;
  /** Wall-clock of the encrypted work only, summed over the ciphertexts. The yields between
   *  ciphertexts are outside it, so this is not inflated by the page repainting. */
  fheMs: number;
  /** `fheMs / ciphertexts`, null with no ciphertext. The unit that stays comparable when the row
   *  count grows: the total for 64 ciphertexts is not a slower CKKS, it is 64 of them. */
  msPerCiphertext: number | null;
  /** Size of ONE input ciphertext; every chunk is encrypted at the same ring and scale. */
  ctBytes: number | null;
  maxAbsError: number;
  rotations: number;
  multiplies: number;
  /** Wall-clock spent INSIDE `rotateVector`, nothing else. The page's `fhe_linear_ms` is this
   *  call's total multiplied by a cost-model scale for the selected model size, so dividing it by
   *  `rotations` does not give a per-rotation cost - it gave 158 ms against a true ~3 ms. This is
   *  the measured quantity, and it is the unit published CKKS primitive benchmarks report. */
  rotateMs: number;
}

/** How far a batch may go past one ciphertext. */
export interface BatchOptions {
  /** Ciphertexts this call may fill. 1, the default, is the original cap: every existing caller
   *  keeps the homomorphic work of a single encryption. A caller that must score a whole held-out
   *  split raises it, and the result reports `ciphertexts` and `msPerCiphertext` so the extra cost
   *  is a number on the page, not a factor hidden inside one total. */
  maxCiphertexts?: number;
}

/** Let the page repaint between ciphertexts. SEAL runs synchronously on the main thread, and 64
 *  back-to-back ciphertexts (finance's 8,192 rows) would freeze the tab for the whole batch. */
const yieldToPage = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

/**
 * Half-width, as a power of two, of the flood on every slot a layer does not output (Round 84).
 *
 * The 0/1 mask in `decryptOutputs` zeroes the rotation-sum's partial sums only up to the mask's own
 * CKKS encoding error, about 1e-11 of each sum, and that error is public: a key holder who encodes the
 * same mask knows it exactly and can divide it out. Measured on this tab's layouts by
 * `tests/test_r84_medtel_ckks_decrypts_only_outputs.py`: without the flood, that division recovers the
 * masked partial sums with a correlation of 1.0. The encoding error of a REAL vector is itself real
 * (SEAL rounds the conjugate-symmetric coefficients symmetrically), so the residual sits in the real
 * parts and the imaginary parts of the masked slots hold double-precision round-off only; a flood of
 * the real parts is therefore enough, and node-seal's real-valued encoder can place one. A uniform
 * flood of half-width 2^16 over a residual e is at statistical distance |e| / 2^17 from one with none:
 * below 1e-14 per slot on the test's layouts (residuals below 1e-9), and up to 1.25e-14 on the pages'
 * own data, whose largest residuals are 5.1e-10 (finance), 9.6e-10 (telecom), 1.28e-9 (medical) and
 * 1.64e-9 (genomics), measured by Round 86's review with a twin of this code on TenSEAL's SEAL, whose
 * encoder node-seal shares (PI-13; this said "below 1e-14" for all of them). The flood is not widened
 * to restore that figure, because it also has to stay inside the modulus: a coefficient is at most the
 * largest slot value times the scale, 2^16 x 2^120 = 2^136 at the masked scale, under the 2^140 of the
 * first three primes.
 */
export const CKKS_FLOOD_BITS = 16;

/** Uniform on [-2^bits, 2^bits), 53 random bits each, from the CSPRNG: the key holder must not
 *  predict them. The twin of `qsh.fhe.masking.uniform_masks`, which floods the server tier's view. */
function uniformFlood(n: number, bits: number): Float64Array {
  const w = randomWords(2 * n);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const u = (w[2 * i] * 2 ** 21 + (w[2 * i + 1] >>> 11)) / 2 ** 53;   // [0, 1) on a 2^-53 grid
    out[i] = (2 * u - 1) * 2 ** bits;
  }
  return out;
}

/** 1 at slots 0, stride, 2 stride, ... (`count` of them), 0 elsewhere: a layer's output slots. */
function outputMask(c: SealCtx, stride: number, count: number): any {
  const key = count === 1 ? "1" : `${stride}x${count}`;
  let pt = c.masks.get(key);
  if (!pt) {
    const m = new Float64Array(c.slots);
    for (let i = 0; i < count; i++) m[i * stride] = 1;
    pt = new c.seal.Plaintext();
    encodeFinite(c.encoder, m, c.scale, pt, "m");
    c.masks.set(key, pt);
  }
  return pt;
}

/**
 * Every SEAL object a computation allocates, freed when it ends (Round 85).
 *
 * node-seal's objects live in the WASM heap, which the JS collector neither sees nor frees: an object
 * nobody deletes stays allocated for the life of the tab. Only `encryptedDense` and `decryptOutputs`
 * deleted theirs, so each held-out evaluation left every ciphertext's input, weights, products and
 * rotations behind: about 330 MB per finance run, measured in node over six runs with the collector
 * forced between them, and the heap does not shrink. Started fourteen times with nothing else
 * changed, the finance page stopped responding in headless Chrome, its renderer past 5 GB (a desktop
 * Chrome walking the page's selections stopped at its twelfth run); a 32-bit WASM heap cannot grow
 * past 4 GiB. `drop` frees a spent rotation or accumulator at once, so the peak stays one layer's
 * worth, and `close`, in a `finally`, frees the rest even when SEAL throws.
 */
class SealScope {
  private readonly live = new Set<{ delete(): void }>();
  constructor(private readonly seal: any) {}
  pt(): any { return this.hold(new this.seal.Plaintext()); }
  ct(): any { return this.hold(new this.seal.Ciphertext()); }
  drop(o: { delete(): void }): void { if (this.live.delete(o)) o.delete(); }
  close(): void { for (const o of this.live) o.delete(); this.live.clear(); }
  private hold<T extends { delete(): void }>(o: T): T { this.live.add(o); return o; }
}

/**
 * Decrypt a layer's OUTPUTS and nothing else (Round 84), each with its bias added under encryption.
 *
 * Until this existed every path here decrypted the whole rotate-and-add result and added the bias in
 * plaintext afterwards. Two things followed. The bias was not part of the encrypted computation at
 * all. And the decrypted vector held the reduction's running partial sums in every other slot: slot k
 * minus slot k + 1 is the k-th product w_k x_k, so a key holder who knows x reads the weights off one
 * query. This is the browser twin of `qsh.fhe.masking.masked_decrypt` on the server tier, with one
 * difference that matters: the client decrypts the outputs IN THE CLEAR here, because ReLU, pooling
 * and the argmax run in plaintext in this tab, so the outputs are not masked by an MPC share.
 *
 *   1. multiply by a 0/1 mask, 1 at each output slot (`outputMask`): every other slot drops to its
 *      encoding residual, about 1e-11 of its partial sum;
 *   2. add ONE ciphertext, the public-key encryption of a plaintext that carries the bias at each
 *      output slot and an independent uniform flood of half-width 2^CKKS_FLOOD_BITS at every other
 *      slot, drawn fresh per call (see the constant); encrypted, so the result is re-randomised;
 *   3. decrypt, and read the output slots.
 *
 * What it does not hide: each output's own CKKS approximation error, of order 1e-9 to 1e-8, which
 * this tab does not flood (flooding the outputs would cost them their precision; the server tier masks
 * the outputs themselves). No rescale: the product sits at scale 2^120 inside the 140-bit modulus of
 * the first three primes, which keeps the outputs' precision where it was.
 */
function decryptOutputs(c: SealCtx, acc: any, stride: number, count: number,
                        bias: (i: number) => number): number[] {
  const { seal, encoder, encryptor, decryptor, evaluator, slots } = c;
  // Freed when this returns: three full-ring objects per output, and the WASM heap is not a pressure
  // the JS collector sees. "node-seal's finalizers would free them eventually" stood here until
  // Round 85; nothing frees them but `delete` (see `SealScope`).
  const sc = new SealScope(seal);
  try {
    const masked = sc.ct();
    evaluator.multiplyPlain(acc, outputMask(c, stride, count), masked);
    const add = uniformFlood(slots, CKKS_FLOOD_BITS);
    for (let i = 0; i < count; i++) add[i * stride] = bias(i);
    const ptAdd = sc.pt();
    encodeFinite(encoder, add, masked.scale(), ptAdd, "add");
    // Added ENCRYPTED, under the public key (Round 86, review PI-5): added as a plaintext, it left the
    // decrypted ciphertext with no fresh encryption randomness, so its noise was a deterministic
    // function of the computation. The server's `add_plain_slots` encrypts its masks for the same
    // reason. Same level and scale: nothing has been rescaled yet.
    const ctAdd = sc.ct();
    encryptor.encrypt(ptAdd, ctAdd);
    const flooded = sc.ct();
    evaluator.add(masked, ctAdd, flooded);
    const out = sc.pt();
    decryptor.decrypt(flooded, out);
    const dec = encoder.decodeFloat64(out);
    return Array.from({ length: count }, (_, i) => dec[i * stride]);
  } finally {
    sc.close();
  }
}

/** One ciphertext's worth of the stride-packed linear layer. `rows.length` is at most
 *  `slots / sumWidth`; the caller chunks. Returns per-class decrypted scores and its own counters. */
async function batchCiphertext(
  c: SealCtx, W: number[][], b: number[], rows: number[][], n: number, sumWidth: number,
) {
  const { seal, encoder, encryptor, evaluator, galois, slots, scale } = c;
  const t0 = performance.now();
  const xPad = new Float64Array(slots);
  rows.forEach((row, r) => {
    for (let i = 0; i < n; i++) xPad[r * sumWidth + i] = row[i] ?? 0;
  });
  const sc = new SealScope(seal);                    // finance fills 64 of these a run (`SealScope`)
  try {
    const ptX = sc.pt();
    encodeFinite(encoder, xPad, scale, ptX, "xPad");
    const ctX = sc.ct();
    encryptor.encrypt(ptX, ctX);
    let ctBytes: number | null = null;
    try { ctBytes = ctX.saveToArray(seal.ComprModeType.none).length; } catch { ctBytes = null; }

    const perClass: number[][] = [];
    let rotations = 0, multiplies = 0, rotateMs = 0;   // see `encryptedLinear` for why rotations are timed
    for (let j = 0; j < W.length; j++) {
      if (W[j].every((v) => v === 0)) {
        perClass.push(rows.map(() => b[j] ?? 0));      // all-zero row: the logit is the bias
        continue;
      }
      const wPad = new Float64Array(slots);
      for (let r = 0; r < rows.length; r++) {
        for (let i = 0; i < n; i++) wPad[r * sumWidth + i] = W[j][i] ?? 0;   // TILED at the same stride
      }
      const ptW = sc.pt();
      encodeFinite(encoder, wPad, scale, ptW, "wPad");
      let acc = sc.ct();
      evaluator.multiplyPlain(ctX, ptW, acc);
      multiplies++;
      for (let s = 1; s < sumWidth; s <<= 1) {
        const rot = sc.ct();
        const tr = performance.now();
        evaluator.rotateVector(acc, s, galois, rot);
        rotateMs += performance.now() - tr;
        const nxt = sc.ct();
        evaluator.add(acc, rot, nxt);
        sc.drop(rot); sc.drop(acc);
        acc = nxt;
        rotations++;
      }
      // The class's bias goes in under encryption, and only the rows' output slots decrypt.
      perClass.push(decryptOutputs(c, acc, sumWidth, rows.length, () => b[j] ?? 0));
      multiplies++;                                    // the output mask
      sc.drop(acc); sc.drop(ptW);
    }
    return {
      logits: rows.map((_, r) => perClass.map((cls) => cls[r])),
      ms: performance.now() - t0, ctBytes, rotations, multiplies, rotateMs,
    };
  } finally {
    sc.close();
  }
}

/**
 * The same linear layer over MANY rows, in the same number of homomorphic operations as one row.
 *
 * WHY THIS EXISTS. `encryptedLinear` below pads one input of 7-8 features into a `slots`-wide vector -
 * `slots` is ring/2, so most of them - and zeroes the rest. Every encrypted-versus-plaintext
 * figure this project reports was therefore a comparison over ONE row: `encrypted_scores_compared`
 * counted CLASSES, not rows, and `encrypted_top1_mismatches` was `0 or 1`. A single agreement is not
 * evidence that encrypted inference reproduces plaintext, and it cannot be plotted against a test set.
 *
 * HOW IT WORKS, AND WHY IT COSTS NOTHING EXTRA. The rotate-and-add reduction sums a sliding window of
 * `sumWidth` slots, so slot `k` ends up holding `sum(slots k..k+sumWidth-1)`. Packing row `r` at slot
 * offset `r*sumWidth`, with the weight row TILED at the same stride, therefore leaves row `r`'s dot
 * product sitting in slot `r*sumWidth` - **with the homomorphic loop unchanged**. One encrypt, one
 * multiply-plain and log2(sumWidth) rotations per class serve every row in the batch. This is
 * ordinary CKKS SIMD batching, and it is what the convolution path here already does with
 * `patchesPerCiphertext`; the linear path simply never did it.
 *
 * Verified before it was written: the stride-packed reduction recovers each row's dot product exactly,
 * checked against the plaintext value in plain JS with the same rotation semantics.
 *
 * MORE THAN ONE CIPHERTEXT ONLY ON REQUEST (Round 84). A held-out split can outgrow one ciphertext:
 * finance's 8,192 test-window rows at width 32 need 64. `maxCiphertexts` lets a caller fill that
 * many, one after another with the same keys, and the result says how many it filled and what each
 * cost. The default stays 1, so no existing caller's homomorphic work changes.
 */
export async function encryptedLinearBatch(
  W: number[][], b: number[], rows: number[][], opts: BatchOptions = {},
): Promise<FHEBatchResult> {
  const c = await getCtx();
  const n = rows[0]?.length ?? 0;
  const sumWidth = 1 << Math.ceil(Math.log2(Math.max(n, 2)));
  const rowsPerCiphertext = Math.max(1, Math.floor(c.slots / sumWidth));
  const maxCiphertexts = Math.max(1, Math.floor(opts.maxCiphertexts ?? 1));
  // Capped rather than chunked silently. Filling more ciphertexts multiplies the homomorphic cost, so
  // it happens only when the caller asks, and `ciphertexts` / `msPerCiphertext` carry the multiple.
  //
  // `rowsOffered` is returned BECAUSE the cap is silent otherwise. Written first without it, and the
  // first run scored 512 of a 600-row held-out split: 88 rows dropped, an agreement of 1.0 reported
  // over the 512, and a caption reading "one of 512 held-out rows" that a reader takes for the whole
  // split. The comment here even claimed "a cap is visible, and the caller reports how many rows it
  // asked for" - it was not, and the caller could not, because this function did not tell it.
  const rowsOffered = rows.length;
  const used = rows.slice(0, rowsPerCiphertext * maxCiphertexts);

  const logits: number[][] = [];
  let fheMs = 0, rotations = 0, multiplies = 0, rotateMs = 0, ciphertexts = 0;
  let ctBytes: number | null = null;
  for (let start = 0; start < used.length; start += rowsPerCiphertext) {
    if (ciphertexts > 0) await yieldToPage();          // outside the timed region, see `fheMs`
    const part = await batchCiphertext(c, W, b, used.slice(start, start + rowsPerCiphertext), n, sumWidth);
    for (const row of part.logits) logits.push(row);
    fheMs += part.ms; rotations += part.rotations; multiplies += part.multiplies;
    rotateMs += part.rotateMs;
    ctBytes ??= part.ctBytes;
    ciphertexts++;
  }

  const plainLogits = used.map((row) =>
    W.map((wr, j) => wr.reduce((a, w, i) => a + w * (row[i] ?? 0), 0) + (b[j] ?? 0)));
  // A loop, not `Math.max(...spread)`: 8,192 rows x 2 classes is past what an argument list holds
  // in some engines, and a RangeError here would read as a CKKS failure.
  let maxAbsError = 0;
  logits.forEach((row, r) => row.forEach((v, j) => {
    maxAbsError = Math.max(maxAbsError, Math.abs(v - plainLogits[r][j]));
  }));
  return { logits, plainLogits, rows: used.length, rowsOffered, rowsPerCiphertext, ciphertexts,
           fheMs, msPerCiphertext: ciphertexts > 0 ? fheMs / ciphertexts : null, ctBytes,
           maxAbsError, rotations, multiplies, rotateMs };
}

/** Plaintext top-1 minus top-2: how far a row is from changing class. 0 is a tie. */
function topTwoGap(row: number[]): number {
  let a = -Infinity, b = -Infinity;
  for (const v of row) { if (v > a) { b = a; a = v; } else if (v > b) b = v; }
  return Number.isFinite(b) ? a - b : Infinity;
}

/** -log2(error) floored: the bits OpenFHE and HElib report. No bits exist for an error >= 1. */
const bitsOf = (e: number | null) => (e != null && e > 0 && e < 1 ? Math.floor(-Math.log2(e)) : null);

/** Everything a run publishes about one held-out encrypted evaluation, and the figure itself. */
export interface HeldOutAgreement {
  plot: EncryptedAgreementPlot;
  batch: FHEBatchResult;
  mismatches: number;
  /** Rows the plaintext model scores identically across its top two classes. */
  tieRows: number;
  decidable: number;
  flippedDecidable: number;
  flipMarginMax: number;
  /** Smallest non-zero plaintext top-1 margin: the closest any decided row came to a flip. */
  smallestMargin: number | null;
  /** Median |encrypted - plaintext| over the scores that went through CKKS (not the bias-only class). */
  medianAbsError: number | null;
  precisionBits: number | null;
  medianPrecisionBits: number | null;
  /** True only when no decided row COULD have flipped: smallest margin above the worst deviation,
   *  times two when every class score is encrypted (two deviations can meet), once when only one is. */
  flipImpossible: boolean;
  /** Against the TRUE labels, over the scored rows. */
  encryptedAccuracy: number | null;
  plaintextAccuracy: number | null;
  /** The share of the most common true label over the same rows: what always calling it would score,
   *  printed beside the accuracy (Round 86, review PI-12; finance's 0.9695 sat beside a 0.98 majority). */
  majorityBaseline: number | null;
}

/**
 * Score a published model on its held-out rows under CKKS, every row, and build the figure (Round 84).
 *
 * This is the path the finance, medical and telecom pages encrypt through. It fills as many
 * ciphertexts as the rows need - 64 for finance's 8,192 at width 32 - because a denominator smaller
 * than the split it names was the defect the cap disclosure exists for; the cost travels with it as
 * `ciphertexts` and `msPerCiphertext`.
 *
 * WHAT IS PLOTTED. A two-class model plots its decision logit (class 1 minus class 0), encrypted
 * against plaintext, so every row sits somewhere informative and a flip is a point in the wrong
 * quadrant. Plotting the winning class's score instead put every row the model calls class 0 at
 * exactly (0, 0) for the bias-only class-0 row, which is most of the finance page. A multiclass model
 * plots the encrypted top-1 class's score, the convention the figure already used.
 */
export async function encryptedHeldOutAgreement(held: HeldOutCorpus): Promise<HeldOutAgreement> {
  const c = await getCtx();
  const n = held.rows[0]?.length ?? 0;
  const sumWidth = 1 << Math.ceil(Math.log2(Math.max(n, 2)));
  const perCt = Math.max(1, Math.floor(c.slots / sumWidth));
  const { W, b } = held.model;
  const batch = await encryptedLinearBatch(W, b, held.rows, {
    maxCiphertexts: Math.ceil(held.rows.length / perCt),
  });
  const argmaxOf = (r: number[]) => r.reduce((bi, v, i, arr) => (v > arr[bi] ? i : bi), 0);
  const encCls = batch.logits.map(argmaxOf), plainCls = batch.plainLogits.map(argmaxOf);
  const margins = batch.plainLogits.map(topTwoGap);
  const flipped = encCls.map((_, i) => i).filter((i) => encCls[i] !== plainCls[i]);
  const tieRows = margins.filter((m) => m === 0).length;
  const decidable = batch.rows - tieRows;
  const flippedDecidable = flipped.filter((i) => margins[i] !== 0).length;
  let flipMarginMax = 0;
  for (const i of flipped) flipMarginMax = Math.max(flipMarginMax, margins[i]);
  let smallestMargin: number | null = null;
  for (const m of margins) if (m > 0 && (smallestMargin == null || m < smallestMargin)) smallestMargin = m;

  // The deviations of the scores that were ENCRYPTED. A bias-only class (finance's and medical's
  // class 0) is exact by construction, and counting its zeros would halve the median.
  const noisy = W.map((row, j) => (row.some((v) => v !== 0) ? j : -1)).filter((j) => j >= 0);
  const devs: number[] = [];
  batch.logits.forEach((row, r) => noisy.forEach((j) => devs.push(Math.abs(row[j] - batch.plainLogits[r][j]))));
  devs.sort((x, y) => x - y);
  const medianAbsError = devs.length
    ? (devs.length % 2 ? devs[(devs.length - 1) / 2] : (devs[devs.length / 2 - 1] + devs[devs.length / 2]) / 2)
    : null;
  const flipBoundFactor: 1 | 2 = noisy.length <= 1 ? 1 : 2;
  const flipImpossible = smallestMargin != null && smallestMargin > flipBoundFactor * batch.maxAbsError;

  const labels = held.labels.slice(0, batch.rows);
  const acc = (cls: number[]) => (labels.length
    ? cls.reduce((a, v, i) => a + (v === labels[i] ? 1 : 0), 0) / labels.length : null);
  const counts = new Map<number, number>();
  for (const l of labels) counts.set(l, (counts.get(l) ?? 0) + 1);
  const majorityBaseline = labels.length ? Math.max(...counts.values()) / labels.length : null;
  const binary = W.length === 2;
  const pts = held.plane?.points;
  const plot: EncryptedAgreementPlot = {
    kind: "encrypted-agreement",
    points: batch.logits.map((row, i) => ({
      enc: binary ? row[1] - row[0] : row[encCls[i]],
      plain: binary ? batch.plainLogits[i][1] - batch.plainLogits[i][0] : batch.plainLogits[i][encCls[i]],
      agree: encCls[i] === plainCls[i],
      label: labels[i], encCls: encCls[i], plainCls: plainCls[i],
      ...(pts?.[i] ? { x: pts[i][0], y: pts[i][1] } : {}),
    })),
    rows: batch.rows, rowsOffered: batch.rowsOffered, classes: W.length,
    maxAbsError: batch.maxAbsError, flipMarginMax,
    undecidableRows: tieRows, decidableRows: decidable,
    agreementOverDecidable: decidable > 0 ? 1 - flippedDecidable / decidable : null,
    heldOut: true, fitNote: held.fitNote, split: held.split, corpus: held.corpus,
    modelKey: held.modelKey, modelLabel: held.modelLabel, licence: held.licence,
    ciphertexts: batch.ciphertexts, rowsPerCiphertext: batch.rowsPerCiphertext, fheMs: batch.fheMs,
    msPerCiphertext: batch.msPerCiphertext, medianAbsError, noisyScores: devs.length,
    smallestMargin, flipBoundFactor, scoreKind: binary ? "logit" : "winning-score",
    classNames: held.classNames,
    plane: held.plane ? { x: held.plane.x, y: held.plane.y } : null,
    accuracy: { encrypted: acc(encCls), plaintext: acc(plainCls), majority: majorityBaseline },
  };
  return {
    plot, batch, mismatches: flipped.length, tieRows, decidable, flippedDecidable, flipMarginMax,
    smallestMargin, medianAbsError, precisionBits: bitsOf(batch.maxAbsError),
    medianPrecisionBits: bitsOf(medianAbsError), flipImpossible,
    encryptedAccuracy: acc(encCls), plaintextAccuracy: acc(plainCls), majorityBaseline,
  };
}

/**
 * Genuinely encrypt ``x`` and evaluate the linear layer ``W·x + b`` under CKKS, returning the decrypted
 * logits + the REAL measured time + the ciphertext size. ``W`` is ``(out, in)``. The dot product is a
 * homomorphic multiply-plain followed by a rotate-and-add slot sum (the standard packed-CKKS inner
 * product). Run on a small model — CKKS on WASM-CPU is ~hundreds of ms.
 */
export async function encryptedLinear(W: number[][], b: number[], x: number[]): Promise<FHEResult> {
  const c = await getCtx();
  const { seal, encoder, encryptor, evaluator, galois, slots, scale } = c;
  const n = x.length;
  const sumWidth = 1 << Math.ceil(Math.log2(Math.max(n, 2)));   // power-of-2 ≥ n for the slot reduction

  const t0 = performance.now();
  const xPad = Float64Array.from({ length: slots }, (_, i) => (i < n ? x[i] : 0));
  const sc = new SealScope(seal);
  const logits: number[] = [];
  let ctBytes: number | null = null;
  let rotations = 0, multiplies = 0, rotateMs = 0;
  try {
    const ptX = sc.pt();
    encodeFinite(encoder, xPad, scale, ptX, "xPad");
    const ctX = sc.ct();
    encryptor.encrypt(ptX, ctX);
    // serialized ciphertext size (node-seal v7: saveToArray(ComprModeType)) → genuine ct/pt ratio.
    //
    // null on failure, NOT 0. A ciphertext is always larger than its plaintext, so a ct/pt ratio of 0
    // is not a small number - it is an impossible one, and it was reaching the benchmark as a MEASURED
    // value: clientCompute divides ctBytes by featDim*8, so a swallowed exception here printed
    // "ct/pt 0" in a column whose other rows are real measurements.
    try { ctBytes = ctX.saveToArray(seal.ComprModeType.none).length; } catch { ctBytes = null; }

    // `rotateMs` times ONLY the `rotateVector` calls, and it exists because the number beside it does
    // not support the division a reader will try. `fhe_linear_ms` on the page is this function's
    // wall-clock MULTIPLIED by `benchScale(...)`, a cost-model projection for the selected model size,
    // while `fhe_dense_rotations` is a raw count from this loop. Dividing one by the other read
    // 158 ms per rotation on genomics against a true figure near 3 ms. Rotation cost is the unit the
    // CKKS literature reports, because each rotation triggers a key-switch and key-switching does
    // dominate here - measured at 57% of a small dense layer, 70% of the student's convolution and
    // 86% of its dense layers. It has to be measured rather than reconstructed from a scaled total
    // downstream, and the denominator has to have the same scope as the numerator: dividing the
    // conv block's rotation time by the whole student's wall-clock produced a bogus 10% figure in
    // this round's first attempt.
    for (let j = 0; j < W.length; j++) {
      // an all-zero weight row (our 2-class "class 0" convention) → logit is just the bias; skip the
      // homomorphic eval (CKKS would yield a "transparent" ciphertext, which the safe build rejects).
      if (W[j].every((v) => v === 0)) {
        logits.push(b[j] ?? 0);
        continue;
      }
      const wPad = Float64Array.from({ length: slots }, (_, i) => (i < n ? (W[j][i] ?? 0) : 0));
      const ptW = sc.pt();
      encodeFinite(encoder, wPad, scale, ptW, "wPad");
      let acc = sc.ct();
      evaluator.multiplyPlain(ctX, ptW, acc);                     // encrypted element-wise W_j ⊙ x
      multiplies++;
      for (let s = 1; s < sumWidth; s <<= 1) {                    // rotate-and-add → Σ slots in slot 0
        const rot = sc.ct();
        const tr = performance.now();
        evaluator.rotateVector(acc, s, galois, rot);
        rotateMs += performance.now() - tr;
        const nxt = sc.ct();
        evaluator.add(acc, rot, nxt);
        sc.drop(rot); sc.drop(acc);
        acc = nxt;
        rotations++;
      }
      // b_j goes in under encryption, and only slot 0, the dot product, decrypts.
      logits.push(decryptOutputs(c, acc, sumWidth, 1, () => b[j] ?? 0)[0]);
      multiplies++;                                               // the output mask
      sc.drop(acc); sc.drop(ptW);
    }
  } finally {
    sc.close();
  }
  const fheMs = performance.now() - t0;

  // approximation error vs the plaintext reference (CKKS is approximate)
  const plain = W.map((row, j) => row.reduce((a, w, i) => a + w * (x[i] ?? 0), 0) + (b[j] ?? 0));
  const maxAbsError = Math.max(...logits.map((v, j) => Math.abs(v - plain[j])), 0);
  return { logits, fheMs, ctBytes, maxAbsError, plainLogits: plain, rotations, multiplies, rotateMs };
}

/**
 * A dense layer with every output row in ONE ciphertext (Round 84): the dual of
 * `encryptedLinearBatch`. There, rows of inputs share one weight row; here one input is replicated
 * under every weight row, each copy in its own power-of-two block, so a single multiply-plain and
 * log2(block) rotations put every row's dot product in its block's first slot, and one masked
 * decryption reads them all. `encryptedLinear` spends one multiply, log2(block) rotations and one
 * decryption PER ROW instead.
 *
 * On the medical student that is 8 rotations for its 196 -> 16 layer (blocks of 256, sixteen of them
 * filling the 4,096 slots exactly) instead of 128, and 4 for its 16 -> 2 layer instead of 8: 12 in all
 * where the row-at-a-time form took 136. The client encrypts the replicated copy itself; the server's
 * plaintext is the tiled weight matrix. Rows beyond one ciphertext go into further ciphertexts.
 */
export async function encryptedDense(W: number[][], b: number[], x: number[]): Promise<FHEResult> {
  const c = await getCtx();
  const { seal, encoder, encryptor, evaluator, galois, slots, scale } = c;
  const n = x.length;
  const block = 1 << Math.ceil(Math.log2(Math.max(n, 2)));
  const rowsPerCt = Math.max(1, Math.floor(slots / block));
  if (block > slots) throw new Error(`a ${n}-input layer needs a ${block}-slot block; the ring has ${slots}`);

  const t0 = performance.now();
  const logits = new Array<number>(W.length).fill(0);
  let rotations = 0, multiplies = 0, rotateMs = 0;
  let ctBytes: number | null = null;
  const sc = new SealScope(seal);                    // its eager frees, and a throw leaks nothing
  try {
    for (let start = 0; start < W.length; start += rowsPerCt) {
      const rows = W.slice(start, start + rowsPerCt);
      // A group of all-zero rows: each output is its bias, and the multiply would yield a
      // "transparent" ciphertext, which the safe build rejects (the `encryptedLinear` convention).
      if (rows.every((row) => row.every((v) => v === 0))) {
        rows.forEach((_, j) => { logits[start + j] = b[start + j] ?? 0; });
        continue;
      }
      const xRep = new Float64Array(slots), wTile = new Float64Array(slots);
      rows.forEach((row, j) => {
        for (let i = 0; i < n; i++) { xRep[j * block + i] = x[i] ?? 0; wTile[j * block + i] = row[i] ?? 0; }
      });
      const ptX = sc.pt();
      encodeFinite(encoder, xRep, scale, ptX, "xRep");
      const ctX = sc.ct();
      encryptor.encrypt(ptX, ctX);
      if (ctBytes == null) { try { ctBytes = ctX.saveToArray(seal.ComprModeType.none).length; } catch { ctBytes = null; } }
      const ptW = sc.pt();
      encodeFinite(encoder, wTile, scale, ptW, "wTile");
      let acc = sc.ct();
      evaluator.multiplyPlain(ctX, ptW, acc);
      multiplies++;
      for (let s = 1; s < block; s <<= 1) {          // the reduction stays inside each row's block
        const rot = sc.ct();
        const tr = performance.now();
        evaluator.rotateVector(acc, s, galois, rot);
        rotateMs += performance.now() - tr;
        const nxt = sc.ct();
        evaluator.add(acc, rot, nxt);
        sc.drop(rot); sc.drop(acc);                   // spent: freed now, as in `decryptOutputs`
        acc = nxt;
        rotations++;
      }
      decryptOutputs(c, acc, block, rows.length, (j) => b[start + j] ?? 0)
        .forEach((v, j) => { logits[start + j] = v; });
      multiplies++;                                    // the output mask
      sc.drop(acc); sc.drop(ptW); sc.drop(ctX); sc.drop(ptX);
    }
  } finally {
    sc.close();
  }
  const fheMs = performance.now() - t0;
  const plain = W.map((row, j) => row.reduce((a, w, i) => a + w * (x[i] ?? 0), 0) + (b[j] ?? 0));
  const maxAbsError = Math.max(...logits.map((v, j) => Math.abs(v - plain[j])), 0);
  return { logits, fheMs, ctBytes, maxAbsError, plainLogits: plain, rotations, multiplies, rotateMs };
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
  ctBytes: number | null;   // null = the size could not be read; NEVER 0, which is impossible
  maxAbsError: number;
  /** Homomorphic rotations performed. Each one triggers a key-switch, and that is the dominant
   *  term here as elsewhere: measured at 70% of this operation's wall-clock. */
  rotations: number;
  /** Ciphertext-plaintext multiplications performed: one per kernel and, since Round 84, one per
   *  output mask (`decryptOutputs`), so twice the channel count. */
  multiplies: number;
  /** How many im2col patches shared one ciphertext. 1 would mean no packing at all. */
  patchesPerCiphertext: number;
  /** Time inside `rotateVector` only. Measured here as well as on the dense path because the
   *  dense path's comment - key-switching dominates CKKS cost - is a claim about THIS operation
   *  too, and the convolution is the one place it can be checked against a different work mix:
   *  4 channels x 196 patches of im2col encoding around the same 16 rotations. Measured: 70%, so
   *  the claim holds here too. It only appeared not to when the share was computed against the
   *  whole student's wall-clock instead of the convolution's. */
  rotateMs: number;
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
  const { seal, encoder, encryptor, evaluator, galois, slots, scale } = c;

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
  const sc = new SealScope(seal);
  let ctBytes: number | null = null;
  let rotations = 0, multiplies = 0, rotateMs = 0;
  const maps: number[][] = [];
  try {
    const ptX = sc.pt();
    encodeFinite(encoder, packed, scale, ptX, "packed");
    const ctX = sc.ct();
    encryptor.encrypt(ptX, ctX);
    // null, not 0 - see the note on the linear path above: a ct/pt ratio of 0 is impossible, and it
    // was being reported as measured.
    try { ctBytes = ctX.saveToArray(seal.ComprModeType.none).length; } catch { ctBytes = null; }

    for (let oc = 0; oc < kernels.length; oc++) {
      // The same kernel tiled into every patch block, so one multiply-plain covers all positions.
      const tiled = new Float64Array(slots);
      for (let p = 0; p < nPatches; p++) {
        for (let ky = 0; ky < kH; ky++) {
          for (let kx = 0; kx < kW; kx++) tiled[p * block + ky * kW + kx] = kernels[oc][ky][kx];
        }
      }
      const ptW = sc.pt();
      encodeFinite(encoder, tiled, scale, ptW, "tiled");
      let acc = sc.ct();
      evaluator.multiplyPlain(ctX, ptW, acc);
      multiplies++;
      // Reduce WITHIN each block. Only log2(block) rotations, independent of how many patches ride
      // along - which is exactly what the packing buys.
      for (let s = 1; s < block; s <<= 1) {
        const rot = sc.ct();
        const tr = performance.now();
        evaluator.rotateVector(acc, s, galois, rot);
        rotateMs += performance.now() - tr;
        const nxt = sc.ct();
        evaluator.add(acc, rot, nxt);
        sc.drop(rot); sc.drop(acc);
        acc = nxt;
        rotations++;
      }
      // The channel's bias goes in under encryption, and only the nPatches output slots decrypt: the
      // other 15 of every 16 slots held the patch's running partial sums, kernel weight by weight.
      maps.push(decryptOutputs(c, acc, block, nPatches, () => biases[oc] ?? 0));
      multiplies++;                                    // the output mask
      sc.drop(acc); sc.drop(ptW);
    }
  } finally {
    sc.close();
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
           rotations, multiplies, patchesPerCiphertext: nPatches, rotateMs };
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
  /** The CONVOLUTION BLOCK's own wall-clock, kept separate from `fheMs`.
   *
   *  `fheMs` on this result is the whole student - conv plus every dense layer - because the spread
   *  of `conv` below is overridden by the accumulated total. That is the right number for "what did
   *  this inference cost", and the wrong denominator for "what fraction of the convolution is
   *  rotation": `rotations` and `rotateMs` come from `...conv` and describe the conv block ALONE,
   *  so dividing one by the other mixes two scopes. Round 38 published a 10% rotation share derived
   *  from exactly that mismatch. */
  convOnlyMs: number;
  /** Wall-clock of the dense layers that follow the convolution. `convOnlyMs + denseMs == fheMs`. */
  denseMs: number;
  /** Rotations in the dense layers only. `rotations + denseRotations == totalRotations`. */
  denseRotations: number;
  /** Time inside `rotateVector` across the dense layers only. */
  denseRotateMs: number;
}

/**
 * Evaluate the PUBLISHED medical student LAYER BY LAYER under CKKS - not end to end.
 *
 * Each layer's linear part runs under CKKS, its bias included: the bias is added to the ciphertext,
 * and every slot that is not one of the layer's outputs is masked and flooded before the client
 * decrypts (`decryptOutputs`), so what the client decrypts is the layer's output and nothing else.
 * The decryption itself stays: the conv's ReLU and the average pool run in plaintext in the client,
 * and the next dense layer re-encrypts the clear activation. The dense layers are packed, every row
 * of a layer in one ciphertext (`encryptedDense`). The student's final argmax is taken in the clear
 * (`argmax(v)` below). The garbled-circuit argmax elsewhere on the page runs on the bundled WDBC
 * logistic regression, not on this network, and it too receives logits the client has already
 * decrypted: in this tab it exercises the protocol and hides nothing from the key holder.
 *
 * This is a demonstration shortcut, not the state of the art: published FHE compilers run whole
 * networks under encryption, bootstrapping where this tab decrypts (Orion, ASPLOS 2025,
 * arXiv:2311.03470, up to ResNet-50 on ImageNet). Safhire (TMLR, arXiv:2509.01253) keeps this tab's
 * pattern, linear layers encrypted and the non-linearities in plaintext at the client, and adds the
 * randomized shuffling that keeps the model confidential from that client, which this tab does not.
 *
 * Until this existed the encrypted convolution ran on the student's SHAPE with a synthetic kernel
 * pattern. It was a genuine homomorphic convolution and an honest cost model, but it was not this
 * model's inference - so the encrypted-against-plaintext agreement of this model could not be
 * checked in a browser. Feeding the real weights makes the encrypted layers compute the published
 * prediction. (`crypto_fidelity`, the student against its teacher, stays an offline training
 * figure either way.)
 *
 * The layer order comes from the weights file rather than from an assumption here: conv, the conv's
 * own activation, average pool, flatten, then each dense with its recorded activation. Omitting the
 * conv's ReLU still reproduces every argmax on the test split while shifting the logits by ~0.15,
 * which is exactly why the file states each activation instead of leaving it to be inferred.
 *
 * ReLU and pooling are applied in the clear here. The project's stated hybrid is linear layers under
 * FHE and non-linear under MPC; for this student neither the ReLU, the pool nor the argmax is routed
 * through `clientMPC`, and that gap is reported rather than papered over.
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
  // Accumulated separately from the totals above, because the return spreads `...conv` (whose
  // `rotations` and `rotateMs` are the CONV BLOCK's) and then overrides `fheMs` with the total.
  // Without these, the only pairing a caller can form is conv-scoped rotations over a
  // student-scoped time - which is the division that produced Round 38's wrong 10% figure.
  let denseMs = 0, denseRotations = 0, denseRotateMs = 0;
  for (const d of w.dense) {
    if (v.length !== d.in) {
      throw new Error(`dense layer expects ${d.in} inputs, got ${v.length} - the published weights `
        + "and this evaluation disagree about the network, which must not be silently reshaped");
    }
    const r = await encryptedDense(d.weight, d.bias ?? new Array(d.out).fill(0), v);
    rotations += r.rotations; multiplies += r.multiplies; fheMs += r.fheMs;
    denseMs += r.fheMs; denseRotations += r.rotations; denseRotateMs += r.rotateMs;
    // Null-safe: the largest size actually read across layers, or null if none could be read.
    ctBytes = r.ctBytes == null ? ctBytes
      : ctBytes == null ? r.ctBytes
      : Math.max(ctBytes, r.ctBytes);
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
    convOnlyMs: conv.fheMs, denseMs, denseRotations, denseRotateMs,
  };
}

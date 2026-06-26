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
  const galois = keyGen.createGaloisKeys();   // power-of-two rotations for the dot-product slot sum
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
    for (let s = 1; s < sumWidth; s <<= 1) {                    // rotate-and-add → Σ slots in slot 0
      const rot = new seal.Ciphertext();
      evaluator.rotateVector(acc, s, galois, rot);
      const nxt = new seal.Ciphertext();
      evaluator.add(acc, rot, nxt);
      acc = nxt;
    }
    const out = new seal.Plaintext();
    decryptor.decrypt(acc, out);
    logits.push(encoder.decodeFloat64(out)[0] + (b[j] ?? 0));
  }
  const fheMs = performance.now() - t0;

  // approximation error vs the plaintext reference (CKKS is approximate)
  const plain = W.map((row, j) => row.reduce((a, w, i) => a + w * (x[i] ?? 0), 0) + (b[j] ?? 0));
  const maxAbsError = Math.max(...logits.map((v, j) => Math.abs(v - plain[j])), 0);
  return { logits, fheMs, ctBytes, maxAbsError };
}

export function argmax(a: number[]): number {
  return a.reduce((bi, v, i, arr) => (v > arr[bi] ? i : bi), 0);
}

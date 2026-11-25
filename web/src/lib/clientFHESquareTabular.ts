/**
 * The finance square student evaluated end to end under CKKS, one row a slot (Round 86, batch 4;
 * references/finance-square-student-preregistration.md).
 *
 * The network is dense 114 -> 128 with batch norm folded in, the square, dense 128 -> 2, the validation threshold folded
 * into the last bias (scripts/train/finance_square_student.py). The client scales and clips each row, then encrypts each
 * of the 114 inputs as its own ciphertext over a block of rows (N / 2 rows a block) and holds the secret key; the
 * evaluator computes every hidden unit as a plaintext-weight combination of the input ciphertexts, squares it, and folds
 * it into the two logits, which come back as two ciphertexts only the client can decrypt. Three levels: a
 * plaintext-weight product, a ciphertext square, a plaintext-weight product, so the medical square students' sets
 * (A2: N = 8192, 202 bits; B: N = 16384, 240 bits) hold it. No rotation, so no Galois keys.
 *
 * COST. node-seal encodes a constant only as a full vector (about 0.6 ms at N = 8192), and the first layer needs
 * 14,592 of them. Each weight is therefore encoded once for a group of blocks and applied to every block in the group;
 * the group is sized so that its input ciphertexts stay near 0.6 GB of the wasm heap.
 *
 * WHAT THIS IS NOT. The weights are public and the evaluator knows the network; it is the threat model of the medical
 * page's CKKS students. Both parties run in one process here.
 *
 * ON THE FINANCE PAGE SINCE F7b. F7 (references/finance-square-student-preregistration.md) did not adopt this network
 * against the same network with a ReLU; F7b (references/finance-square-student-page-preregistration.md) adopted it
 * against the linear student under parameter set A2, and on the owner's choice of 2026-11-25 it replaces the linear
 * student on the finance page, through `runSquareTabularOffThread` below. The trainers run this module on every test row
 * for the records (`scripts/train/finance_square_student.py ckks`, `scripts/train/finance_square_student_page.py ckks`),
 * and tests/test_r86_finance_square_ckks_in_node.py holds it to a NumPy reference.
 */
import sealWasmUrl from "/node_modules/node-seal/dist/seal_throws.wasm?url";
import { plainLogits, scaleRow, type SquareTabParams, type SquareTabWeights } from "./squareTabularMath";
export { plainLogits, scaleRow } from "./squareTabularMath";
export type { SquareTabParams, SquareTabWeights } from "./squareTabularMath";

export interface SquareTabRun {
  set: string; rows: number; blocks: number;
  /** Decrypted logits, row-major: [2 r] licit, [2 r + 1] laundering. */
  logits: Float64Array;
  /** The same in the clear, float64. */
  plain: Float64Array;
  maxAbsError: number; decisionsDiffer: number;
  keyMs: number; keyBytes: number;
  /** Per full block: the median times, and what crossed. */
  per_block: { rows: number; encrypt_ms: number; evaluate_ms: number; decrypt_ms: number;
               ciphertexts: { input: number; output: number }; bytes: { input: number; output: number } };
}

function check(w: SquareTabWeights): void {
  if (w.kind !== "dense-bn-square-dense" || w.w1.length !== w.width || w.w1[0]?.length !== w.n_in || w.b1.length !== w.width
      || w.w2.length !== 2 || w.w2[0]?.length !== w.width || w.b2.length !== 2 || w.centre.length !== w.n_in
      || w.scale.length !== w.n_in) {
    throw new Error("these weights are not the finance square student's layout");
  }
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

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : NaN;
};

/**
 * `rows` scaled rows (flat, row-major, n_in a row) under one parameter set. `progress(done, total)` is called after each
 * block. Returns the decrypted and the plaintext logits of every row and the cost of a full block.
 */
export async function runSquareTabular(w: SquareTabWeights, z: ArrayLike<number>, rows: number, set: string,
                                       progress?: (done: number, total: number) => void): Promise<SquareTabRun> {
  check(w);
  const p = w.ckks[set];
  if (!p) throw new Error(`no parameter set ${set} in these weights`);
  if (z.length < rows * w.n_in) throw new Error(`${z.length} values for ${rows} rows of ${w.n_in}`);
  const seal = await sealModule();
  const slots = p.N / 2, delta = Math.pow(2, p.scale_bits);
  // About 0.6 GB of input ciphertexts a group, well inside wasm's 4 GB: 114 inputs x (2 x N x 4 limbs x 8 bytes) a block.
  const group = Math.max(1, Math.floor(6.0e8 / (w.n_in * 2 * p.N * (p.coeff_mod_bits.length - 1) * 8)));
  const live: { delete(): void }[] = [];
  const keep = <T extends { delete(): void }>(o: T) => { live.push(o); return o; };
  let groupLive: { delete(): void }[] = [];
  const temp = <T extends { delete(): void }>(o: T) => { groupLive.push(o); return o; };
  const free = (o: { delete(): void } | null) => { if (o) { try { o.delete(); } catch { /* already freed */ } } };
  try {
    const tk = performance.now();
    const parms = keep(new seal.EncryptionParameters(seal.SchemeType.ckks));
    parms.setPolyModulusDegree(p.N);
    parms.setCoeffModulus(keep(seal.CoeffModulus.Create(p.N, Int32Array.from(p.coeff_mod_bits))));
    const context = keep(new seal.SEALContext(parms, true, seal.SecLevelType.tc128));
    if (!context.parametersSet()) throw new Error(`SEAL rejected N = ${p.N} with ${p.coeff_mod_bits.join("+")} bits`);
    const keyGen = keep(new seal.KeyGenerator(context));
    const pk = keep(keyGen.createPublicKey()), sk = keep(keyGen.secretKey());
    const relin = keep(keyGen.createRelinKeys());
    const keyBytes = pk.saveToArray(seal.ComprModeType.none).length + relin.saveToArray(seal.ComprModeType.none).length;
    const encoder = keep(new seal.CKKSEncoder(context));
    const encryptor = keep(new seal.Encryptor(context, pk));
    const decryptor = keep(new seal.Decryptor(context, sk));
    const ev = keep(new seal.Evaluator(context));
    const keyMs = performance.now() - tk;

    const constant = (v: number) => new Float64Array(slots).fill(v);
    /** A constant at a ciphertext's level and exact scale (for an addition), or at scale `at` (for a product). */
    const plainAt = (v: number, like: any, at: number) => {
      const out = new seal.Plaintext();
      encoder.encode(constant(v), at, out);
      ev.plainModSwitchToInplace(out, like.parmsId());
      return out;
    };

    const nBlocks = Math.ceil(rows / slots);
    const logits = new Float64Array(rows * 2);
    const enc: number[] = [], evalMs: number[] = [], dec: number[] = [];
    let upBytes = 0, downBytes = 0, blocksDone = 0;
    for (let g0 = 0; g0 < nBlocks; g0 += group) {
      const blocks = Array.from({ length: Math.min(group, nBlocks - g0) }, (_, i) => g0 + i);
      // The client: each input of each block, one row a slot.
      let t = performance.now();
      const inputs = blocks.map((b) => {
        const lo = b * slots, hi = Math.min(rows, lo + slots);
        return Array.from({ length: w.n_in }, (_, j) => {
          const v = new Float64Array(slots);
          for (let r = lo; r < hi; r++) v[r - lo] = Number(z[r * w.n_in + j]);
          const pt = new seal.Plaintext();
          encoder.encode(v, delta, pt);
          const ct = temp(new seal.Ciphertext());
          encryptor.encrypt(pt, ct);
          free(pt);
          if (b === blocks[0] && b === 0 && j === 0) upBytes = ct.saveToArray(seal.ComprModeType.none).length;
          return ct;
        });
      });
      enc.push((performance.now() - t) / blocks.length);

      // The evaluator: each hidden unit for every block of the group, its weights encoded once for the group.
      t = performance.now();
      const acc: (any | null)[][] = blocks.map(() => [null, null]);
      for (let k = 0; k < w.width; k++) {
        const h: any[] = blocks.map(() => null);
        for (let j = 0; j < w.n_in; j++) {
          const c = w.w1[k][j];
          if (c === 0) continue;
          const wp = plainAt(c, inputs[0][j], delta);
          for (let g = 0; g < blocks.length; g++) {
            if (h[g] === null) {
              h[g] = new seal.Ciphertext();
              ev.multiplyPlain(inputs[g][j], wp, h[g]);
            } else {
              const prod = new seal.Ciphertext();
              ev.multiplyPlain(inputs[g][j], wp, prod);
              ev.add(h[g], prod, h[g]);
              free(prod);
            }
          }
          free(wp);
        }
        if (h[0] === null) continue;                     // a unit with no weight: its square is b1^2, folded below
        for (const x of h) ev.rescaleToNextInplace(x);
        const bp = plainAt(w.b1[k], h[0], h[0].scale());
        const sq: any[] = h.map((x) => {
          ev.addPlain(x, bp, x);
          const s = new seal.Ciphertext();
          ev.square(x, s);
          ev.relinearizeInplace(s, relin);
          ev.rescaleToNextInplace(s);
          free(x);
          return s;
        });
        free(bp);
        for (let c = 0; c < 2; c++) {
          const vp = plainAt(w.w2[c][k], sq[0], delta);
          for (let g = 0; g < blocks.length; g++) {
            if (acc[g][c] === null) {
              acc[g][c] = temp(new seal.Ciphertext());
              ev.multiplyPlain(sq[g], vp, acc[g][c]);
            } else {
              const prod = new seal.Ciphertext();
              ev.multiplyPlain(sq[g], vp, prod);
              ev.add(acc[g][c], prod, acc[g][c]);
              free(prod);
            }
          }
          free(vp);
        }
        for (const s of sq) free(s);
      }
      // A unit with no weights contributes w2[c][k] * b1[k]^2, a constant.
      const offset = [0, 1].map((c) => w.b2[c] + w.w1.reduce((a, row, k) => (row.every((v) => v === 0) ? a + w.w2[c][k] * w.b1[k] ** 2 : a), 0));
      for (let g = 0; g < blocks.length; g++) {
        for (let c = 0; c < 2; c++) {
          ev.rescaleToNextInplace(acc[g][c]);
          const op = plainAt(offset[c], acc[g][c], acc[g][c].scale());
          ev.addPlain(acc[g][c], op, acc[g][c]);
          free(op);
        }
      }
      evalMs.push((performance.now() - t) / blocks.length);
      if (g0 === 0) downBytes = acc[0][0].saveToArray(seal.ComprModeType.none).length * 2;

      // The client: decrypt the two logits of each block.
      t = performance.now();
      for (let g = 0; g < blocks.length; g++) {
        const lo = blocks[g] * slots, hi = Math.min(rows, lo + slots);
        for (let c = 0; c < 2; c++) {
          const pt = new seal.Plaintext();
          decryptor.decrypt(acc[g][c], pt);
          const vals: Float64Array = encoder.decodeFloat64(pt);
          free(pt);
          for (let r = lo; r < hi; r++) logits[2 * r + c] = vals[r - lo];
        }
      }
      dec.push((performance.now() - t) / blocks.length);
      for (const o of groupLive.reverse()) free(o);
      groupLive = [];
      blocksDone += blocks.length;
      progress?.(blocksDone, nBlocks);
    }

    const plain = new Float64Array(rows * 2);
    let maxAbsError = 0, decisionsDiffer = 0;
    for (let r = 0; r < rows; r++) {
      const [a, b] = plainLogits(w, z, r * w.n_in);
      plain[2 * r] = a;
      plain[2 * r + 1] = b;
      maxAbsError = Math.max(maxAbsError, Math.abs(logits[2 * r] - a), Math.abs(logits[2 * r + 1] - b));
      if ((logits[2 * r + 1] > logits[2 * r]) !== (b > a)) decisionsDiffer++;
    }
    return {
      set, rows, blocks: nBlocks, logits, plain, maxAbsError, decisionsDiffer, keyMs, keyBytes,
      per_block: { rows: slots, encrypt_ms: median(enc), evaluate_ms: median(evalMs), decrypt_ms: median(dec),
                   ciphertexts: { input: w.n_in, output: 2 }, bytes: { input: upBytes * w.n_in, output: downBytes } },
    };
  } finally {
    for (const o of groupLive.reverse()) free(o);
    for (const o of live.reverse()) free(o);
  }
}


/** One run over many raw rows: each worker's block, put back in row order. */
export interface SquareTabBatch {
  set: string; rows: number; blocks: number;
  /** Row-major, [2 r] licit and [2 r + 1] laundering: decrypted, and in the clear in float64. */
  logits: Float64Array; plain: Float64Array;
  maxAbsError: number; decisionsDiffer: number;
  /** The slowest worker's time from its key generation to its last decryption, and its key generation alone. */
  totalMs: number; keyMs: number; keyBytes: number;
  per_block: SquareTabRun["per_block"];
  /** Where it ran, for the caption: "a Web Worker", "2 Web Workers at once", or "the calling thread". */
  ranIn: string;
}

/** How many blocks run at once: half the logical cores, at most four, never more than the blocks; one below 8 GB. */
export function tabularPoolSize(blocks: number, logicalCores: number, deviceMemoryGb?: number): number {
  if (deviceMemoryGb != null && deviceMemoryGb < 8) return 1;
  return Math.max(1, Math.min(blocks, 4, Math.floor(logicalCores / 2)));
}

/** The raw rows under CKKS, a block of N / 2 rows to a Web Worker, as the finance page runs the square network. */
export async function runSquareTabularOffThread(w: SquareTabWeights, rows: number[][], set: string,
                                                onBlock?: (done: number, total: number) => void): Promise<SquareTabBatch> {
  const p = w.ckks[set];
  if (!p) throw new Error(`parameter set ${set} is not one these weights offer`);
  const slots = p.N / 2;
  const blocks: number[][][] = [];
  for (let i = 0; i < rows.length; i += slots) blocks.push(rows.slice(i, i + slots));
  type Part = { run: SquareTabRun; totalMs: number };
  let parts: Part[];
  let ranIn: string;
  if (typeof window === "undefined" || typeof Worker === "undefined") {
    parts = [];
    for (const block of blocks) {
      const z = new Float64Array(block.length * w.n_in);
      block.forEach((x, i) => z.set(scaleRow(w, x), i * w.n_in));
      const t0 = performance.now();
      parts.push({ run: await runSquareTabular(w, z, block.length, set, () => {}), totalMs: performance.now() - t0 });
      onBlock?.(parts.length, blocks.length);
    }
    ranIn = "the calling thread";
  } else {
    const n = tabularPoolSize(blocks.length, navigator.hardwareConcurrency || 1,
                              (navigator as { deviceMemory?: number }).deviceMemory);
    parts = new Array(blocks.length);
    let next = 0, finished = 0;
    // Vite bundles the worker from exactly this form: `new URL(...)` written inside `new Worker(...)`.
    const workers = Array.from({ length: n },
      () => new Worker(new URL("./clientFHESquareTabular.worker.ts", import.meta.url), { type: "module" }));
    try {
      await Promise.all(workers.map((worker) => (async () => {
        while (next < blocks.length) {
          const k = next++;
          parts[k] = await new Promise<Part>((resolve, reject) => {
            worker.onmessage = (e: MessageEvent<{ ok: boolean; run?: SquareTabRun; totalMs?: number; error?: string }>) => {
              if (e.data.ok && e.data.run) resolve({ run: e.data.run, totalMs: e.data.totalMs ?? 0 });
              else reject(new Error(`in its worker: ${e.data.error ?? "no result"}`));
            };
            worker.onerror = (e) => reject(new Error(`its worker failed: ${e.message || "the worker could not be loaded"}`));
            worker.onmessageerror = () => reject(new Error("its worker's result could not be read back"));
            worker.postMessage({ id: k + 1, rows: blocks[k], weights: w, set });
          });
          onBlock?.(++finished, blocks.length);
        }
      })()));
    } finally {
      // Every worker, also one still running when another failed: each frees its own wasm heap.
      for (const worker of workers) worker.terminate();
    }
    ranIn = workers.length === 1 ? "a Web Worker" : `${workers.length} Web Workers at once`;
  }
  const logits = new Float64Array(rows.length * 2), plain = new Float64Array(rows.length * 2);
  let at = 0;
  for (const { run } of parts) {
    logits.set(run.logits, at);
    plain.set(run.plain, at);
    at += run.rows * 2;
  }
  const first = parts[0].run;
  return {
    set, rows: rows.length, blocks: parts.length, logits, plain,
    maxAbsError: Math.max(...parts.map((q) => q.run.maxAbsError)),
    decisionsDiffer: parts.reduce((a, q) => a + q.run.decisionsDiffer, 0),
    totalMs: Math.max(...parts.map((q) => q.totalMs)), keyMs: Math.max(...parts.map((q) => q.run.keyMs)),
    keyBytes: first.keyBytes, per_block: first.per_block, ranIn,
  };
}

/**
 * A binary tree QCNN evaluated under CKKS stage by stage, one row a slot (Round 86; references/satellite-qcnn-tn-preregistration.md, D).
 *
 * The circuit is a tree network (`src/qsh/qhe/qcnn_tree.py`): each wire starts as the Bloch vector of RY(theta)|0>, (1, sin theta, 0, cos theta), and each pooling stage maps a pair's two
 * Bloch vectors to the survivor's by a bilinear map, r'_k = sum_ij M_kij r_a,i r_b,j, the same M for every pair of the stage. The client encrypts each wire's two non-zero components for a
 * block of rows, one row a slot, and holds the secret key; the evaluator applies the stages with the maps' coefficients as plaintext scalars and returns one ciphertext, the block's <Z0>,
 * which only the client can decrypt; it applies the readout head and the threshold. Each stage costs two levels (a plaintext-scalar combination, then one ciphertext product), and the last
 * stage's map is folded with the readout vector, so eight wires need six levels: N = 16384 and 60 + 6 x 40 + 60 = 360 bits, inside the HomomorphicEncryption.org standard's post-quantum bound
 * at N = 16384 (411 bits). No rotation, so no Galois keys.
 *
 * WHAT THIS IS NOT. Classical computation under RLWE, not quantum homomorphic encryption: the circuit is simulated classically, as it already is on the page. The weights are public and the
 * evaluator knows the circuit; it is the threat model of the medical page's CKKS students. Both parties run in one process here.
 */
import sealWasmUrl from "/node_modules/node-seal/dist/seal_throws.wasm?url";

export interface TreeNetwork {
  /** One map a pooling stage, [k][i][j] for k, i, j in 0..3 (0 the trace, then x, y, z). */
  maps: number[][][][];
  /** <Z0> = readout . r of the last survivor. */
  readout: number[];
  scale: number; bias: number;
  feature_mean: number[]; feature_std: number[]; encoding_scale: number; n_qubits: number;
}

export interface TreeCkksParams { N: number; coeff_mod_bits: number[]; scale_bits: number }
export const TREE_CKKS: TreeCkksParams = { N: 16384, coeff_mod_bits: [60, 40, 40, 40, 40, 40, 40, 60], scale_bits: 40 };

export interface TreeCkksRun {
  readouts: Float64Array; rows: number; blocks: number; ciphertextsUp: number;
  upBytes: number; downBytes: number; keyBytes: number;
  keyMs: number; encryptMs: number; evaluateMs: number; decryptMs: number;
}

/** Each wire's Bloch vector for each row: [wire][row] -> [1, x, y, z]. */
export function encodeRows(net: TreeNetwork, rows: ArrayLike<number>[]): Float64Array[][] {
  const n = net.n_qubits;
  return Array.from({ length: n }, (_, q) => {
    const x = new Float64Array(rows.length), z = new Float64Array(rows.length);
    for (let r = 0; r < rows.length; r++) {
      const sd = net.feature_std[q] || 1;
      const theta = net.encoding_scale * Math.tanh((Number(rows[r][q]) - net.feature_mean[q]) / sd) * Math.PI;
      x[r] = Math.sin(theta);
      z[r] = Math.cos(theta);
    }
    return [new Float64Array(rows.length).fill(1), x, new Float64Array(rows.length), z];
  });
}

/** The contraction in the clear, float64: what the encrypted path must reproduce. */
export function clearReadout(net: TreeNetwork, rows: ArrayLike<number>[]): Float64Array {
  let wires = encodeRows(net, rows);
  for (const m of net.maps) {
    const next: Float64Array[][] = [];
    for (let p = 0; p + 1 < wires.length; p += 2) {
      const a = wires[p], b = wires[p + 1];
      next.push([0, 1, 2, 3].map((k) => {
        const out = new Float64Array(rows.length);
        for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
          const c = m[k][i][j];
          if (c === 0) continue;
          for (let r = 0; r < rows.length; r++) out[r] += c * a[i][r] * b[j][r];
        }
        return out;
      }));
    }
    wires = next;
  }
  const out = new Float64Array(rows.length);
  for (let k = 0; k < 4; k++) for (let r = 0; r < rows.length; r++) out[r] += net.readout[k] * wires[0][k][r];
  return out;
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

/** Coefficients smaller than this are dropped: the maps carry float noise of about 1e-17 where the circuit gives 0. */
const ZERO = 1e-14;

export async function treeReadoutCkks(net: TreeNetwork, rows: ArrayLike<number>[], p: TreeCkksParams = TREE_CKKS): Promise<TreeCkksRun> {
  if (net.n_qubits & (net.n_qubits - 1)) throw new Error(`${net.n_qubits} wires: a power of two is needed`);
  if (net.maps.length !== Math.log2(net.n_qubits)) throw new Error(`${net.maps.length} stages for ${net.n_qubits} wires`);
  if (p.coeff_mod_bits.length - 2 < 2 * net.maps.length) throw new Error(`${p.coeff_mod_bits.length - 2} levels for ${net.maps.length} stages of two`);
  const seal = await sealModule();
  const slots = p.N / 2, delta = Math.pow(2, p.scale_bits);
  const live: { delete(): void }[] = [];
  const keep = <T extends { delete(): void }>(o: T) => { live.push(o); return o; };
  // Objects made for one block are freed when the block is decrypted: a run over many blocks would otherwise hold every
  // intermediate ciphertext (2 MB each at N = 16384) until the end and outgrow wasm's 4 GB.
  let blockLive: { delete(): void }[] = [];
  const temp = <T extends { delete(): void }>(o: T) => { blockLive.push(o); return o; };
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

    const ct = () => temp(new seal.Ciphertext());
    const fill = (v: number) => new Float64Array(slots).fill(v);
    /** A constant at a ciphertext's level and exact scale, for an addition. */
    const constLike = (v: number, like: any) => {
      const out = temp(new seal.Plaintext());
      encoder.encode(fill(v), like.scale(), out);
      ev.plainModSwitchToInplace(out, like.parmsId());
      return out;
    };
    /** A scalar at a ciphertext's level, at scale `at` (delta unless given), for a multiplication. */
    const scalarAt = (v: number, like: any, at: number = delta) => {
      const out = temp(new seal.Plaintext());
      encoder.encode(fill(v), at, out);
      ev.plainModSwitchToInplace(out, like.parmsId());
      return out;
    };
    const times = (c: any, v: number) => { const out = ct(); ev.multiplyPlain(c, scalarAt(v, c), out); return out; };
    const sum = (terms: any[]) => {
      if (terms.length === 1) return terms[0];
      const out = ct();
      ev.add(terms[0], terms[1], out);
      for (let t = 2; t < terms.length; t++) ev.add(out, terms[t], out);
      return out;
    };
    const down = (c: any, like: any) => { const out = ct(); ev.cipherModSwitchTo(c, like.parmsId(), out); return out; };

    /** Sum_i coef[i] x[i] + c0, as one ciphertext one level down, or null when every coefficient is zero. */
    const affine = (xs: (any | null)[], coef: number[], c0: number) => {
      const terms = xs.map((x, i) => (x && Math.abs(coef[i]) > ZERO ? times(x, coef[i]) : null)).filter((t): t is any => t !== null);
      if (!terms.length) return null;
      const s = sum(terms);
      ev.rescaleToNextInplace(s);
      if (Math.abs(c0) > ZERO) ev.addPlain(s, constLike(c0, s), s);
      return s;
    };

    const nBlocks = Math.ceil(rows.length / slots);
    const readouts = new Float64Array(rows.length);
    let upBytes = 0, downBytes = 0, ciphertextsUp = 0, encryptMs = 0, evaluateMs = 0, decryptMs = 0;
    const bloch = encodeRows(net, rows);
    for (let b = 0; b < nBlocks; b++) {
      const lo = b * slots, hi = Math.min(rows.length, lo + slots);
      // The client: x and z of every wire for this block (y is 0 at the encoding).
      let t = performance.now();
      let wires: (any | null)[][] = bloch.map((comp) => [null, 1, null, 3].map((c) => {
        if (c === null) return null;
        const v = new Float64Array(slots);
        v.set(comp[c as number].subarray(lo, hi));
        const pt = temp(new seal.Plaintext());
        encoder.encode(v, delta, pt);
        const out = ct();
        encryptor.encrypt(pt, out);
        upBytes += out.saveToArray(seal.ComprModeType.none).length;
        ciphertextsUp++;
        return out;
      }));
      encryptMs += performance.now() - t;

      // The evaluator: the stages, the last one folded with the readout vector.
      t = performance.now();
      for (let s = 0; s < net.maps.length; s++) {
        const m = net.maps[s], last = s === net.maps.length - 1;
        const outputs = last ? [net.readout] : [null, [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
        const next: (any | null)[][] = [];
        for (let pr = 0; pr + 1 < wires.length; pr += 2) {
          const a = wires[pr], bw = wires[pr + 1];
          const res: (any | null)[] = [null, null, null, null];
          outputs.forEach((w, slot) => {
            if (!w) return;
            // The coefficients of this output: c[i][j] = sum_k w[k] M[k][i][j].
            const c = [0, 1, 2, 3].map((i) => [0, 1, 2, 3].map((j) => w.reduce((acc, wk, k) => acc + wk * m[k][i][j], 0)));
            const parts: any[] = [];
            let constant = c[0][0];
            for (let j = 1; j < 4; j++) {
              if (!bw[j]) continue;
              const u = affine(a, c.map((row) => row[j]), c[0][j]);      // sum_i c[i][j] a_i + c[0][j], one level down
              if (!u) continue;
              const prod = ct();
              ev.multiply(u, down(bw[j], u), prod);
              ev.relinearizeInplace(prod, relin);
              ev.rescaleToNextInplace(prod);
              parts.push(prod);
            }
            const v = affine(a, c.map((row) => row[0]), 0);              // the terms in a alone, one level down
            if (v) {
              // Times 1 at b's own scale, so this term lands on the products' level AND scale: SEAL adds only equal scales.
              const bScale = (bw.find((x) => x) as any).scale();
              const one = ct();
              ev.multiplyPlain(v, scalarAt(1, v, bScale), one);
              ev.rescaleToNextInplace(one);
              parts.push(one);
            }
            if (!parts.length) throw new Error("a stage output with no encrypted term");
            const out = sum(parts);
            if (Math.abs(constant) > ZERO) ev.addPlain(out, constLike(constant, out), out);
            res[last ? 0 : slot] = out;
          });
          next.push(res);
        }
        wires = next;
      }
      const result = wires[0][0];
      evaluateMs += performance.now() - t;
      downBytes += result.saveToArray(seal.ComprModeType.none).length;

      t = performance.now();
      const pt = temp(new seal.Plaintext());
      decryptor.decrypt(result, pt);
      const vals: Float64Array = encoder.decodeFloat64(pt);
      readouts.set(vals.subarray(0, hi - lo), lo);
      decryptMs += performance.now() - t;
      for (const o of blockLive.reverse()) { try { o.delete(); } catch { /* already freed */ } }
      blockLive = [];
    }
    return { readouts, rows: rows.length, blocks: nBlocks, ciphertextsUp, upBytes, downBytes, keyBytes, keyMs, encryptMs, evaluateMs, decryptMs };
  } finally {
    for (const o of blockLive.reverse()) { try { o.delete(); } catch { /* already freed */ } }
    for (const o of live.reverse()) { try { o.delete(); } catch { /* already freed */ } }
  }
}

/**
 * A trained QCNN's prediction with its input under CKKS (Round 86).
 *
 * The readout is a linear form over 3^n products of sines and cosines of the encoded angles (`qcnnPauli.ts`). The client computes those products from its own features, encrypts them and
 * sends the ciphertexts, and holds the secret key; the server multiplies by the plaintext coefficients the circuit determines, sums the slots and returns one ciphertext per readout, which only
 * the client can decrypt. One multiplicative level, the demo's frozen parameters (N = 8192, moduli 60-40-40-60), no bootstrapping and no relinearisation key.
 *
 * WHAT THIS IS, AND WHAT IT IS NOT. It is classical computation under RLWE: the circuit is simulated classically, as it already is on this page, and only the input is hidden from the evaluator. It is
 * not quantum homomorphic encryption, there is no quantum hardware, and nothing about it is faster or more accurate than the plaintext circuit, whose decision it reproduces. The evaluator
 * knows the circuit. The scores the client receives determine the model: the form has 3^n coefficients, so that many chosen inputs recover it, which the published weights make moot here and a
 * deployment with a private model would have to address. Both parties run in this tab, which is what allows the comparison with the clear circuit below.
 *
 * The server's sum runs over the whole slot cycle, so every slot of the returned ciphertext holds the full value and no partial sum reaches the key holder. The other slots still carry a fresh
 * flood, added under the public key as in `clientFHE.ts`, so the decryption the client reads is re-randomised.
 */

import { CKKS_FLOOD_BITS } from "./clientFHE";
import { Arena, sealed } from "./clientHybridCnn";
import { evaluateForm, pauliFeatures, pauliForm } from "./qcnnPauli";
import { qcnnInferTrained, type QcnnWeights } from "./quantumSim";

export interface QcnnCkksResult {
  cls: number; clearClass: number; matchesClear: boolean;
  /** One readout per output, decrypted: <Z0> on a binary model, class marginals on a multiclass one. */
  readouts: number[];
  /** Largest |decrypted - clear circuit| over the readouts, and the same for the form in the clear. */
  maxError: number; maxIdentityError: number;
  nFeatures: number; nTerms: number; nCiphertexts: number; nOutputs: number;
  rotations: number; upBytes: number; downBytes: number; keyBytes: number;
  formMs: number; keyMs: number; encryptMs: number; evaluateMs: number; decryptMs: number; totalMs: number;
}

const ROTATION_STEPS = (slots: number) => { const s: number[] = []; for (let step = 1; step < slots; step <<= 1) s.push(step); return s; };

export async function qcnnInferCkks(w: QcnnWeights, features: number[]): Promise<QcnnCkksResult> {
  const t0 = performance.now();
  const { seal, context, slots, scale } = await sealed();
  const a = new Arena();
  try {
    const form = pauliForm(w), formMs = performance.now() - t0;
    const clear = qcnnInferTrained(features, w);
    const phi = pauliFeatures(w, features);
    const nCiphertexts = Math.ceil(form.nTerms / slots);
    const chunk = (v: Float64Array, k: number) => {
      const out = new Float64Array(slots);
      for (let i = 0; i < slots && k * slots + i < v.length; i++) out[i] = v[k * slots + i];
      return out;
    };

    let t = performance.now();
    const gen = a.hold(new seal.KeyGenerator(context));
    const pk = a.hold(gen.createPublicKey()), sk = a.hold(gen.secretKey());
    const galois = a.hold(gen.createGaloisKeysWithSteps(Int32Array.from(ROTATION_STEPS(slots))));
    const keyBytes = pk.saveToArray(seal.ComprModeType.none).length + galois.saveToArray(seal.ComprModeType.none).length;
    const keyMs = performance.now() - t;
    const encoder = a.hold(new seal.CKKSEncoder(context)), encryptor = a.hold(new seal.Encryptor(context, pk));
    const decryptor = a.hold(new seal.Decryptor(context, sk)), evaluator = a.hold(new seal.Evaluator(context));

    t = performance.now();
    const cts: any[] = [];
    let upBytes = 0;
    for (let k = 0; k < nCiphertexts; k++) {
      const p = a.hold(new seal.Plaintext()), ct = a.hold(new seal.Ciphertext());
      encoder.encode(chunk(phi, k), scale, p);
      encryptor.encrypt(p, ct);
      upBytes += ct.saveToArray(seal.ComprModeType.none).length;
      cts.push(ct);
    }
    const encryptMs = performance.now() - t;

    t = performance.now();
    const returned: any[] = [];
    let rotations = 0, downBytes = 0;
    for (const c of form.coefficients) {
      let acc: any = null;
      for (let k = 0; k < nCiphertexts; k++) {
        const pc = a.hold(new seal.Plaintext()), part = a.hold(new seal.Ciphertext());
        encoder.encode(chunk(c, k), scale, pc);
        evaluator.multiplyPlain(cts[k], pc, part);
        if (acc) { const s = a.hold(new seal.Ciphertext()); evaluator.add(acc, part, s); acc = s; } else acc = part;
      }
      for (const step of ROTATION_STEPS(slots)) {
        const r = a.hold(new seal.Ciphertext()), s = a.hold(new seal.Ciphertext());
        evaluator.rotateVector(acc, step, galois, r);
        evaluator.add(acc, r, s);
        acc = s; rotations++;
      }
      const flood = new Float64Array(slots);
      for (let i = 1; i < slots; i++) flood[i] = (Math.random() * 2 - 1) * 2 ** CKKS_FLOOD_BITS;
      const pf = a.hold(new seal.Plaintext()), f = a.hold(new seal.Ciphertext()), out = a.hold(new seal.Ciphertext());
      encoder.encode(flood, acc.scale(), pf);
      encryptor.encrypt(pf, f);
      evaluator.add(acc, f, out);
      downBytes += out.saveToArray(seal.ComprModeType.none).length;
      returned.push(out);
    }
    const evaluateMs = performance.now() - t;

    t = performance.now();
    const readouts = returned.map((ct) => {
      const p = a.hold(new seal.Plaintext());
      decryptor.decrypt(ct, p);
      return encoder.decodeFloat64(p)[0] as number;
    });
    const decryptMs = performance.now() - t;

    const clearReadouts = w.readout?.readout_qubits ? clear.probs! : [clear.expectation];
    const identity = form.coefficients.map((c) => evaluateForm(c, phi));
    const maxError = Math.max(...readouts.map((v, i) => Math.abs(v - clearReadouts[i])));
    const maxIdentityError = Math.max(...identity.map((v, i) => Math.abs(v - clearReadouts[i])));
    let cls: number;
    if (w.readout?.readout_qubits) {
      const sc = w.scale as number[], bi = w.bias as number[];
      cls = 0;
      readouts.forEach((p, c) => { if (p * sc[c] + bi[c] > readouts[cls] * sc[cls] + bi[cls]) cls = c; });
    } else {
      cls = readouts[0] * (w.scale as number) + (w.bias as number) > 0 ? 1 : 0;
    }
    return {
      cls, clearClass: clear.cls, matchesClear: cls === clear.cls, readouts, maxError, maxIdentityError,
      nFeatures: form.nFeatures, nTerms: form.nTerms, nCiphertexts, nOutputs: form.coefficients.length,
      rotations, upBytes, downBytes, keyBytes, formMs, keyMs, encryptMs, evaluateMs, decryptMs, totalMs: performance.now() - t0,
    };
  } finally {
    a.close();
  }
}

/**
 * A trained QCNN's readout as a linear form over its encoding's Pauli products (Round 86).
 *
 * Every shipped circuit encodes a feature z on its own wire as RY(theta)|0>, with theta = c pi tanh(z). On one wire that state's density matrix is
 *
 *     rho(theta) = (I + sin(theta) X + cos(theta) Z) / 2
 *
 * with no Y term, because the state is real. So over the n encoded wires, and for an observable O after the circuit U,
 *
 *     <O> = Tr(U^dagger O U rho(x)) = sum over P in {I, X, Z}^n of c_P phi_P(x),    c_P = 2^-n Tr(A P),    phi_P(x) = prod_q phi_{P_q}(theta_q),
 *
 * where phi_I = 1, phi_X = sin, phi_Z = cos, and A is the real part of U^dagger O U on the 2^n encoded basis states (the wires past `n_features` stay in |0>). There are 3^n
 * terms, and the identity is exact, not an approximation: the terms differ from the circuit's own readout by rounding, 1e-14 on the shipped weights.
 *
 * That linearity is the known reason a model of this kind has a classical description (Schuld, arXiv:2101.11020; on QCNNs with a product encoding, Bermejo et al., PRX Quantum 7,
 * 020304, 2026). Here it is also what makes the evaluation cheap under CKKS: the client computes phi(x) in the clear from its own features, and the server's part is one inner product.
 *
 * `coefficients` is built from the page's own circuit code, through the same gates the state-vector simulator applies.
 */

import { applyQcnnCircuit, type QcnnWeights, stateVectorOps, validateQcnnWeights } from "./quantumSim";

export interface PauliForm {
  /** Wires that carry a feature; the form has 3^nFeatures terms. */
  nFeatures: number;
  nTerms: number;
  /** One vector c per readout: a single <Z0> on a binary model, one class marginal per class on a multiclass one. */
  coefficients: Float64Array[];
  /** The surviving wires the readout read, from the run that built the form. */
  survivors: number[];
}

/** U's columns on the 2^nf encoded basis states, by running the circuit on each. */
function circuitColumns(w: QcnnWeights, dim: number) {
  const L = validateQcnnWeights(w), N = 1 << L.nq;
  const columnsRe: Float64Array[] = [], columnsIm: Float64Array[] = [];
  let survivors: number[] = [];
  for (let j = 0; j < dim; j++) {
    const re = new Float64Array(N), im = new Float64Array(N);
    re[j] = 1;
    survivors = applyQcnnCircuit(w, L, stateVectorOps(re, im, N));
    columnsRe.push(re); columnsIm.push(im);
  }
  return { L, N, columnsRe, columnsIm, survivors };
}

/** One sign or indicator vector over the N basis states per readout, by the same bit conventions `qcnnInferTrained` reads. */
function readoutDiagonals(w: QcnnWeights, N: number, m: number, survivors: number[]): Float64Array[] {
  if (m === 0) return [Float64Array.from({ length: N }, (_, i) => ((i & 1) ? -1 : 1))];
  const k = 1 << m, wires = survivors.slice(0, m), out = Array.from({ length: k }, () => new Float64Array(N));
  for (let i = 0; i < N; i++) {
    let c = 0;
    for (let j = 0; j < m; j++) c |= ((i >>> wires[j]) & 1) << j;
    out[c][i] = 1;
  }
  return out;
}

/** c_P = 2^-n Tr(A P) for every P in {I, X, Z}^n; digit q of the label (base 3) is wire q's letter, 0 = I, 1 = X, 2 = Z. */
function pauliCoefficients(A: Float64Array[], nf: number): Float64Array {
  const dim = 1 << nf, nTerms = 3 ** nf, c = new Float64Array(nTerms);
  for (let label = 0; label < nTerms; label++) {
    let x = 0, z = 0, v = label;
    for (let q = 0; q < nf; q++) {
      const d = v % 3;
      v = (v - d) / 3;
      if (d === 1) x |= 1 << q; else if (d === 2) z |= 1 << q;
    }
    let trace = 0;
    for (let b = 0; b < dim; b++) {
      let parity = 0, bits = b & z;
      while (bits) { parity ^= 1; bits &= bits - 1; }
      trace += (parity ? -1 : 1) * A[b][b ^ x];
    }
    c[label] = trace / dim;
  }
  return c;
}

export function pauliForm(w: QcnnWeights): PauliForm {
  const nf = w.n_features ?? w.n_qubits, dim = 1 << nf;
  const { L, N, columnsRe, columnsIm, survivors } = circuitColumns(w, dim);
  const diagonals = readoutDiagonals(w, N, L.m, survivors);
  const coefficients = diagonals.map((o) => {
    const A = Array.from({ length: dim }, () => new Float64Array(dim));
    for (let a = 0; a < dim; a++) {
      for (let b = a; b < dim; b++) {
        let v = 0;
        for (let i = 0; i < N; i++) v += o[i] * (columnsRe[a][i] * columnsRe[b][i] + columnsIm[a][i] * columnsIm[b][i]);
        A[a][b] = v; A[b][a] = v;
      }
    }
    return pauliCoefficients(A, nf);
  });
  return { nFeatures: nf, nTerms: 3 ** nf, coefficients, survivors };
}

/** phi(x), in the label order of `pauliForm`. Computed by the party that holds the features, in the clear. `features` is the row the artifact declares (`input_space`). */
export function pauliFeatures(w: QcnnWeights, features: number[]): Float64Array {
  const nf = w.n_features ?? w.n_qubits, c = w.encoding_scale ?? 1;
  const phi = new Float64Array(3 ** nf);
  phi[0] = 1;
  let filled = 1;
  for (let q = 0; q < nf; q++) {
    let v = features[q] ?? 0;
    if (w.input_space === "raw") v = (v - w.feature_mean![q]) / (w.feature_std![q] || 1);
    const theta = c * Math.tanh(v) * Math.PI, s = Math.sin(theta), co = Math.cos(theta);
    for (let i = 0; i < filled; i++) { phi[i + filled] = phi[i] * s; phi[i + 2 * filled] = phi[i] * co; }
    filled *= 3;
  }
  return phi;
}

/** The form's value in the clear: the identity the tests and the CKKS self-check rest on. */
export const evaluateForm = (c: Float64Array, phi: Float64Array): number => { let v = 0; for (let i = 0; i < c.length; i++) v += c[i] * phi[i]; return v; };

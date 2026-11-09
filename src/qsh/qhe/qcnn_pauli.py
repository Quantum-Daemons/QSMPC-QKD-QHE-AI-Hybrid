"""A trained QCNN's readout as a linear form, and its evaluation under CKKS (Round 86).

The twin of ``web/src/lib/qcnnPauli.ts`` and ``clientQcnnCkks.ts``. Every shipped circuit encodes a feature on its own wire as RY(theta)|0>, so
``rho(theta) = (I + sin(theta) X + cos(theta) Z) / 2`` with no Y term, and for an observable after the circuit U

    <O> = sum over P in {I, X, Z}^n of c_P phi_P(x),    c_P = 2^-n Tr(A P),

with A the real part of U^dagger O U on the 2^n encoded basis states and phi_P a product of sines and cosines of the encoded angles. The identity is exact. It is why a model of this
kind has a classical description (Schuld, arXiv:2101.11020; on QCNNs with a product encoding, Bermejo et al., PRX Quantum 7, 020304, 2026), and it makes the evaluation cheap under CKKS: the
party with the features computes phi(x) in the clear and encrypts it, and the party with the circuit takes one inner product. Classical computation under RLWE, not quantum
homomorphic encryption. This module serves the binary artifacts that ``qcnn_trained`` serves.
"""

from __future__ import annotations

import numpy as np

from ..config import get_config
from .qcnn_trained import _run_circuit, final_state, validate


def pauli_form(weights: dict) -> np.ndarray:
    """c_P for <Z0>; digit q of the base-3 label is wire q's letter, 0 = I, 1 = X, 2 = Z (the browser's order)."""
    layout = validate(weights)
    nq, nf = layout["nq"], layout["nf"]
    dim, n = 1 << nf, 1 << nq
    re = np.zeros((dim, n))
    im = np.zeros((dim, n))
    for j in range(dim):
        e = np.zeros(n)
        e[j] = 1.0
        re[j], im[j] = final_state(e, weights)
    sign = np.where((np.arange(n) & 1) == 0, 1.0, -1.0)
    # A = Re(U^dagger O U) restricted to the encoded basis states, by rows of U.
    a = (re * sign) @ re.T + (im * sign) @ im.T
    n_terms = 3 ** nf
    c = np.zeros(n_terms)
    index = np.arange(dim)
    for label in range(n_terms):
        x, z, v = 0, 0, label
        for q in range(nf):
            d = v % 3
            v //= 3
            if d == 1:
                x |= 1 << q
            elif d == 2:
                z |= 1 << q
        parity = np.array([bin(int(i)).count("1") & 1 for i in (index & z)])
        c[label] = float(((1 - 2 * parity) * a[index, index ^ x]).sum()) / dim
    return c


def pauli_features(weights: dict, features) -> np.ndarray:
    """phi(x) in the label order of :func:`pauli_form`, from a row in the artifact's input space."""
    layout = validate(weights)
    nf, c = layout["nf"], layout["c"]
    x = np.asarray(features, dtype=float).ravel()[:nf]
    if weights.get("input_space") == "raw":
        sd = np.asarray(weights["feature_std"], dtype=float)[:nf]
        x = (x - np.asarray(weights["feature_mean"], dtype=float)[:nf]) / np.where(sd == 0, 1.0, sd)
    phi = np.zeros(3 ** nf)
    phi[0] = 1.0
    filled = 1
    for q in range(nf):
        theta = c * float(np.tanh(x[q])) * np.pi
        s, co = np.sin(theta), np.cos(theta)
        phi[filled:2 * filled] = phi[:filled] * s
        phi[2 * filled:3 * filled] = phi[:filled] * co
        filled *= 3
    return phi


def under_ckks(weights: dict, features, backend) -> dict:
    """The readout with ``features`` under CKKS: ``backend`` is a TenSEAL backend, so CPU. Returns the decrypted readout, the class and the difference from the clear circuit."""
    c = pauli_form(weights)
    phi = pauli_features(weights, features)
    slots = int(backend.slot_count)
    value = 0.0
    chunks = int(np.ceil(c.size / slots))
    for k in range(chunks):
        enc = backend.encrypt(phi[k * slots:(k + 1) * slots].tolist())
        value += backend.decrypt(enc.dot(c[k * slots:(k + 1) * slots].tolist()))[0]
    expectation, cls, _logit, _re, _im = _run_circuit(features, weights)
    logit = value * float(weights["scale"]) + float(weights["bias"])
    return {"readout": float(value), "class": int(logit > 0), "clear_class": int(cls), "max_error": abs(float(value) - expectation), "n_terms": int(c.size), "n_ciphertexts": chunks}


def fhe_note() -> str:
    """What the run reports about the parameters it used, from the config rather than a literal."""
    fhe = get_config().fhe
    return f"N = {fhe.poly_modulus_degree}, moduli {tuple(fhe.coeff_mod_bit_sizes)}, scale 2^{fhe.global_scale_bits}"

"""Encrypted (CKKS) inference for a linear classifier, one layer ``W x + b``.

The client encrypts its feature vector once; for each output row ``w_j`` the evaluator computes
the encrypted dot product ``<w_j, x>`` with ``dot_plain`` (a ciphertext-plaintext multiply and a
rotate-and-sum), the result is decrypted, and the bias ``b_j`` is added in the clear. Nothing else is
evaluated: there is no activation here, polynomial or otherwise, and the argmax over the decrypted
logits is taken in plaintext. The hybrid that also evaluates non-linear layers, under MPC, is
:mod:`qsh.mpc.secret_inference`. This is the linear CKKS path the classical MPC-FHE benchmark
(:mod:`qsh.benchmark.classical_mpc_fhe`) runs against the quantum QSMPC-FHE path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backends import CKKSBackend, get_backend


@dataclass
class LinearModel:
    """A plaintext linear classifier W x + b (logits). Owned by the server."""

    weight: np.ndarray   # (out, in)
    bias: np.ndarray     # (out,)

    @property
    def n_in(self) -> int:
        return self.weight.shape[1]

    @property
    def n_out(self) -> int:
        return self.weight.shape[0]


def encrypted_linear_inference(
    model: LinearModel, x: np.ndarray, backend: CKKSBackend | None = None
) -> dict:
    """Run W x + b under CKKS encryption; return decrypted logits + the plaintext ref.

    Returns a dict with ``logits`` (decrypted), ``plain_logits`` (reference) and the
    max absolute error introduced by the homomorphic evaluation.
    """
    backend = backend or get_backend("tenseal")
    x = np.asarray(x, dtype=float).ravel()[: model.n_in]
    ct = backend.encrypt(x)
    logits = []
    for j in range(model.n_out):
        row = model.weight[j]
        enc_dot = backend.dot_plain(ct, row)        # encrypted <w_j, x>
        logits.append(backend.decrypt(enc_dot)[0] + float(model.bias[j]))
    plain = (model.weight @ x + model.bias).tolist()
    err = float(np.max(np.abs(np.array(logits) - np.array(plain)))) if plain else 0.0
    return {
        "backend": backend.name,
        "logits": logits,
        "plain_logits": plain,
        "max_abs_error": err,
        "prediction": int(np.argmax(logits)),
    }

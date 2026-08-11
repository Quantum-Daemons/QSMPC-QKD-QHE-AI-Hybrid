"""Trained variational QCNN inference for the Python pipeline.

WHY THIS EXISTS. Round 25 found that the browser's QCNN backend surfaced a class carrying no
information about its input, and fixed it there. The Python path had the same problem and was not
touched, so the fix reached only half the codebase. Reaching the Python path is not hypothetical:
``api/main.py`` passes ``config`` straight through and the Parameters panel offers ``qcnn``, so a
full-deployment run selects it exactly as a browser run does.

The two failures are NOT identical, and the difference matters:

    browser   ``qotpQcnnInfer`` applied a QOTP mask it never undid, and the class flipped with the
              run seed for 100% of inputs. Visibly random.
    python    ``QCNN.predict_proba`` never exceeds ~0.24 over 200 random inputs at 4 or 8 qubits,
              so it never crosses the 0.5 decision boundary and the class is **0 for every input**,
              at every seed. Correlation with the input is -0.05.

A coin flip looks broken. A CONSTANT PREDICTOR looks confident and reproducible, which is why it
survived longer - and it is a shape this project has been bitten by before, in the medical model
that scored exactly the majority-class rate.

WHAT THIS MODULE DOES. Loads the weights fitted by ``scripts/train/qcnn_variational.py`` and
evaluates the same circuit. The implementation is deliberately the plain-numpy one that was used as
the INDEPENDENT reimplementation when the trainer was verified: it agreed with the torch training
circuit to 2.4e-07 after a crossed-CNOT bug in the trainer was found by exactly that comparison. Two
implementations that were checked against each other are worth more than one that was not.

WHAT IT DOES NOT DO. No QOTP mask on this path. Undoing a Pauli frame through Clifford+T is not
implemented anywhere in this repository - ``T X T-dagger`` is not Pauli - so hiding and prediction
stay separate rather than claiming a correction that does not exist. ``qhe/qotp.py`` remains the
hiding demonstration and proves its own claim properly.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]

#: Use case -> trained-weights file. A use case absent here has no trained circuit, and the caller
#: must decline to surface a class rather than reporting an untrained one.
_WEIGHTS = {"medical_fl": _REPO / "models" / "qcnn_medical" / "weights.json",
            "finance_aml": _REPO / "models" / "qcnn_finance" / "weights.json"}


@lru_cache(maxsize=8)
def load_weights(usecase: str) -> dict | None:
    """Trained weights for ``usecase``, or None when none have been fitted."""
    path = _WEIGHTS.get(usecase)
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def infer(features, weights: dict) -> tuple[float, int, float]:
    """Return ``(expectation, cls, logit)`` for one feature vector.

    Mirrors ``scripts/train/qcnn_variational.py`` gate for gate: RY(tanh(z)*pi) data encoding, then
    ``layers`` rounds of trainable RY plus a CNOT ring, then <Z0>.

    The tanh encoding is not cosmetic. The features arriving here are z-scores, and the first
    version of the trained model min-max scaled them against RAW units instead - which pinned seven
    of eight features to -1 and produced a degenerate state, making the TRAINED path exactly as
    meaningless as the untrained one it replaced. tanh bounds an unbounded z-score without
    reordering it.
    """
    nq = int(weights["n_qubits"])
    theta = np.asarray(weights["theta"], dtype=float)
    n = 1 << nq
    re = np.zeros(n)
    im = np.zeros(n)
    re[0] = 1.0

    idx0 = {q: np.flatnonzero((np.arange(n) & (1 << q)) == 0) for q in range(nq)}

    def ry(q: int, th: float) -> None:
        c, s = np.cos(th / 2.0), np.sin(th / 2.0)
        i0 = idx0[q]
        i1 = i0 | (1 << q)
        r0, m0, r1, m1 = re[i0].copy(), im[i0].copy(), re[i1].copy(), im[i1].copy()
        re[i0] = c * r0 - s * r1
        im[i0] = c * m0 - s * m1
        re[i1] = s * r0 + c * r1
        im[i1] = s * m0 + c * m1

    def cnot(ctl: int, tgt: int) -> None:
        i = np.arange(n)
        sel = ((i & (1 << ctl)) != 0) & ((i & (1 << tgt)) == 0)
        src = i[sel]
        dst = src | (1 << tgt)
        re[src], re[dst] = re[dst].copy(), re[src].copy()
        im[src], im[dst] = im[dst].copy(), im[src].copy()

    x = np.asarray(features, dtype=float).ravel()
    # The caller's units differ per use case, and assuming one of them is how the first trained model
    # became as meaningless as the untrained one. medical arrives already StandardScaler-ed; finance
    # arrives RAW and mixed-scale (log amount, a z-score, five binary flags), so it must be
    # standardised here with the moments fitted at training time.
    if weights.get("input_space") == "raw":
        mu = np.asarray(weights["feature_mean"], dtype=float)
        sd = np.asarray(weights["feature_std"], dtype=float)
        sd = np.where(sd == 0, 1.0, sd)
        x = (x[:mu.size] - mu) / sd
    for q in range(nq):
        ry(q, float(np.tanh(x[q] if q < x.size else 0.0)) * np.pi)
    for layer in range(int(weights["layers"])):
        for q in range(nq):
            ry(q, float(theta[layer, q]))
        for q in range(nq):
            cnot(q, (q + 1) % nq)

    p = re * re + im * im
    sign = np.where((np.arange(n) & 1) == 0, 1.0, -1.0)
    expectation = float((p * sign).sum())
    logit = expectation * float(weights["scale"]) + float(weights["bias"])
    return expectation, int(logit > 0), logit

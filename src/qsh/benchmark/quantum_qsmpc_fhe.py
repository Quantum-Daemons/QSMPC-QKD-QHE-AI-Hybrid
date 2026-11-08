"""Quantum QSMPC-FHE comparison row (simulated; not a private inference).

Evaluates the quantum path: an untrained PennyLane QCNN-style circuit (parameterised RY/RZ/CNOT)
run in PLAINTEXT, beside a separate QOTP hiding check on a tiny fixed state - nothing is evaluated
homomorphically and nothing is decrypted (see :func:`qsh.qhe.private_inference.private_inference`).
It records latency, an ESTIMATED T-gate budget, qubit overhead and the security model of the
scheme it simulates.

CORRECTED ROUND 55, AND THE CORRECTION HAD ALREADY BEEN MADE ELSEWHERE. This said "Clifford+T QCNN"
and "T-gate count". `qsh/qhe/qcnn.py` builds no T gates at all - the circuit is RY/RZ rotations and
CNOT entanglers - and `t_count()` is a Solovay-Kitaev BUDGET, `ceil(3*log2(1/eps))` per rotation
times the parameter count, which its own docstring calls "rough" and "-ish". At 8 qubits it returns
630, a number no gate list contains. The technical report's data script recorded this correction as *fixed* and
`docs/deliverable-verification.md` names README and architecture.md as the two files corrected. This
module was the third copy, and its docstring is what a reader of this code sees first.

The browser row is different and is genuinely counted: 56 T-gates from an actual gate list. So the
two rows of one table mean different things by "T-gates", which `BenchmarkPanel` now says.

CORRECTED 2026-10-23. This said the information-theoretic model "the classical RLWE path cannot match"
and that the Rate-1 QFHE ratio is measured "to contrast communication efficiency". The one-time pad is
information-theoretically hiding as a SCHEME, but this run's pad is numpy-generated and wrapped by
ML-KEM-768 + X25519, so the run is computational, like the RLWE path. And the ratio below is in bytes of
a simulated state vector, which no CKKS byte ratio can be compared with; classical FHE reaches rate 1 too
(Gentry and Halevi, TCC 2019; Brakerski, Doettling, Garg and Malavolta, TCC 2019).
"""

from __future__ import annotations

import time

import numpy as np

from ..qhe.private_inference import private_inference
from ..qhe.qcnn import QCNN
from .metrics import BenchResult

#: The QSMPC-FHE row's security label, defined ONCE because it is read twice below and parsed for the
#: substring "information-theoretic" by the security scorer
#: (`tests/test_security_model_prose_is_not_a_silent_score_input.py`). It used to read
#: "information-theoretic (QOTP perfect hiding)" with the backend "perfectly-secure QHE", and neither
#: holds for a run of this code: the quantum one-time pad is information-theoretically hiding AS A
#: SCHEME, given a uniform secret pad (Ambainis, Mosca, Tapp, de Wolf, FOCS 2000), but here the pad
#: comes from numpy's default generator and travels under ML-KEM-768 + X25519, so the composite is
#: computational (MLWE, FIPS 203). The substring is kept so no (unrendered) score moves.
QSMPC_SECURITY_MODEL = (
    "information-theoretic as a scheme (QOTP); not this run - numpy pad, ML-KEM-768 + X25519 wrap")


def run_quantum(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_qubits: int = 8,
    workload: str = "qcnn",
    n_samples: int = 20,
    seed: int = 0,
) -> BenchResult:
    """Benchmark the untrained QCNN-style circuit over ``n_samples``, evaluated in plaintext beside a
    QOTP hiding check (no homomorphic evaluation; see the module docstring).

    It took the transport's KEM, QKD key bits and QBER until Round 86, for the 0-100 security score alone;
    the score is retired, so the row carries no transport facts it did not measure.
    """
    qc = QCNN(n_qubits=n_qubits, seed=seed)
    n = min(n_samples, len(X))
    t0 = time.perf_counter()
    rep = None
    for i in range(n):
        feats = np.asarray(X[i], dtype=float).ravel()[:n_qubits]
        rep = private_inference(qc, feats, verify_hidden=(i == 0))
    dt = time.perf_counter() - t0
    # THE `correct` COUNTER THAT USED TO LIVE HERE WAS `correct += 1` INSIDE THE LOOP, UNCONDITIONAL
    # AND NEVER READ, and it fed `accuracy=1.0` below. There is nothing here to be accurate about:
    # the QCNN carries no trained weights, and `private_inference` says so itself - "Nothing is
    # decrypted and nothing is compared" - and names the probability-versus-plaintext comparison a
    # tautology. What this function legitimately measures is latency, throughput, the counted T-gate
    # total, the qubit count and the security model.

    # Rate-1 QFHE communication ratio for a representative state
    from ..pqc.hybrid_kem import HybridKEMRecipient
    from ..qhe.rate1_qfhe import ciphertext_to_plaintext_ratio, encrypt_state

    with HybridKEMRecipient() as rcpt:
        psi = np.random.default_rng(seed).normal(size=2 ** min(n_qubits, 12)) + 0j
        psi /= np.linalg.norm(psi)
        ct, _, _ = encrypt_state(psi, rcpt.public_key)
        rate = ciphertext_to_plaintext_ratio(ct)

    return BenchResult(
        path="QSMPC-FHE",
        backend="qotp+qcnn (state-vector simulation; QOTP pad numpy-generated)",
        workload=workload,
        latency_ms=round(dt / n * 1000, 3),
        throughput_ips=round(n / dt, 2),
        # NOT 1.0. The browser twin was corrected to `accuracy: null` with a comment recording that
        # it "reported 1.0 for three rounds while the MPC-FHE row eight lines above carries a comment
        # forbidding exactly that". That fix reached ONE LANGUAGE, and this is the transport the full
        # deployment serves - `usePipelineRun.ts` PREFERS `st.artifacts.benchmark_result` - so the
        # identical row on the identical page printed 1.0 here and an em dash there.
        accuracy=None,
        ciphertext_plaintext_ratio=round(rate, 4),
        t_gate_count=qc.t_count(),
        qubits=n_qubits,
        security_model=QSMPC_SECURITY_MODEL,
        device="cpu-sim",
        extra={"samples": n, "qcnn_params": qc.n_params,
               "input_hidden": bool(rep.input_hidden) if rep else None,
               # The ANALOGY this row draws, not what this run does: in a QHE scheme Clifford
               # gates would be the cheap "linear" layer and T-gates the expensive "non-linear"
               # one, with no FHE-to-MPC boundary because both act on the encrypted state. Here
               # nothing is evaluated on an encrypted state (the circuit runs in plaintext beside
               # a QOTP hiding check), the T figure is a Solovay-Kitaev ESTIMATE, and the finance
               # pipeline's PSI is a classical OPRF PSI (RFC 9497), not QOT-based secure MPC.
               "linear_analogue": "Clifford gates",
               "nonlinear_analogue": f"T-gates (×{qc.t_count()}, the expensive gates in this analogy)",
               "no_fhe_mpc_boundary": True},
    )

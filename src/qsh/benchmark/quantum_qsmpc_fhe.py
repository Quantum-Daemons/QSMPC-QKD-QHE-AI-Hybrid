"""Quantum QSMPC-FHE private-inference benchmark path.

Evaluates the quantum path: a QHE-protected QCNN private inference (QOTP + Clifford+T
QCNN), recording latency, T-gate count, qubit overhead and -- crucially -- the
*information-theoretic* security model (QOTP perfect hiding), which the classical
RLWE path cannot match. The Rate-1 QFHE ciphertext/plaintext ratio is also measured
to contrast communication efficiency.
"""

from __future__ import annotations

import time

import numpy as np

from ..qhe.private_inference import private_inference
from ..qhe.qcnn import QCNN
from .metrics import BenchResult


def run_quantum(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_qubits: int = 8,
    workload: str = "qcnn",
    n_samples: int = 20,
    seed: int = 0,
    kem: str | None = None,
    qkd_key_bits: int | None = None,
    qber: float | None = None,
) -> BenchResult:
    """Benchmark QHE private inference of a QCNN over ``n_samples``.

    ``kem`` / ``qkd_key_bits`` / ``qber`` describe the TRANSPORT the caller ran under. They are
    parameters rather than literals because this function has no QKD channel in it - nothing here
    produces a QBER - and hardcoding one made an unmeasured term look measured. ``None`` means
    "not measured", which ``compute_security_score`` already scores as zero rather than as good.
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

    from .security import compute_security_score

    # `qber=0.0128` and `qkd_key_bits=256` were literals here. 0.0128 is the same fabricated value
    # that was removed from the browser mirror in Round 22: it bought the full eavesdropper-detection
    # points on every run, including runs whose channel produced no key, and the classical row it is
    # compared against passed `qber=None`. That asymmetry - not the cryptography - was most of the
    # 68-vs-96 gap the benchmark reported. The leakage bound is a LITERATURE figure (PRX 15 041056),
    # not a property of this run, so it is no longer asserted as one.
    sec = compute_security_score(
        security_model="information-theoretic (QOTP perfect hiding)",
        pqc_hybrid=True, kem=kem or "ml-kem-768+x25519",
        qkd_key_bits=qkd_key_bits, qber=qber, leakage_bits_per_weight=None,
    )
    return BenchResult(
        path="QSMPC-FHE",
        backend="qotp+qcnn (perfectly-secure QHE)",
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
        security_model="information-theoretic (QOTP perfect hiding)",
        security_score=sec["total"],
        device="cpu-sim",
        extra={"samples": n, "qcnn_params": qc.n_params,
               "input_hidden": bool(rep.input_hidden) if rep else None,
               # quantum analogue of the FHE-linear / MPC-non-linear split: Clifford gates are
               # the cheap "linear" layer, T-gates the expensive "non-linear" one. Unlike the
               # classical hybrid there is NO FHE↔MPC boundary — QOTP/Rate-1 QFHE evaluate both
               # on the encrypted state (ct/pt≈1), and the QSMS/QOT path is the quantum secure-MPC.
               "linear_analogue": "Clifford gates",
               "nonlinear_analogue": f"T-gates (×{qc.t_count()}, the non-linear cost)",
               "no_fhe_mpc_boundary": True,
               "security_components": sec["components"]},
    )

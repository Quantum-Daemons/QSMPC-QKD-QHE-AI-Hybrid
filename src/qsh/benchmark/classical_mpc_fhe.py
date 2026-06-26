"""Classical MPC-FHE private-inference benchmark path.

Evaluates a small model's linear layer under CKKS encryption (TenSEAL CPU and/or
OpenFHE CPU) and records latency, throughput, accuracy preservation and the
ciphertext/plaintext ratio. This is the "classical" baseline the quantum QSMPC-FHE
path is compared against. All backends are commercial-OK (TenSEAL Apache-2.0,
OpenFHE BSD-2-Clause, HEonGPU Apache-2.0); DESILO was removed (non-commercial license).

The optional high-performance GPU path (upstream HEonGPU + PIGEON/hpmpc, driven by
our own glue in :mod:`~qsh.benchmark.gpu_mpc_adapter`) is included when those
submodules are built; otherwise the CPU backends (TenSEAL / OpenFHE) are used.
"""

from __future__ import annotations

import time

import numpy as np

from ..fhe.backends import get_backend
from ..fhe.ckks_inference import LinearModel, encrypted_linear_inference
from .metrics import BenchResult


def _ciphertext_ratio(backend, n_features: int) -> float | None:
    """Estimate the CKKS ciphertext/plaintext size ratio for one encrypted vector."""
    try:
        ct = backend.encrypt(np.zeros(n_features))
        if hasattr(ct, "serialize"):
            ct_bytes = len(ct.serialize())
        else:
            return None
        plain_bytes = n_features * 8
        return ct_bytes / plain_bytes
    except Exception:
        return None


def run_classical(
    model: LinearModel,
    X: np.ndarray,
    y: np.ndarray,
    *,
    backend_name: str = "tenseal",
    workload: str = "linear",
    n_samples: int = 30,
) -> BenchResult:
    """Benchmark encrypted linear inference over ``n_samples`` of (X, y)."""
    backend = get_backend(backend_name)
    n = min(n_samples, len(X))
    correct = 0
    errs = []
    t0 = time.perf_counter()
    for i in range(n):
        out = encrypted_linear_inference(model, X[i], backend)
        errs.append(out["max_abs_error"])
        if out["prediction"] == int(y[i]):
            correct += 1
    dt = time.perf_counter() - t0

    device = "gpu" if getattr(backend, "on_gpu", False) else "cpu"
    from .security import compute_security_score

    sec = compute_security_score(
        security_model="computational (RLWE / FIPS-203 family)",
        pqc_hybrid=True, kem="ml-kem-768+x25519", qber=None,
    )
    return BenchResult(
        path="MPC-FHE",
        backend=backend.name,
        workload=workload,
        latency_ms=round(dt / n * 1000, 3),
        throughput_ips=round(n / dt, 2),
        accuracy=round(correct / n, 4),
        max_abs_error=float(np.max(errs)),
        ciphertext_plaintext_ratio=_ciphertext_ratio(backend, model.n_in),
        security_model="computational (RLWE / FIPS-203 family)",
        security_score=sec["total"],
        device=device,
        extra={"samples": n, "security_components": sec["components"]},
    )


def _first_feature_count(specs) -> int:
    """Feature count of the first encrypted operand (dense input width, or a conv patch length)."""
    from ..mpc.layers import ConvSpec, DenseSpec

    first = specs[0]
    if isinstance(first, ConvSpec):
        return int(first.weight.shape[1] * first.weight.shape[2] * first.weight.shape[3])
    if isinstance(first, DenseSpec):
        return int(first.weight.shape[1])
    return 8


def run_hybrid(layers, X, y, *, backend_name: str = "tenseal", workload: str = "linear",
               model_size: str = "medium", n_samples: int = 8) -> BenchResult:
    """The faithful MPC-FHE hybrid: linear layers under CKKS (FHE), non-linear (ReLU/argmax/maxpool)
    under MPyC secret-sharing (MPC). Reports the per-stage split + estimated MPC communication.

    ``layers`` may be a legacy list of ``(W, b)`` (a dense MLP) **or** a list of conv/dense specs
    (:mod:`qsh.mpc.layers`) for a genuine CNN; reuses :func:`qsh.mpc.hybrid_private_inference`.
    """
    from ..mpc import hybrid_private_inference, normalize_layers
    from .security import compute_security_score

    specs = normalize_layers(layers)
    backend = get_backend(backend_name)
    n = min(n_samples, len(X))
    fhe = mpc = comm = 0.0
    correct = cmp_total = 0
    arch_class = "mlp"
    t0 = time.perf_counter()
    for i in range(n):
        r = hybrid_private_inference(layers, X[i], backend, use_mpc=True)
        fhe += r["fhe_linear_ms"]; mpc += r["mpc_nonlinear_ms"]
        comm += r["mpc_comm_kb_estimated"]; cmp_total += r["mpc_comparisons"]
        correct += int(r["decision_matches_plaintext"])
        arch_class = r.get("arch_class", arch_class)
    dt = (time.perf_counter() - t0) * 1000
    device = "gpu" if getattr(backend, "on_gpu", False) else "cpu"
    sec = compute_security_score(
        security_model="computational (RLWE) + secret-sharing MPC",
        pqc_hybrid=True, kem="ml-kem-768+x25519", qber=None)
    return BenchResult(
        path="MPC-FHE",
        backend=f"{backend.name} + mpyc-shamir",
        workload=workload, model_size=model_size,
        latency_ms=round(dt / n, 3), throughput_ips=round(n / (dt / 1000), 2),
        accuracy=round(correct / n, 4),                       # decision vs plaintext (correctness)
        ciphertext_plaintext_ratio=_ciphertext_ratio(backend, _first_feature_count(specs)),
        security_model="computational (RLWE) + secret-sharing MPC",
        security_score=sec["total"],
        fhe_linear_ms=round(fhe / n, 3), mpc_nonlinear_ms=round(mpc / n, 3),
        mpc_comm_kb=round(comm / n, 3), device=device,
        extra={"samples": n, "avg_comparisons": round(cmp_total / n, 1),
               "arch_class": arch_class,
               "linear_under": "FHE/CKKS", "nonlinear_under": "MPC/mpyc-shamir",
               "security_components": sec["components"]},
    )

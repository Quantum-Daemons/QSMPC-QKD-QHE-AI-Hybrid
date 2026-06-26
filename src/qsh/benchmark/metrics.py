"""Benchmark metric types + helpers.

Captures, for each private-inference path, the dimensions the report's benchmark
chapter compares: latency, throughput, accuracy preservation, communication
(ciphertext/plaintext ratio), quantum resource (T-gates, qubits), and the security
model (computational vs information-theoretic).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field


@dataclass
class BenchResult:
    """One benchmarked private-inference configuration."""

    path: str                      # "MPC-FHE" | "QSMPC-FHE"
    backend: str                   # tenseal-cpu / openfhe-cpu / heongpu-gpu / qotp-qcnn ...
    workload: str                  # cnn / transformer / linear
    latency_ms: float = 0.0
    throughput_ips: float = 0.0    # inferences per second
    accuracy: float | None = None  # vs plaintext (1.0 = identical decision)
    max_abs_error: float | None = None
    ciphertext_plaintext_ratio: float | None = None
    t_gate_count: int | None = None
    qubits: int | None = None
    security_model: str = ""       # "computational (RLWE)" | "information-theoretic"
    security_score: float | None = None   # 0-100 (see benchmark/security.py)
    device: str = "cpu"
    # hybrid split (2026 standard): linear→FHE (GPU-capable), non-linear→MPC (secret-sharing/CPU)
    model_size: str | None = None          # lightweight | medium | large
    model_arch: str | None = None          # per-use-case architecture (e.g. "tabular fraud MLP")
    model_params: int | None = None        # parameter count of the benchmarked model
    fhe_linear_ms: float | None = None      # linear layers under CKKS
    mpc_nonlinear_ms: float | None = None   # ReLU/argmax under secret-sharing MPC
    mpc_comm_kb: float | None = None        # estimated MPC communication
    projected: bool = False                 # True = extrapolated (e.g. the 'large' tier), not measured
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return asdict(self)


@contextmanager
def timer():
    """Context manager yielding a callable that returns elapsed ms."""
    t0 = time.perf_counter()
    elapsed = {}
    try:
        yield lambda: (time.perf_counter() - t0) * 1000.0
    finally:
        elapsed["ms"] = (time.perf_counter() - t0) * 1000.0

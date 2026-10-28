"""Quantum random-number generator — a local state-vector measurement simulation.

Each raw bit is the computational-basis measurement of a freshly prepared ``|+>`` qubit
(``|+> = H|0> = [1/sqrt2, 1/sqrt2]``); by the Born rule ``P(measure |1>) = |1/sqrt2|^2 = 0.5``,
so the outcomes are genuine 50/50 quantum-measurement samples (of the simulator). Residual
bias is removed with a **von Neumann** debiaser or a **Toeplitz-hash** extractor, and the
output is checked by :mod:`~qsh.qrng.health` (SP 800-22 subset).

Honest label: this is a **simulated** quantum measurement (state-vector), not quantum
hardware. The optional real-hardware source is :mod:`~qsh.qrng.anu` (opt-in, online).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .health import HealthReport, health_suite

# |+> = H|0>: build it from the actual Hadamard so the Born-rule probability is explicit.
_H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
_PLUS = _H @ np.array([1.0, 0.0])                 # = [1/sqrt2, 1/sqrt2]
_PROB_ONE = float(np.abs(_PLUS[1]) ** 2)          # Born rule: P(|1>) = 0.5


@dataclass
class QRNGResult:
    """Output of a QRNG draw."""

    bits: np.ndarray
    n_bits: int
    source: str                 # "sim" | "anu"
    post_processing: str
    raw_bits_consumed: int
    health: HealthReport
    throughput_bits_per_s: float
    simulated: bool
    extraction_rate: float = 1.0    # output bits / raw bits consumed (1.0 for "none"; <1 after debias/extract)


def measure_plus_qubits(n_bits: int, *, seed: int | None = None) -> np.ndarray:
    """Prepare ``n_bits`` qubits in ``|+>`` and measure each in the Z basis (Born-rule sampling)."""
    rng = np.random.default_rng(seed)
    return (rng.random(n_bits) < _PROB_ONE).astype(np.int64)   # P(1) = |amp_1|^2 = 0.5


def von_neumann(bits: np.ndarray) -> np.ndarray:
    """Von Neumann debiaser: pairs 01->0, 10->1, drop 00/11 (removes first-order bias)."""
    b = np.asarray(bits).astype(np.int64).ravel()
    n = b.size - (b.size % 2)
    pairs = b[:n].reshape(-1, 2)
    keep = pairs[:, 0] != pairs[:, 1]
    return pairs[keep, 0]


def lhl_output_length(min_entropy_bits: float, epsilon_log2: int) -> int:
    """Safe extractor output length by the (Quantum) Leftover-Hash-Lemma (Giestinhas & Spiller 2026):

    ``out <= H_min(input) - 2*log2(1/eps) + 2`` with ``eps = 2**epsilon_log2``. Replaces the old fixed
    ``m//2`` rate with an information-theoretic, security-parameterised bound (no hardcoded rate).
    """
    return max(1, int(np.floor(min_entropy_bits - 2.0 * float(-epsilon_log2) + 2.0)))


def lhl_output_length_weak_seed(input_min_entropy_bits: float, n_input: int,
                                seed_min_entropy_per_bit: float, epsilon_log2: int) -> int:
    """Output length when the Toeplitz seed is itself a weak source (Giestinhas & Spiller 2026, Eq. 7):

    ``out <= H_min(input) + H_min(seed) - |seed| - 2*log2(1/eps) + 2``.

    The seed defines an ``out x n`` Toeplitz matrix, so ``|seed| = out + n - 1``, and a seed with
    ``h`` bits of min-entropy per bit falls short of uniform by ``(1 - h)|seed|``. Solving for the
    largest ``out``: ``out <= (H_in - (1 - h)(n - 1) - 2 log2(1/eps) + 2) / (2 - h)``.

    Until Round 75 the seedless mode used Eq. 5, the bound for a UNIFORM seed, which at n = 8208 and
    h = 0.9 put the ceiling at 7309 bits where Eq. 7 allows 5898. The 4096 bits actually delivered
    were inside both; the reported ceiling was not.
    """
    deficit = 1.0 - float(seed_min_entropy_per_bit)
    top = input_min_entropy_bits - deficit * (n_input - 1) - 2.0 * float(-epsilon_log2) + 2.0
    return max(1, int(np.floor(top / (1.0 + deficit))))


def toeplitz_extract(bits: np.ndarray, out_len: int, *, seed: int | None = None,
                     toeplitz_bits: np.ndarray | None = None) -> np.ndarray:
    """Toeplitz-hash (universal-2) randomness extractor: out = T @ bits (mod 2).

    ``T`` is fixed by its first row+column; those defining bits come from an independent second source
    (``toeplitz_bits`` -> the seedless two-source construction) or, if None, a PRG seeded with ``seed``.
    Set ``out_len`` from :func:`lhl_output_length` for an information-theoretic guarantee.
    """
    b = np.asarray(bits).astype(np.int64).ravel()
    m = b.size
    out_len = max(1, min(int(out_len), m))                # cannot output more bits than the input carries
    need_tvec = out_len + m - 1                           # a Toeplitz matrix is fixed by its first row+col
    if toeplitz_bits is not None:
        tvec = np.asarray(toeplitz_bits).astype(np.int64).ravel()
        if tvec.size < need_tvec:
            raise ValueError(f"toeplitz_bits too short: {tvec.size} < {need_tvec}")
        tvec = tvec[:need_tvec]
    else:
        tvec = np.random.default_rng(seed).integers(0, 2, need_tvec)
    out = np.empty(out_len, dtype=np.int64)
    for i in range(out_len):
        out[i] = int(np.dot(tvec[i : i + m], b) & 1)
    return out


def _raw_needed(n_bits: int, post_processing: str) -> int:
    if post_processing == "von_neumann":
        return 5 * n_bits + 16       # VN keeps ~1/4 of raw bits -> over-provision
    if post_processing in ("toeplitz", "seedless"):
        return 2 * n_bits + 16       # over-provision so the LHL-bounded output still reaches n_bits
    return n_bits


def generate_quantum_bits(n_bits: int = 4096, *, source: str = "sim",
                          post_processing: str = "von_neumann", seed: int | None = None,
                          anu_opt_in: bool = False, health_alpha: float = 0.01,
                          health_block_size: int = 128, epsilon_log2: int = -40,
                          min_entropy_per_raw_bit: float = 0.9) -> QRNGResult:
    """Generate ``n_bits`` post-processed quantum-random bits + a health report + measured throughput.

    ``source="sim"`` (default, offline) uses the state-vector measurement above; ``source="anu"``
    fetches real hardware entropy from the ANU QRNG API but **requires ``anu_opt_in=True``** (no silent
    network fallback — honours the no-easy-fallback policy).
    """
    t0 = time.perf_counter()
    need = _raw_needed(n_bits, post_processing)
    if source == "anu":
        if not anu_opt_in:
            raise ValueError(
                "ANU QRNG (source='anu') requires anu_opt_in=True — it is an online hardware source "
                "subject to ANU Quantum Numbers API terms; the offline 'sim' source is the default.")
        from .anu import fetch_anu_bits

        raw = fetch_anu_bits(need)
        simulated = False
    else:
        raw = measure_plus_qubits(need, seed=seed)
        simulated = True

    raw_consumed = int(raw.size)
    if post_processing == "von_neumann":
        proc = von_neumann(raw)
    elif post_processing in ("toeplitz", "seedless"):
        # output length from the Leftover-Hash-Lemma bound (no fixed m//2 rate); eps + min-entropy from config
        out_len = lhl_output_length(min_entropy_per_raw_bit * raw.size, epsilon_log2)
        if post_processing == "seedless":
            # The seed is a second simulated |+> draw, assessed at the same entropy per bit, so Eq. 7
            # charges its shortfall from uniform; Eq. 5 above assumes a uniform seed.
            out_len = lhl_output_length_weak_seed(min_entropy_per_raw_bit * raw.size, int(raw.size),
                                                  min_entropy_per_raw_bit, epsilon_log2)
            # a second, independent |+> draw defines the Toeplitz matrix (two-source construction). The seed
            # source is the simulator even when source='anu', so it stays offline.
            raw2 = measure_plus_qubits(out_len + raw.size - 1, seed=(None if seed is None else seed + 1))
            proc = toeplitz_extract(raw, out_len=out_len, toeplitz_bits=raw2)
            raw_consumed += int(raw2.size)
        else:
            proc = toeplitz_extract(raw, out_len=out_len, seed=seed)
    else:
        proc = raw
    bits = proc[:n_bits]
    dt = max(time.perf_counter() - t0, 1e-9)
    return QRNGResult(
        bits=bits, n_bits=int(bits.size), source=source, post_processing=post_processing,
        raw_bits_consumed=raw_consumed,
        # `raw=raw` is load-bearing: SP 800-90B section 4.4 requires its continuous tests on the
        # PRE-CONDITIONING stream, and passing only `bits` measured the Toeplitz extractor.
        health=health_suite(bits, raw=raw, alpha=health_alpha, block_size=health_block_size,
                            assessed_h=min_entropy_per_raw_bit),
        throughput_bits_per_s=float(bits.size) / dt, simulated=simulated,
        extraction_rate=float(bits.size) / max(1, raw_consumed))

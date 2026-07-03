"""Randomness health suite (a compact NIST SP 800-22 subset).

Three of the SP 800-22 statistical tests — **monobit (frequency)**, **runs**, and
**block-frequency** — computed with exact reference p-values (``erfc`` / regularized
upper incomplete gamma), enough to give an honest pass/fail on a QRNG bit-stream.
Not the full 15-test battery; labelled as a *health check*, not a certification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc, gammaincc


@dataclass
class HealthReport:
    """Result of the randomness health suite (p >= alpha => pass for that test)."""

    monobit_p: float
    runs_p: float
    block_frequency_p: float
    passed: bool
    alpha: float
    min_entropy_per_bit: float = 0.0   # SP 800-90B most-common-value estimate: bits of min-entropy per output bit
    min_entropy_bits: float = 0.0      # = min_entropy_per_bit * n (total estimated min-entropy)


def _monobit(bits: np.ndarray) -> float:
    """SP 800-22 frequency (monobit) test p-value."""
    n = bits.size
    if n == 0:
        return 0.0
    s = float(np.sum(2 * bits.astype(np.int64) - 1))
    s_obs = abs(s) / np.sqrt(n)
    return float(erfc(s_obs / np.sqrt(2.0)))


def _runs(bits: np.ndarray) -> float:
    """SP 800-22 runs test p-value (guarded by the monobit precondition)."""
    n = bits.size
    if n < 2:
        return 0.0
    pi = float(np.mean(bits))
    if abs(pi - 0.5) >= (2.0 / np.sqrt(n)):     # monobit precondition failed -> runs test not applicable
        return 0.0
    v_obs = 1 + int(np.sum(bits[1:] != bits[:-1]))
    num = abs(v_obs - 2.0 * n * pi * (1 - pi))
    den = 2.0 * np.sqrt(2.0 * n) * pi * (1 - pi)
    return float(erfc(num / den)) if den > 0 else 0.0


def _block_frequency(bits: np.ndarray, block_size: int) -> float:
    """SP 800-22 block-frequency test p-value."""
    n = bits.size
    n_blocks = n // block_size
    if n_blocks == 0:
        return 1.0
    blocks = bits[: n_blocks * block_size].reshape(n_blocks, block_size)
    pi = blocks.mean(axis=1)
    chi_sq = 4.0 * block_size * float(np.sum((pi - 0.5) ** 2))
    return float(gammaincc(n_blocks / 2.0, chi_sq / 2.0))


# 99% one-sided confidence Z for the SP 800-90B most-common-value bound (intentionally fixed spec constant).
_MCV_Z = 2.5758293035489004


def _min_entropy_mcv(bits: np.ndarray) -> float:
    """SP 800-90B (6.3.1) most-common-value min-entropy estimate: bits of min-entropy per bit.

    Uses the 99% upper confidence bound on the most-common-value probability ``p_max`` (so a perfectly
    unbiased stream yields ~1.0 and a biased stream less) -> a conservative, honest entropy read that
    bridges the empirical p-value tests to the information-theoretic extractor bound (:func:`lhl_output_length`).
    """
    n = int(bits.size)
    if n == 0:
        return 0.0
    ones = int(np.count_nonzero(bits))
    p_hat = max(ones, n - ones) / n
    p_upper = min(1.0, p_hat + _MCV_Z * float(np.sqrt(p_hat * (1.0 - p_hat) / n)))
    return float(-np.log2(p_upper)) if p_upper > 0 else 0.0


def health_suite(bits, *, alpha: float = 0.01, block_size: int = 128) -> HealthReport:
    """Run the monobit + runs + block-frequency tests + a min-entropy estimate; pass iff every p-value >= alpha."""
    b = np.asarray(bits).astype(np.int64).ravel()
    mp = _monobit(b)
    rp = _runs(b)
    bp = _block_frequency(b, block_size)
    h = _min_entropy_mcv(b)
    passed = (mp >= alpha) and (rp >= alpha) and (bp >= alpha)
    return HealthReport(monobit_p=mp, runs_p=rp, block_frequency_p=bp, passed=passed, alpha=alpha,
                        min_entropy_per_bit=h, min_entropy_bits=h * float(b.size))

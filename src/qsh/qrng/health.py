"""Entropy-source health testing: NIST SP 800-90B continuous tests + an SP 800-22 output check.

These are two different jobs and the standard is explicit that one cannot do the other's:

**SP 800-90B section 4 — health testing the NOISE SOURCE.** This is what an entropy source is
*required* to do. Section 4.4 defines two approved continuous tests, Repetition Count and Adaptive
Proportion, which run on raw samples **before any conditioning** and are meant to catch a source
that has failed outright or degraded badly. Section 4.1 is equally explicit that the other kind of
test will not do this job:

    "traditional statistical procedures (e.g., the randomness tests described in NIST SP 800-22)
     ... will almost always fail, and thus are not useful for monitoring the noise source."

**SP 800-22 — statistical testing of the OUTPUT stream.** Useful as a sanity check on conditioned
output, and that is the only claim made for it here.

Until Round 17 this module implemented only the SP 800-22 subset while the docs described it as the
QRNG's health suite — precisely the conflation section 4.1 warns against. The continuous tests are
now implemented and are what :attr:`HealthReport.passed` is gated on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import erfc, gammaincc

#: SP 800-90B section 4.4 recommends a false-positive rate between 2^-20 and 2^-40 for the
#: continuous tests. 2^-20 is the standard's own worked-example value.
HEALTH_ALPHA = 2.0 ** -20

#: Adaptive Proportion window. Section 4.4.2: "W shall be 1024 for binary sources".
ADAPTIVE_PROPORTION_W = 1024

#: Section 4.3 requires start-up testing over at least this many consecutive samples.
STARTUP_SAMPLES = 1024


@dataclass
class HealthReport:
    """Result of the health suite.

    ``passed`` reflects the **SP 800-90B continuous tests**, which are the required ones. The
    SP 800-22 p-values are reported alongside as an output sanity check and deliberately do not gate
    it — a healthy noise source can and does fail an individual statistical test at rate alpha.
    """

    monobit_p: float
    runs_p: float
    block_frequency_p: float
    passed: bool
    alpha: float
    min_entropy_per_bit: float = 0.0   # SP 800-90B most-common-value estimate: bits of min-entropy per output bit
    min_entropy_bits: float = 0.0      # = min_entropy_per_bit * n (total estimated min-entropy)
    # --- SP 800-90B section 4.4 continuous health tests (the required ones) ---
    repetition_count_passed: bool = True
    repetition_count_cutoff: int = 0
    repetition_count_max_run: int = 0
    adaptive_proportion_passed: bool = True
    adaptive_proportion_cutoff: int = 0
    adaptive_proportion_max_count: int = 0
    startup_passed: bool = True
    sp80022_passed: bool = True        # the output-stream check, reported but not gating
    failures: list = field(default_factory=list)


def repetition_count_cutoff(h: float, alpha: float = HEALTH_ALPHA) -> int:
    """SP 800-90B section 4.4.1 cutoff: ``C = 1 + ceil(-log2(alpha) / H)``.

    Worked example in the standard: H = 2.0 bits/sample at alpha = 2^-20 gives C = 11.
    """
    if h <= 0:
        return 1 + int(np.ceil(-np.log2(alpha) / 1e-9))
    return 1 + int(np.ceil(-np.log2(alpha) / h))


def repetition_count_test(samples: np.ndarray, h: float, alpha: float = HEALTH_ALPHA):
    """Fail if any value repeats ``C`` or more times consecutively — a stuck noise source."""
    cutoff = repetition_count_cutoff(h, alpha)
    s = np.asarray(samples).ravel()
    if s.size == 0:
        return True, cutoff, 0
    # Longest run of identical consecutive samples.
    change = np.flatnonzero(np.diff(s) != 0)
    bounds = np.concatenate([[-1], change, [s.size - 1]])
    longest = int(np.diff(bounds).max()) if bounds.size > 1 else int(s.size)
    return bool(longest < cutoff), cutoff, longest


def adaptive_proportion_cutoff(h: float, w: int = ADAPTIVE_PROPORTION_W,
                               alpha: float = HEALTH_ALPHA) -> int:
    """SP 800-90B section 4.4.2 cutoff: the smallest C with ``Binom(W-1, 2^-H)`` tail <= alpha.

    Computed with the exact binomial survival function rather than the normal approximation, which
    is inaccurate in the far tail where alpha = 2^-20 lives.
    """
    from scipy.stats import binom

    p = 2.0 ** -max(h, 1e-9)
    return int(min(w, 1 + binom.isf(alpha, w - 1, min(p, 1.0))))


def adaptive_proportion_test(samples: np.ndarray, h: float, w: int = ADAPTIVE_PROPORTION_W,
                             alpha: float = HEALTH_ALPHA):
    """Fail if one value dominates any window of ``W`` samples — a source that has collapsed."""
    cutoff = adaptive_proportion_cutoff(h, w, alpha)
    s = np.asarray(samples).ravel()
    if s.size < w:
        return True, cutoff, 0
    worst = 0
    # Non-overlapping windows, as the standard describes: each window fixes its reference sample.
    for start in range(0, s.size - w + 1, w):
        win = s[start:start + w]
        worst = max(worst, int(np.count_nonzero(win == win[0])))
    return bool(worst < cutoff), cutoff, worst


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


def health_suite(bits, *, raw=None, alpha: float = 0.01, block_size: int = 128,
                 health_alpha: float = HEALTH_ALPHA) -> HealthReport:
    """Run the SP 800-90B continuous health tests plus the SP 800-22 output checks.

    ``passed`` is gated on the **SP 800-90B** tests, because those are the ones the standard
    requires of an entropy source. The SP 800-22 p-values are reported and summarised in
    ``sp80022_passed`` but do not gate: at ``alpha=0.01`` a perfectly healthy source fails one of
    them roughly 1 run in 33, and wiring that to a hard failure would make the check noise.

    ``alpha`` is the SP 800-22 significance level; ``health_alpha`` is the SP 800-90B false-positive
    rate, which the standard puts between 2^-20 and 2^-40 — four to twelve orders of magnitude
    apart, which is itself a hint that the two families answer different questions.
    
    ``raw`` is the pre-conditioning stream. The SP 800-90B tests and the min-entropy estimate run on
    it; SP 800-22 runs on ``bits``. Passing only ``bits`` measures the CONDITIONER, which is what
    this function did until Round 51 - see the note at the call site below.
    """
    b = np.asarray(bits).astype(np.int64).ravel()
    # SP 800-22 answers a question about the OUTPUT stream, so it stays on `bits`.
    mp = _monobit(b)
    rp = _runs(b)
    bp = _block_frequency(b, block_size)

    # THE TWO FAMILIES TAKE DIFFERENT STREAMS, and until Round 51 both took whatever the caller
    # passed - which was the CONDITIONED output. The comment on the line below already said "on the
    # raw stream"; it described an argument the function never received.
    #
    # Measured, because the symptom is not that a broken source fails to fail. At p(1) = 0.90 the
    # raw stream has min-entropy 0.1472 bit/bit; run after the Toeplitz extractor the same estimator
    # reports 0.9806, against 0.9892 for a fair 50/50 source. A badly biased source and a healthy one
    # are then a hundredth of a bit apart, because an extractor produces near-uniform output BY
    # CONSTRUCTION. The figure was not wrong about what it measured - it was measuring the extractor
    # while every label around it said entropy source, and it is the number a reader would use to
    # judge the noise.
    #
    # `raw` defaults to `bits` so existing callers keep their behaviour rather than silently changing
    # meaning; the entropy-source callers pass it explicitly.
    r = b if raw is None else np.asarray(raw).astype(np.int64).ravel()
    h = _min_entropy_mcv(r)

    # SP 800-90B section 4.4 — the required continuous tests, on the raw stream.
    rc_ok, rc_cutoff, rc_run = repetition_count_test(r, h, health_alpha)
    ap_ok, ap_cutoff, ap_count = adaptive_proportion_test(r, h, ADAPTIVE_PROPORTION_W, health_alpha)
    # Section 4.3 start-up: the same tests over the first >= 1024 samples, before any output is used.
    startup = r[:STARTUP_SAMPLES]
    startup_ok = bool(r.size < STARTUP_SAMPLES or (
        repetition_count_test(startup, h, health_alpha)[0]
        and adaptive_proportion_test(startup, h, ADAPTIVE_PROPORTION_W, health_alpha)[0]))

    failures = []
    if not rc_ok:
        failures.append(f"SP 800-90B repetition-count: run of {rc_run} >= cutoff {rc_cutoff}")
    if not ap_ok:
        failures.append(f"SP 800-90B adaptive-proportion: {ap_count} >= cutoff {ap_cutoff} "
                        f"in a {ADAPTIVE_PROPORTION_W}-sample window")
    if not startup_ok:
        failures.append(f"SP 800-90B start-up test failed over the first {STARTUP_SAMPLES} samples")

    sp22_ok = (mp >= alpha) and (rp >= alpha) and (bp >= alpha)
    return HealthReport(
        monobit_p=mp, runs_p=rp, block_frequency_p=bp,
        passed=bool(rc_ok and ap_ok and startup_ok), alpha=alpha,
        min_entropy_per_bit=h, min_entropy_bits=h * float(b.size),
        repetition_count_passed=rc_ok, repetition_count_cutoff=rc_cutoff,
        repetition_count_max_run=rc_run,
        adaptive_proportion_passed=ap_ok, adaptive_proportion_cutoff=ap_cutoff,
        adaptive_proportion_max_count=ap_count,
        startup_passed=startup_ok, sp80022_passed=sp22_ok, failures=failures)

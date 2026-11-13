"""Differential privacy for the medical federation and the genomics counts: the twin of `web/src/lib/clientDp.ts` (Round 86).

The distributed discrete Gaussian of Kairouz, Liu and Steinke (ICML 2021): each of n parties adds N_Z(0, V) to its integer
vector inside the secure sum, and their Theorem 1 makes the sum (1/2) e^2-concentrated DP with

    e = min( sqrt(D2^2 / (n V) + 2 tau d),  D2 / sqrt(n V) + tau sqrt(d) ),  tau = 10 sum_{k=1}^{n-1} exp(-2 pi^2 V k/(k+1)),

T releases composing to rho = T e^2 / 2, and eps <= rho + 2 sqrt(rho log(1/delta)) (Canonne, Kamath and Steinke 2020).
The sampler is their exact Algorithms 1 to 3 in integers, from `secrets` by default. `seeded_rand_below` reproduces the
page's test source, SHA-256 of `label:counter`; a run on it is NOT private.
"""

from __future__ import annotations

import hashlib
import math
import secrets
from collections.abc import Callable

RandBelow = Callable[[int], int]


def crypto_rand_below(m: int) -> int:
    return secrets.randbelow(m) if m > 1 else 0


def seeded_rand_below(label: str) -> RandBelow:
    """NOT PRIVATE: the twin of `seededRandBelow` in clientDp.ts, for tests only."""
    counter = 0

    def rb(m: int) -> int:
        nonlocal counter
        if m <= 1:
            return 0
        bits = (m - 1).bit_length()
        mask = (1 << bits) - 1
        while True:
            x = int.from_bytes(hashlib.sha256(f"{label}:{counter}".encode()).digest(), "big") & mask
            counter += 1
            if x < m:
                return x

    return rb


def _bernoulli(n: int, d: int, rb: RandBelow) -> bool:
    return rb(d) < n


def _bernoulli_exp_neg(n: int, d: int, rb: RandBelow) -> bool:
    if n <= d:
        k = 1
        while _bernoulli(n, d * k, rb):
            k += 1
        return k % 2 == 1
    for _ in range(n // d):
        if not _bernoulli_exp_neg(1, 1, rb):
            return False
    return _bernoulli_exp_neg(n % d, d, rb)


def _discrete_laplace(t: int, rb: RandBelow) -> int:
    while True:
        u = rb(t)
        if not _bernoulli_exp_neg(u, t, rb):
            continue
        v = 0
        while _bernoulli_exp_neg(1, 1, rb):
            v += 1
        x = u + t * v
        neg = rb(2) == 1
        if neg and x == 0:
            continue
        return -x if neg else x


def discrete_gaussian(variance: int, rb: RandBelow = crypto_rand_below) -> int:
    """One draw from N_Z(0, variance) for an integer variance >= 0; 0 draws 0."""
    if variance < 0 or int(variance) != variance:
        raise ValueError(f"variance {variance} is not an integer >= 0")
    if variance == 0:
        return 0
    v = int(variance)
    lo, hi = 0, 1
    while hi * hi <= v:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mid * mid <= v:
            lo = mid
        else:
            hi = mid
    t = lo + 1
    while True:
        y = _discrete_laplace(t, rb)
        g = abs(y) * t - v
        if _bernoulli_exp_neg(g * g, 2 * v * t * t, rb):
            return y


def ddg_tau(n: int, variance: float) -> float:
    s = 0.0
    for k in range(1, n):
        s += math.exp(-2 * math.pi * math.pi * variance * k / (k + 1))
    return 10 * s


def ddg_epsilon(d2: float, n: int, variance: float, d: int) -> float:
    tau = ddg_tau(n, variance)
    return min(math.sqrt(d2 * d2 / (n * variance) + 2 * tau * d), d2 / math.sqrt(n * variance) + tau * math.sqrt(d))


def cdp_to_dp(rho: float, delta: float) -> float:
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


def epsilon_of(variance: int, releases: int, d2: float, n: int, d: int, delta: float) -> float:
    if variance == 0:
        return math.inf
    e = ddg_epsilon(d2, n, variance, d)
    return cdp_to_dp(releases * e * e / 2, delta)


def variance_for(target: float, releases: int, d2: float, n: int, d: int, delta: float) -> int:
    if not math.isfinite(target):
        return 0
    hi = 1
    while epsilon_of(hi, releases, d2, n, d, delta) > target:
        hi *= 2
        if hi > 2 ** 52:
            raise ValueError(f"no variance below 2^52 reaches epsilon {target}")
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if epsilon_of(mid, releases, d2, n, d, delta) <= target:
            hi = mid
        else:
            lo = mid
    return hi


GENOMICS_SCALE, GENOMICS_SENSITIVITY, GENOMICS_DELTA = 16, 2, 1e-5
GENOMICS_EPSILONS = (0.5, 1.0, 2.0, 4.0, 8.0)


def genomics_dp(counts: list[int], epsilon: float, rb: RandBelow = crypto_rand_below) -> dict:
    """`genomicsDp` in clientDp.ts: each cohort's count in the fixed point 1/16 plus its share of the noise."""
    n, d2 = len(counts), GENOMICS_SENSITIVITY * GENOMICS_SCALE
    V = variance_for(epsilon, 1, d2, n, 1, GENOMICS_DELTA)
    return {
        "contributions": [c * GENOMICS_SCALE + discrete_gaussian(V, rb) for c in counts],
        "epsilon_achieved": epsilon_of(V, 1, d2, n, 1, GENOMICS_DELTA),
        "epsilon_one_dishonest": epsilon_of(V, 1, d2, n - 1, 1, GENOMICS_DELTA),
        "variance_per_cohort": V,
    }

"""The zCDP-to-DP conversion behind the medical, genomics and finance rows (Round 86, the owner's choice of 2026-11-14).

`qsh.ai.dp.cdp_to_dp` turns rho-zCDP into (eps, delta)-DP by Canonne, Kamath and Steinke's conversion (NeurIPS 2020,
Proposition 12) applied to the Renyi curve a rho, minimised over the order by golden-section search; `clientDp.ts` is its
twin (`tests/test_r86_dp_twin.py`). Held here: it never exceeds the simpler bound the pages used before, it finds the
minimum a dense grid of orders finds, it rises with rho and falls with delta, and it reproduces the figure an
independent accountant gives for the study's federated row.
"""

from __future__ import annotations

import math
import random

import pytest

from qsh.ai import dp


def _dense(rho: float, delta: float) -> float:
    best = math.inf
    for k in range(40001):
        u = -12 + 28 * k / 40000
        a = 1 + math.exp(u)
        best = min(best, a * rho + math.log1p(-1 / a) - (math.log(delta) + math.log(a)) / (a - 1))
    return min(best, dp.cdp_to_dp_simple(rho, delta))


def test_never_above_the_simple_bound_and_at_the_grid_minimum() -> None:
    rng = random.Random(7)
    for _ in range(60):
        rho, delta = 10 ** rng.uniform(-7, 3), 10 ** rng.uniform(-14, -2)
        eps = dp.cdp_to_dp(rho, delta)
        assert eps <= dp.cdp_to_dp_simple(rho, delta) * (1 + 1e-15)
        assert eps <= _dense(rho, delta) * (1 + 1e-9)


def test_monotone_and_the_edges() -> None:
    assert dp.cdp_to_dp(0.0, 1e-6) == 0.0 and math.isinf(dp.cdp_to_dp(math.inf, 1e-6))
    rhos = [10 ** (k / 4) for k in range(-24, 9)]
    eps = [dp.cdp_to_dp(r, 1e-6) for r in rhos]
    assert all(b > a for a, b in zip(eps, eps[1:], strict=False)), "rises with rho"
    assert dp.cdp_to_dp(0.5, 1e-8) > dp.cdp_to_dp(0.5, 1e-5), "falls as delta grows"


def test_a_figure_an_independent_accountant_gives() -> None:
    # rho 1.0491 at delta 1e-5 is the federated arm's eps 8 under the simple bound; a separate accountant written with
    # numpy and scipy for the murmur study (CKS conversion) gives 7.28 for it.
    assert dp.cdp_to_dp_simple(1.0491, 1e-5) == pytest.approx(8.0, abs=0.01)
    assert dp.cdp_to_dp(1.0491, 1e-5) == pytest.approx(7.28, abs=0.005)

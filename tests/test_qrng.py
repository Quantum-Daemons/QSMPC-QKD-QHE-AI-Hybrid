"""QRNG entropy source tests (offline-safe, deterministic)."""

import numpy as np
import pytest

from qsh.qkd.bb84 import run_bb84
from qsh.qkd.channel import ChannelParams
from qsh.qrng import generate_quantum_bits, health_suite, quantum_seed, von_neumann


def test_qrng_health_passes_on_sim_output():
    r = generate_quantum_bits(8192, seed=0)
    assert r.health.passed
    assert r.health.monobit_p > r.health.alpha
    assert r.health.runs_p > r.health.alpha
    assert r.health.block_frequency_p > r.health.alpha
    assert r.simulated is True and r.source == "sim"
    assert r.throughput_bits_per_s > 0


def test_qrng_deterministic_under_seed():
    a = generate_quantum_bits(4096, seed=7)
    b = generate_quantum_bits(4096, seed=7)
    assert np.array_equal(a.bits, b.bits)


def test_von_neumann_debiases():
    rng = np.random.default_rng(1)
    biased = (rng.random(40000) < 0.75).astype(int)
    out = von_neumann(biased)
    assert abs(out.mean() - 0.5) < abs(biased.mean() - 0.5)     # closer to unbiased
    assert health_suite(out).monobit_p > 0.01                   # debiased stream passes monobit


def test_quantum_seed_deterministic_and_seeds_bb84():
    assert quantum_seed(seed=3) == quantum_seed(seed=3)
    assert isinstance(quantum_seed(seed=3), int)
    res = run_bb84(40_000, ChannelParams(distance_km=50), seed=1, entropy="qrng")
    assert res.sifted_bits > 0 and 0.0 <= res.qber <= 0.5


def test_anu_requires_opt_in():
    """The ANU (online, hardware) source must never be used without an explicit opt-in — no network in CI."""
    with pytest.raises(ValueError):
        generate_quantum_bits(64, source="anu", anu_opt_in=False)

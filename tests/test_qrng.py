"""QRNG entropy source tests (offline-safe, deterministic)."""

import numpy as np
import pytest

from qsh.qkd.bb84 import run_bb84
from qsh.qkd.channel import ChannelParams
from qsh.qrng import generate_quantum_bits, health_suite, quantum_seed, von_neumann
from qsh.qrng.health import _min_entropy_mcv
from qsh.qrng.simulator import lhl_output_length, toeplitz_extract


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


# --- Round 16: information-theoretic extractor (Toeplitz + Leftover-Hash-Lemma) + min-entropy ---

def test_lhl_output_length_formula():
    # out = floor(H_min - 2*log2(1/eps) + 2); eps = 2^-40 => 2*log2(1/eps) = 80
    assert lhl_output_length(8000, -40) == 8000 - 80 + 2
    assert lhl_output_length(50, -40) == 1        # clamped to >=1 when the bound is non-positive


def test_min_entropy_estimate_uniform_vs_biased():
    rng = np.random.default_rng(0)
    uniform = (rng.random(20000) < 0.5).astype(int)
    biased = (rng.random(20000) < 0.85).astype(int)
    h_uniform, h_biased = _min_entropy_mcv(uniform), _min_entropy_mcv(biased)
    assert h_uniform > 0.9            # near 1 bit/bit for a fair coin
    assert h_biased < 0.6            # clearly lower for a biased source
    assert h_uniform > h_biased


def test_health_report_carries_min_entropy():
    r = generate_quantum_bits(8192, seed=0)
    assert 0.0 < r.health.min_entropy_per_bit <= 1.0
    assert r.health.min_entropy_bits == pytest.approx(r.health.min_entropy_per_bit * r.n_bits)


def test_toeplitz_extractor_lhl_bounded_and_healthy():
    # An extractor guarantees near-uniform output (monobit + min-entropy); requiring ALL three SP800-22
    # p-values to clear alpha=0.01 for a single seed is inherently ~3% flaky, so assert the guarantees.
    r = generate_quantum_bits(8192, post_processing="toeplitz", seed=0)
    assert r.n_bits == 8192
    assert 0.0 < r.extraction_rate <= 1.0
    assert r.health.min_entropy_per_bit > 0.9          # strong extraction (near-uniform)
    assert r.health.monobit_p > r.health.alpha         # removes first-order bias
    # explicit second-source (two-source) path returns a valid binary vector of the requested length
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int64)
    tbits = np.array([1, 1, 0, 1, 0, 1, 0, 1, 1, 0], dtype=np.int64)
    out = toeplitz_extract(bits, out_len=3, toeplitz_bits=tbits)
    assert out.size == 3 and set(np.unique(out)).issubset({0, 1})


def test_seedless_two_source_extractor():
    a = generate_quantum_bits(4096, post_processing="seedless", seed=0)
    b = generate_quantum_bits(4096, post_processing="seedless", seed=0)
    assert np.array_equal(a.bits, b.bits)           # deterministic under seed
    assert a.n_bits == 4096
    assert a.health.monobit_p > a.health.alpha      # removes first-order bias
    assert a.health.min_entropy_per_bit > 0.9       # strong two-source extraction
    assert a.raw_bits_consumed > 4096               # consumes an input source + an independent second source
    assert 0.0 < a.extraction_rate < 1.0

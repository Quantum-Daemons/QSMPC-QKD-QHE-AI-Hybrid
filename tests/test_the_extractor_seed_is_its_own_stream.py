"""The seeded Toeplitz extractor draws its matrix from a stream independent of the raw bits.

WHY THIS EXISTS

The Leftover Hash Lemma bounds a seeded extractor's output for a seed INDEPENDENT of the source.
Until Round 76 both languages broke that in the seeded mode. Python passed the run's `seed` to
`toeplitz_extract`, whose `default_rng(seed)` re-drew the matrix from the very state that had just
drawn the raw bits. The browser continued `u`, the same generator, after the raw draw; unseeded,
that generator was `Math.random`, which is not a CSPRNG. The seedless mode already took its second
source from `seed + 1`.

WHAT IS ASSERTED

- Python hands the extractor `seed + 2` in a seeded run and fresh OS entropy (None) otherwise.
- The browser's seeded branch passes matrix bits it drew for the purpose, from `seed + 2` or from
  `csprng.ts`, never the raw draw's generator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qsh.qrng import simulator

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("seed,expected", [(7, 9), (None, None)])
def test_python_gives_the_extractor_its_own_seed(monkeypatch, seed, expected) -> None:
    seen = {}
    real = simulator.toeplitz_extract

    def spy(bits, out_len, *, seed=None, toeplitz_bits=None):
        seen["seed"] = seed
        return real(bits, out_len, seed=seed, toeplitz_bits=toeplitz_bits)

    monkeypatch.setattr(simulator, "toeplitz_extract", spy)
    simulator.generate_quantum_bits(1024, post_processing="toeplitz", seed=seed)
    assert seen["seed"] == expected


def test_the_browser_seeded_branch_draws_its_own_matrix() -> None:
    src = (REPO / "web" / "src" / "lib" / "clientQRNG.ts").read_text()
    assert "proc = toeplitzExtract(raw, outLen, u);" not in src, "the matrix is the raw draw's generator again"
    assert "toeplitzExtract(raw, outLen, u, extractorSeedBits(needT, opts.seed))" in src
    assert 'import { randomWords } from "./csprng";' in src and "rng(seed + 2)" in src

"""Bridge the QRNG to the rest of the stack: a quantum-derived integer seed.

``quantum_seed`` folds post-processed QRNG bits through SHAKE-256 into a 64-bit integer suitable
for ``numpy.random.default_rng`` — so e.g. the BB84 basis/bit choices can genuinely originate from a
(simulated) quantum entropy source. Honest label: the default source is simulated (state-vector), not
quantum hardware.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .simulator import generate_quantum_bits


def quantum_seed(*, source: str = "sim", seed: int | None = None, bits: int = 256,
                 anu_opt_in: bool = False) -> int:
    """Return a 64-bit integer seed derived from ``bits`` post-processed quantum-random bits."""
    result = generate_quantum_bits(bits, source=source, post_processing="von_neumann",
                                   seed=seed, anu_opt_in=anu_opt_in)
    packed = np.packbits(result.bits.astype(np.uint8)).tobytes()
    digest = hashlib.shake_256(packed).digest(8)
    return int.from_bytes(digest, "big")

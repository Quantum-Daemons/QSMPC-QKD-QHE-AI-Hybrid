"""BB84 quantum key distribution (decoy-state) — simulation.

A prepare-and-measure BB84 simulation: Alice encodes random bits in random bases,
the :mod:`~qsh.qkd.channel` transports them with loss/noise/eavesdropping, Bob
measures in random bases, the parties sift, estimate QBER, and a finite-key secret
key rate (SKR) is computed with the GLLP/decoy-state secure-fraction formula.

This is the physical substrate for WS1 (AI-QKD): the AI optimizer
(:mod:`~qsh.qkd.ai_optimizer`) tunes parameters here to maximise SKR / minimise QBER.
A small Qiskit circuit demo is provided in :func:`bb84_circuit_demo` to show the
quantum mechanics explicitly; the Monte-Carlo path scales to long keys.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import ChannelParams, intrinsic_qber


def _h2(x: float) -> float:
    """Binary entropy H2(x)."""
    if x <= 0 or x >= 1:
        return 0.0
    return float(-x * np.log2(x) - (1 - x) * np.log2(1 - x))


@dataclass
class QKDResult:
    """Outcome of a BB84 session."""

    sifted_bits: int
    qber: float
    secret_key_rate: float        # bits of secure key per channel use
    secure_key_bits: int          # finite secure key length after EC + PA
    sifted_key: np.ndarray        # Bob's sifted key (pre error-correction), for demos
    params: ChannelParams


def run_bb84(
    n_pulses: int = 100_000,
    params: ChannelParams | None = None,
    *,
    error_correction_eff: float = 1.16,   # f_EC, real Cascade/LDPC overhead (>1)
    seed: int | None = None,
) -> QKDResult:
    """Simulate a decoy-state BB84 session over ``n_pulses`` channel uses."""
    params = params or ChannelParams()
    rng = np.random.default_rng(seed)

    alice_bits = rng.integers(0, 2, n_pulses)
    alice_bases = rng.integers(0, 2, n_pulses)
    bob_bases = rng.integers(0, 2, n_pulses)

    # which pulses are detected at all (loss + detector efficiency)
    detected = rng.random(n_pulses) < params.transmittance()
    # sift: keep detected pulses where the bases match
    sift = detected & (alice_bases == bob_bases)

    qber_phys = intrinsic_qber(params)
    bob_bits = alice_bits.copy()
    flips = rng.random(n_pulses) < qber_phys
    bob_bits = np.where(flips, 1 - bob_bits, bob_bits)

    sifted_idx = np.flatnonzero(sift)
    n_sift = sifted_idx.size
    if n_sift == 0:
        return QKDResult(0, qber_phys, 0.0, 0, np.array([], dtype=int), params)

    measured_qber = float(np.mean(alice_bits[sifted_idx] != bob_bits[sifted_idx]))

    # GLLP secure fraction: 1 - H2(e) [phase] - f*H2(e) [error correction leakage]
    secure_fraction = max(
        0.0, 1.0 - _h2(measured_qber) - error_correction_eff * _h2(measured_qber)
    )
    skr = secure_fraction * (n_sift / n_pulses)
    secure_bits = int(secure_fraction * n_sift)

    return QKDResult(
        sifted_bits=n_sift,
        qber=measured_qber,
        secret_key_rate=skr,
        secure_key_bits=secure_bits,
        sifted_key=bob_bits[sifted_idx],
        params=params,
    )


def derive_shared_key(result: QKDResult, n_bytes: int = 32, *, seed: int | None = None) -> bytes:
    """Derive an n-byte symmetric key from the secure BB84 material.

    Privacy amplification is modelled by hashing the secure portion of the sifted
    key to a fixed-length output (real systems use a 2-universal hash family).
    """
    import hashlib

    if result.secure_key_bits < n_bytes * 8:
        raise ValueError(
            f"insufficient secure key: {result.secure_key_bits} bits < {n_bytes * 8} requested"
        )
    bits = result.sifted_key[: result.secure_key_bits]
    packed = np.packbits(bits).tobytes()
    return hashlib.shake_256(packed).digest(n_bytes)


def bb84_circuit_demo(n: int = 8, *, seed: int | None = None) -> dict:
    """Tiny Qiskit BB84 demo (no channel noise) showing sift + key agreement.

    Returns a dict with Alice/Bob bits, bases, and the sifted key. Used by the SPA
    'Validations' page to render the quantum mechanics explicitly.
    """
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    rng = np.random.default_rng(seed)
    a_bits = rng.integers(0, 2, n)
    a_bases = rng.integers(0, 2, n)
    b_bases = rng.integers(0, 2, n)
    sim = AerSimulator()
    b_bits = []
    for i in range(n):
        qc = QuantumCircuit(1, 1)
        if a_bits[i]:
            qc.x(0)
        if a_bases[i]:
            qc.h(0)          # prepare in X basis
        if b_bases[i]:
            qc.h(0)          # measure in X basis
        qc.measure(0, 0)
        counts = sim.run(qc, shots=1, seed_simulator=int(rng.integers(1 << 30))).result().get_counts()
        b_bits.append(int(next(iter(counts))))
    match = a_bases == b_bases
    return {
        "alice_bits": a_bits.tolist(),
        "alice_bases": a_bases.tolist(),
        "bob_bases": b_bases.tolist(),
        "bob_bits": b_bits,
        "sifted_key": [int(a_bits[i]) for i in range(n) if match[i]],
    }

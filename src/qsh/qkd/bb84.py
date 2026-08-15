"""BB84 quantum key distribution (decoy-state) — simulation.

A prepare-and-measure BB84 simulation: Alice encodes random bits in random bases,
the :mod:`~qsh.qkd.channel` transports them with loss/noise/eavesdropping, Bob
measures in random bases, the parties sift, estimate QBER, and an ASYMPTOTIC secret
key rate (SKR) is computed - by the two-intensity decoy-state bound of Ma, Qi, Zhao
& Lo (2005) when ``qkd_params`` is supplied, and by the legacy non-decoy GLLP form
when it is not.

Round 45 corrected the word "finite-key" here. Neither branch applies a finite-key
correction: both are asymptotic bounds, paying no statistical penalty at all.

ROUND 46 CORRECTED THE NUMBER IN THIS PARAGRAPH AND THEN THE PARAMETER IT DESCRIBED.
It read "roughly 1,900 sifted bits" - a figure that matched no computed quantity
anywhere in the repository. The 200,000-pulse run gave ``nSift = N * Q_mu * 0.5``
= **1,244** in the browser and 1,188 in this Monte Carlo. Round 45 wrote a caveat
about a block size it had not measured.

The size of the omission is now quantified rather than left as a direction. At
~1,244 sifted bits a composably-secure finite-key analysis yields **no key at all**;
the positive-key threshold for these device parameters at 50 km is ~5e3 to 1.1e4
sifted bits (Rusca, Boaron, Grunenfelder, Martin & Zbinden, *Appl. Phys. Lett.*
**112**, 171104 (2018)). So the default pulse count was raised to 2,000,000
(12,442 sifted bits) - see ``QKDConfig.pulses``. That does not make the rate
finite-key correct; it stops an asymptotic rate being published for a block that
provably yields nothing.

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


def _decoy_secure_fraction(eta: float, mu: float, nu: float, e_det: float, y0: float,
                           f_ec: float, sift_ratio: float) -> tuple[float, float, float]:
    """Two-intensity decoy-state secret-key rate per pulse (Ma, Qi, Zhao & Lo 2005).

    A signal (``mu``) + decoy (``nu``) intensity pair lower-bounds the single-photon yield ``Y1`` and
    upper-bounds the single-photon error ``e1``; the GLLP rate then keeps only the *secure* single-photon
    contribution: ``R = q [ Q1 (1 - H2(e1)) - f Q_mu H2(E_mu) ]``. The decoy intensity ``nu`` genuinely
    moves the rate through the tightness of the Y1/e1 bounds. Returns ``(skr_per_pulse, E_mu, e1_upper)``.
    """
    e0 = 0.5                                             # dark counts are unbiased (random) errors
    q_mu = y0 + 1.0 - float(np.exp(-eta * mu))           # overall signal gain
    q_nu = y0 + 1.0 - float(np.exp(-eta * nu))           # overall decoy gain
    if q_mu <= 0.0:
        return 0.0, 0.5, 0.5
    e_mu = (e0 * y0 + e_det * (1.0 - float(np.exp(-eta * mu)))) / q_mu
    if mu <= nu or q_nu <= 0.0:                          # decoy must be strictly weaker than the signal
        return 0.0, e_mu, 0.5
    e_nu = (e0 * y0 + e_det * (1.0 - float(np.exp(-eta * nu)))) / q_nu
    # single-photon yield lower bound + error upper bound (Ma et al. 2005, Eqs. 34-37)
    y1 = max(0.0, (mu / (mu * nu - nu * nu)) * (
        q_nu * float(np.exp(nu)) - (nu * nu) / (mu * mu) * q_mu * float(np.exp(mu))
        - (mu * mu - nu * nu) / (mu * mu) * y0))
    q1 = y1 * mu * float(np.exp(-mu))                    # secure single-photon gain
    e1 = min(0.5, max(0.0, (e_nu * q_nu * float(np.exp(nu)) - e0 * y0) / (y1 * nu))) if y1 > 0.0 else 0.5
    r = q1 * (1.0 - _h2(e1)) - f_ec * q_mu * _h2(e_mu)   # GLLP single-photon rate per pulse
    return max(0.0, r) * sift_ratio, e_mu, e1


@dataclass
class QKDResult:
    """Outcome of a BB84 session."""

    sifted_bits: int
    qber: float
    secret_key_rate: float        # bits of secure key per channel use
    secure_key_bits: int          # finite secure key length after EC + PA
    sifted_key: np.ndarray        # Bob's sifted key (pre error-correction), for demos
    params: ChannelParams


@dataclass
class QKDParams:
    """Tunable protocol parameters the AI optimizer (:mod:`~qsh.qkd.optimize`) searches over.

    ``basis_prob`` = P(Z basis): biased-basis (efficient) BB84 raises the sift ratio ``p^2+(1-p)^2``
    above 0.5 (Lo-Chau-Ardehali 2005). ``mu_signal`` = decoy-state signal mean photon number: higher mu
    means more detections but more PNS-leakable multi-photon pulses (a QBER penalty) -> a genuine optimum
    near ~0.5. ``pp_threshold`` = post-processing QBER abort threshold. When ``run_bb84`` is called with
    ``qkd_params=None`` these have no effect (the legacy uniform-basis path runs unchanged)."""

    mu_signal: float = 0.5
    mu_decoy: float = 0.1
    basis_prob: float = 0.5
    pp_threshold: float = 0.11


def run_bb84(
    n_pulses: int = 100_000,
    params: ChannelParams | None = None,
    *,
    error_correction_eff: float = 1.16,   # f_EC, real Cascade/LDPC overhead (>1)
    seed: int | None = None,
    qkd_params: QKDParams | None = None,
    entropy: str | None = None,           # "qrng" -> seed the basis/bit choices from the quantum RNG
) -> QKDResult:
    """Simulate a decoy-state BB84 session over ``n_pulses`` channel uses.

    ``qkd_params`` (a :class:`QKDParams`) turns on the decoy-state + efficient-BB84 model the AI optimizer
    tunes; ``None`` keeps the original uniform-basis behaviour byte-for-byte. ``entropy="qrng"`` derives the
    RNG seed from the (simulated) quantum RNG so the basis/bit choices have a genuine quantum entropy root.
    """
    params = params or ChannelParams()
    if entropy == "qrng":
        from ..qrng import quantum_seed
        seed = quantum_seed(seed=seed)
    rng = np.random.default_rng(seed)
    eta = params.transmittance()

    if qkd_params is None:
        # --- legacy path (unchanged: uniform bases, detection == transmittance, no explicit abort) ---
        alice_bits = rng.integers(0, 2, n_pulses)
        alice_bases = rng.integers(0, 2, n_pulses)
        bob_bases = rng.integers(0, 2, n_pulses)
        detected = rng.random(n_pulses) < eta
        sift = detected & (alice_bases == bob_bases)
        qber_phys = intrinsic_qber(params)
        pp_threshold = 1.0
    else:
        # --- decoy-state + efficient (biased-basis) BB84: the AI-optimizable model ---
        p = float(np.clip(qkd_params.basis_prob, 0.01, 0.99))
        alice_bits = rng.integers(0, 2, n_pulses)
        alice_bases = (rng.random(n_pulses) >= p).astype(int)   # P(basis 0) = p
        bob_bases = (rng.random(n_pulses) >= p).astype(int)
        mu = max(1e-3, float(qkd_params.mu_signal))
        detected = rng.random(n_pulses) < (1.0 - np.exp(-mu * eta))   # P(>=1 photon detected), Poisson
        sift = detected & (alice_bases == bob_bases)
        # the multiphoton (PNS) insecurity is now accounted for analytically by the decoy-state single-photon
        # bound (see the secure-rate tail), NOT by inflating the physical QBER -> measured QBER = true error.
        qber_phys = intrinsic_qber(params)
        pp_threshold = float(qkd_params.pp_threshold)

    bob_bits = alice_bits.copy()
    flips = rng.random(n_pulses) < qber_phys
    bob_bits = np.where(flips, 1 - bob_bits, bob_bits)

    sifted_idx = np.flatnonzero(sift)
    n_sift = sifted_idx.size
    if n_sift == 0:
        return QKDResult(0, qber_phys, 0.0, 0, np.array([], dtype=int), params)

    measured_qber = float(np.mean(alice_bits[sifted_idx] != bob_bits[sifted_idx]))
    if measured_qber > pp_threshold:
        # QBER above the post-processing threshold -> abort, no secure key
        return QKDResult(n_sift, measured_qber, 0.0, 0, bob_bits[sifted_idx], params)

    if qkd_params is None:
        # legacy GLLP secure fraction on the overall QBER: 1 - H2(e) [phase] - f*H2(e) [EC leakage]
        secure_fraction = max(
            0.0, 1.0 - _h2(measured_qber) - error_correction_eff * _h2(measured_qber)
        )
        skr = secure_fraction * (n_sift / n_pulses)
        secure_bits = int(secure_fraction * n_sift)
    else:
        # two-intensity decoy-state single-photon secure rate — mu_decoy genuinely tightens Y1/e1 (Ma 2005)
        nu = min(max(1e-4, float(qkd_params.mu_decoy)), mu * 0.9)      # decoy strictly weaker than the signal
        y0 = 2.0 * params.dark_count_prob                             # background yield (two detectors)
        sift_ratio = p * p + (1.0 - p) * (1.0 - p)                    # biased-basis reconciliation ratio
        skr, _e_mu, _e1 = _decoy_secure_fraction(
            eta, mu, nu, intrinsic_qber(params), y0, error_correction_eff, sift_ratio)
        secure_bits = int(max(0.0, skr) * n_pulses)

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

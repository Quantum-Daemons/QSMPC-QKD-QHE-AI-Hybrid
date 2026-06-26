r"""QSMS — Quantum Secure Multiparty Summation (Shamir + d-level QFT).

Faithful simulation of the (t, n)-threshold protocol of
Sutradhar, *Efficient Simulation of Quantum Secure Multiparty Computation*,
arXiv:2501.10083, 2025 (``references/Efficient_Simulation_of_Quantum_Secure_
Multiparty_Computation.pdf``).

Protocol (per the paper, steps 5-11):
  1.  Each party Shamir-shares its secret over GF(d) (d prime, n <= d <= 2n).
  2.  Each of the ``t`` computing parties sums the shares it holds: h(x_u).
  3.  Party u forms its Lagrange "shadow"   m_u = h(x_u) * l_u(0) mod d,
      so that   sum_u m_u  =  sum of all secrets   (Shamir reconstruction at 0).
  4.  An initiator prepares the t-qudit GHZ state  |Psi1> = d^{-1/2} sum_c |c..c>.
  5.  Each party applies the d-level QFT to its qudit, then the Pauli shift X_d^{m_u}.
  6.  Measuring yields outcomes (a_u + m_u) with  sum_u a_u == 0 (mod d), hence
          S = sum_u (a_u + m_u) mod d = sum_u m_u mod d = sum of secrets.

Qudit (d-level) systems are not native to qubit SDKs, so the d-level QFT and shift
are simulated exactly on the statevector with NumPy. For small (t, d) we run the
full GHZ statevector ("honest" simulation, also used to *verify* the sum_a == 0
property); for larger sizes we sample from the analytically-identical outcome
distribution. Either way the recovered sum is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .shamir import (
    Share,
    is_probable_prime,
    lagrange_coefficient,
    next_prime,
    split,
)

# Full-statevector simulation is used while d**t stays under this many amplitudes.
_MAX_STATE = 1 << 22  # ~4.2M complex128 ~= 64 MB


def qft_matrix(d: int) -> np.ndarray:
    """The d x d quantum Fourier transform unitary over Z_d."""
    j = np.arange(d)
    omega = np.exp(2j * np.pi / d)
    return omega ** np.outer(j, j) / np.sqrt(d)


def shift_matrix(d: int, m: int) -> np.ndarray:
    """The cyclic Pauli-X^m shift on a qudit: |c> -> |c+m mod d>."""
    x = np.zeros((d, d), dtype=complex)
    for c in range(d):
        x[(c + m) % d, c] = 1.0
    return x


def _apply_single(state: np.ndarray, u: np.ndarray, axis: int) -> np.ndarray:
    """Apply single-qudit gate ``u`` to ``axis`` of a (d,)*t tensor state."""
    moved = np.moveaxis(state, axis, 0)
    out = np.tensordot(u, moved, axes=([1], [0]))
    return np.moveaxis(out, 0, axis)


@dataclass
class QSMSResult:
    """Outcome of a QSMS run."""

    total: int  # reconstructed sum of secrets, mod d
    modulus: int  # the field prime d
    shadows: list[int]  # per-party Lagrange shadows m_u (kept private in practice)
    broadcasts: list[int]  # per-party measured (a_u + m_u) mod d
    masks: list[int]  # the a_u masks (sum to 0 mod d); empty unless honest sim
    honest_statevector: bool  # True if the full GHZ statevector was simulated


@dataclass
class QSMS:
    """Quantum secure multiparty summation over GF(prime).

    Args:
        n: total parties holding shares.
        t: threshold / number of computing parties (must be <= n).
        prime: field modulus d (prime, > n and > any secret). Auto-chosen if None.
    """

    n: int
    t: int
    prime: int | None = None
    _xs: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not (1 < self.t <= self.n):
            raise ValueError("require 1 < t <= n")
        if self.prime is None:
            self.prime = next_prime(self.n + 1)
        if not is_probable_prime(self.prime):
            raise ValueError("prime must be prime")
        if self.prime <= self.n:
            raise ValueError("prime must exceed n")
        # computing parties use evaluation points 1..t (a qualified subset)
        self._xs = list(range(1, self.n + 1))

    # -- classical share preparation ---------------------------------------
    def _shadows(self, secrets: list[int], rng=None) -> list[int]:
        """Compute the per-(computing-)party Lagrange shadows m_u."""
        d = self.prime
        # share every secret; sum the share-vectors elementwise (additive over shares)
        summed_y = [0] * self.n
        for s in secrets:
            shares: list[Share] = split(s % d, self.n, self.t, d, xs=self._xs, rng=rng)
            for i, sh in enumerate(shares):
                summed_y[i] = (summed_y[i] + sh.y) % d
        subset_xs = self._xs[: self.t]
        shadows = []
        for u in range(self.t):
            l_u0 = lagrange_coefficient(subset_xs, u, d, at=0)
            shadows.append(summed_y[u] * l_u0 % d)
        return shadows

    # -- quantum summation of the shadows ----------------------------------
    def _quantum_sum(self, shadows: list[int], *, seed: int | None = None
                     ) -> tuple[list[int], list[int], bool]:
        """Securely sum the shadows via the GHZ + QFT_d + shift protocol.

        Returns (broadcasts, masks, honest) where broadcasts[u] = (a_u + m_u) mod d.
        """
        d = self.prime
        t = self.t
        rs = np.random.default_rng(seed)
        if d**t <= _MAX_STATE:
            # honest full-statevector simulation
            state = np.zeros((d,) * t, dtype=complex)
            for c in range(d):
                state[(c,) * t] = 1.0 / np.sqrt(d)
            f = qft_matrix(d)
            for u in range(t):
                state = _apply_single(state, f, u)
                state = _apply_single(state, shift_matrix(d, shadows[u]), u)
            probs = np.abs(state.reshape(-1)) ** 2
            probs /= probs.sum()
            idx = rs.choice(probs.size, p=probs)
            outcome = np.unravel_index(idx, (d,) * t)  # (a_u + m_u) values
            broadcasts = [int(o) for o in outcome]
            masks = [(broadcasts[u] - shadows[u]) % d for u in range(t)]
            return broadcasts, masks, True
        # analytic sampler (identical distribution): a_u uniform with sum 0 mod d
        masks = [int(rs.integers(0, d)) for _ in range(t - 1)]
        masks.append((-sum(masks)) % d)
        broadcasts = [(masks[u] + shadows[u]) % d for u in range(t)]
        return broadcasts, masks, False

    def run(self, secrets: list[int], *, seed: int | None = None, rng=None) -> QSMSResult:
        """Run the full QSMS protocol on a list of per-party ``secrets``.

        The returned ``total`` equals ``sum(secrets) mod prime`` and is recovered
        without any party revealing its individual shadow.
        """
        if not secrets:
            raise ValueError("need at least one secret")
        shadows = self._shadows(secrets, rng=rng)
        broadcasts, masks, honest = self._quantum_sum(shadows, seed=seed)
        total = sum(broadcasts) % self.prime
        return QSMSResult(
            total=total,
            modulus=self.prime,
            shadows=shadows,
            broadcasts=broadcasts,
            masks=masks,
            honest_statevector=honest,
        )


def quantum_secure_sum(
    secrets: list[int],
    *,
    n: int | None = None,
    t: int | None = None,
    prime: int | None = None,
    seed: int | None = None,
) -> int:
    """Convenience: securely sum ``secrets`` and return the total mod prime.

    Defaults to a 3-party computing set (t=3) over the smallest valid prime.
    """
    n = n or max(len(secrets), 3)
    t = t or min(3, n)
    return QSMS(n=n, t=t, prime=prime).run(secrets, seed=seed).total

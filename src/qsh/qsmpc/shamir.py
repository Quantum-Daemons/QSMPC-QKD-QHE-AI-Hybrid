"""Shamir (t, n)-threshold secret sharing over a prime field GF(p).

Used as the classical layer of the QSMS quantum-secure-summation protocol
(Sutradhar, arXiv:2501.10083, 2025 -- `references/Efficient_Simulation_of_Quantum_
Secure_Multiparty_Computation.pdf`). Any `t` of `n` shares reconstruct the secret;
fewer reveal nothing (information-theoretic, in the field).

This module is pure Python / NumPy and has no quantum dependency, so it is unit
testable on its own.
"""

from __future__ import annotations

import secrets as _secrets
from dataclasses import dataclass


def _egcd(a: int, b: int) -> tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    g, x, y = _egcd(b % a, a)
    return g, y - (b // a) * x, x


def modinv(a: int, p: int) -> int:
    """Modular inverse of ``a`` mod prime ``p``."""
    a %= p
    if a == 0:
        raise ZeroDivisionError("no inverse for 0")
    g, x, _ = _egcd(a, p)
    if g != 1:
        raise ValueError(f"{a} has no inverse mod {p}")
    return x % p


def is_probable_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 3.3e24 (enough for our small primes)."""
    if n < 2:
        return False
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % q == 0:
            return n == q
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(n: int) -> int:
    """Smallest prime >= n."""
    m = max(2, n)
    while not is_probable_prime(m):
        m += 1
    return m


@dataclass(frozen=True)
class Share:
    """A single Shamir share: the point (x, f(x) mod p)."""

    x: int
    y: int
    prime: int


def _eval_poly(coeffs: list[int], x: int, p: int) -> int:
    """Horner evaluation of a polynomial (coeffs low->high) at x over GF(p)."""
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % p
    return acc


def split(
    secret: int,
    n: int,
    t: int,
    prime: int,
    *,
    xs: list[int] | None = None,
    rng=None,
) -> list[Share]:
    """Split ``secret`` into ``n`` shares with threshold ``t`` over GF(``prime``).

    Args:
        secret: the value to share (0 <= secret < prime).
        n: number of shares to produce.
        t: threshold -- any ``t`` shares reconstruct; ``t-1`` reveal nothing.
        prime: field modulus (must be prime and > n, > secret).
        xs: optional explicit evaluation points (defaults to 1..n).
        rng: optional callable rng(p)->int for the random coefficients (testing).
    """
    if not is_probable_prime(prime):
        raise ValueError("prime must be prime")
    if not (0 <= secret < prime):
        raise ValueError("secret out of field range")
    if not (1 <= t <= n < prime):
        raise ValueError("require 1 <= t <= n < prime")
    rand = rng or (lambda p: _secrets.randbelow(p))
    coeffs = [secret] + [rand(prime) for _ in range(t - 1)]
    points = xs if xs is not None else list(range(1, n + 1))
    if len(points) != n or any(x % prime == 0 for x in points):
        raise ValueError("need n distinct nonzero evaluation points")
    return [Share(x, _eval_poly(coeffs, x, prime), prime) for x in points]


def lagrange_coefficient(xs: list[int], i: int, prime: int, at: int = 0) -> int:
    """Lagrange basis weight l_i(``at``) for points ``xs`` over GF(``prime``)."""
    xi = xs[i]
    num, den = 1, 1
    for j, xj in enumerate(xs):
        if j == i:
            continue
        num = num * ((at - xj) % prime) % prime
        den = den * ((xi - xj) % prime) % prime
    return num * modinv(den, prime) % prime


def reconstruct(shares: list[Share], *, at: int = 0) -> int:
    """Reconstruct f(``at``) (the secret when at=0) from >= t shares via Lagrange."""
    if not shares:
        raise ValueError("no shares")
    p = shares[0].prime
    xs = [s.x for s in shares]
    if len(set(xs)) != len(xs):
        raise ValueError("shares must have distinct x")
    acc = 0
    for i, s in enumerate(shares):
        acc = (acc + s.y * lagrange_coefficient(xs, i, p, at)) % p
    return acc

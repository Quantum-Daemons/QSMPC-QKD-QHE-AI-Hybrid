"""Tests for the QSMPC layer: Shamir SS, QSMS secure summation, QOT, PSI."""

import random

import pytest

from qsh.qsmpc.qot import multiparty_psi, private_set_intersection, quantum_oblivious_transfer
from qsh.qsmpc.secure_summation import QSMS, quantum_secure_sum
from qsh.qsmpc.shamir import next_prime, reconstruct, split


def test_shamir_roundtrip():
    shares = split(42, n=5, t=3, prime=97)
    assert reconstruct(shares[:3]) == 42
    assert reconstruct(shares[1:4]) == 42
    assert reconstruct(shares[2:5]) == 42


def test_shamir_threshold_hides_secret():
    # fewer than t shares must not determine the secret: differing reconstructions
    shares = split(7, n=5, t=3, prime=97)
    guesses = {reconstruct(shares[:2] + [type(shares[0])(9, y, 97)]) for y in range(5)}
    assert len(guesses) > 1


def test_qsms_paper_example():
    # Sutradhar 2025 worked example: t=3, n=7, d=11, X=2, Y=3 -> S=5
    res = QSMS(n=7, t=3, prime=11).run([2, 3], seed=1)
    assert res.total == 5
    assert res.honest_statevector
    assert sum(res.masks) % 11 == 0  # GHZ+QFT structure: masks sum to zero


@pytest.mark.parametrize("seed", range(25))
def test_qsms_randomized_correct(seed):
    rng = random.Random(seed)
    secrets = [rng.randint(0, 10) for _ in range(rng.randint(2, 5))]
    assert quantum_secure_sum(secrets, n=7, t=3, prime=11) == sum(secrets) % 11


def test_qsms_analytic_path_for_large_state():
    # large prime forces the analytic sampler (d**t > state cap) but stays correct
    res = QSMS(n=40, t=5, prime=next_prime(60)).run([10, 20, 30], seed=2)
    assert not res.honest_statevector
    assert res.total == 60 % res.modulus


def test_psi_intersection():
    res = private_set_intersection({"a", "b", "c", "x"}, {"c", "x", "z"}, b"k" * 32)
    assert set(res.intersection) == {"c", "x"}
    assert res.intersection_size == 2


def test_multiparty_psi():
    res = multiparty_psi([{1, 2, 3, 9}, {2, 3, 9}, {3, 9, 100}], b"k" * 32)
    assert set(res.intersection) == {3, 9}


def test_qot_receiver_gets_choice():
    for choice in (0, 1):
        t = quantum_oblivious_transfer(b"msg-zero", b"msg-one!", choice=choice, seed=choice)
        assert t.received == (b"msg-one!" if choice else b"msg-zero")
        assert t.test_passed

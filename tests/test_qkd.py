"""Tests for the QKD layer: BB84 sifting, QBER, eavesdropper detection, key derivation."""

from qsh.qkd.bb84 import bb84_circuit_demo, derive_shared_key, run_bb84
from qsh.qkd.channel import ChannelParams, intrinsic_qber


def test_bb84_low_qber_clean_channel():
    r = run_bb84(40_000, ChannelParams(distance_km=25, misalignment=0.01), seed=1)
    assert r.sifted_bits > 0
    assert r.qber < 0.05
    assert r.secret_key_rate > 0


def test_bb84_eavesdropper_raises_qber():
    clean = run_bb84(40_000, ChannelParams(distance_km=25), seed=2)
    eve = run_bb84(40_000, ChannelParams(distance_km=25, eve_intercept_fraction=0.5), seed=2)
    assert eve.qber > clean.qber + 0.05  # intercept-resend is detectable


def test_intrinsic_qber_monotone_in_eavesdropping():
    base = intrinsic_qber(ChannelParams(eve_intercept_fraction=0.0))
    half = intrinsic_qber(ChannelParams(eve_intercept_fraction=0.5))
    assert half > base
    assert half <= 0.5


def test_derive_shared_key_length():
    r = run_bb84(200_000, ChannelParams(distance_km=10, misalignment=0.005), seed=3)
    key = derive_shared_key(r, 32)
    assert len(key) == 32


def test_bb84_circuit_demo_sifts():
    demo = bb84_circuit_demo(12, seed=5)
    assert len(demo["alice_bits"]) == 12
    # matching-basis positions agree between Alice and Bob (noiseless demo)
    for i in range(12):
        if demo["alice_bases"][i] == demo["bob_bases"][i]:
            assert demo["alice_bits"][i] == demo["bob_bits"][i]

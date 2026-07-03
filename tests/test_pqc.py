"""Tests for the PQC layer: hybrid KEM, ML-DSA signatures, AEAD transport."""

import pytest

from qsh.pqc.hybrid_kem import HybridKEMRecipient, encapsulate
from qsh.pqc.signatures import generate_keypair, verify
from qsh.pqc.transport import _mix_qkd, _mix_qkd_its, establish_initiator, establish_responder


def test_hybrid_kem_agreement():
    with HybridKEMRecipient() as bob:
        ct, ss_sender = encapsulate(bob.public_key)
        ss_recipient = bob.decapsulate(ct)
    assert ss_sender == ss_recipient
    assert len(ss_sender) == 32


def test_hybrid_kem_distinct_sessions():
    with HybridKEMRecipient() as bob:
        _, ss1 = encapsulate(bob.public_key)
        _, ss2 = encapsulate(bob.public_key)
    assert ss1 != ss2  # ephemeral X25519 => fresh secret each time


def test_mldsa_sign_verify():
    kp = generate_keypair()
    try:
        msg = b"orchestration-stage-complete"
        sig = kp.sign(msg)
        assert verify(msg, sig, kp.public_key)
        assert not verify(b"tampered", sig, kp.public_key)
    finally:
        kp.free()


@pytest.mark.parametrize("qkd_key", [None, b"q" * 32])
def test_transport_roundtrip(qkd_key):
    with HybridKEMRecipient() as bob:
        ct, chan_a = establish_initiator(bob.public_key, qkd_key=qkd_key)
        chan_b = establish_responder(bob, ct, qkd_key=qkd_key)
        msg = b"private cross-bank record"
        rec = chan_a.seal(msg, b"header")
        assert chan_b.open(rec, b"header") == msg


def test_transport_tamper_detected():
    with HybridKEMRecipient() as bob:
        ct, chan_a = establish_initiator(bob.public_key)
        chan_b = establish_responder(bob, ct)
        rec = bytearray(chan_a.seal(b"x" * 16))
        rec[-1] ^= 0x01  # flip a ciphertext bit
        with pytest.raises(Exception):
            chan_b.open(bytes(rec))


# --- Round 16: the config-gated PQC+QKD hybrid combiner (HKDF vs information-theoretic Toeplitz+LHL) ---

@pytest.mark.parametrize("mode", ["hkdf", "its_toeplitz"])
def test_hybrid_combiner_modes_agree(mode):
    """Both PQC+QKD combiners derive an identical 32-byte session key on both sides (byte-compatible channel)."""
    qkd_key = b"Q" * 32
    with HybridKEMRecipient() as bob:
        ct, chan_a = establish_initiator(bob.public_key, qkd_key=qkd_key, combiner_mode=mode)
        chan_b = establish_responder(bob, ct, qkd_key=qkd_key, combiner_mode=mode)
    assert chan_a.session_key == chan_b.session_key
    assert len(chan_a.session_key) == 32
    assert chan_b.open(chan_a.seal(b"msg")) == b"msg"


def test_its_combiner_deterministic_and_distinct():
    """The information-theoretic Toeplitz+LHL "secure-if-either" combiner is deterministic and genuinely
    different from the default HKDF (an extractor over pqc||qkd, not a plain KDF); default stays passthrough."""
    pqc, qkd = b"P" * 32, b"Q" * 32
    assert _mix_qkd_its(pqc, qkd) == _mix_qkd_its(pqc, qkd) and len(_mix_qkd_its(pqc, qkd)) == 32
    assert _mix_qkd(pqc, qkd, mode="hkdf") != _mix_qkd(pqc, qkd, mode="its_toeplitz")
    assert _mix_qkd(pqc, None) == pqc                                   # no QKD key -> passthrough

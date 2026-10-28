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


# --------------------------------------------------------------------------------------------- #
#  FIPS 203 input checks on the Python backend (Round 75)
# --------------------------------------------------------------------------------------------- #
# liboqs-python copies an input into a buffer of the expected size, so a SHORT key or ciphertext
# was zero-padded and used: a 1183-byte encapsulation key was encapsulated to, and a 1087-byte
# ciphertext decapsulated to a key. FIPS 203 sections 7.2 and 7.3 require both to be rejected
# before anything else. `check:pqc` held only the browser library to this.

import gzip  # noqa: E402
from pathlib import Path  # noqa: E402

from qsh.pqc.hybrid_kem import HybridCiphertext, HybridPublicKey  # noqa: E402

VECTORS = Path(__file__).resolve().parents[1] / "web" / "scripts" / "vectors"


@pytest.mark.parametrize("delta", [-1, +1])
def test_encapsulate_refuses_a_key_of_the_wrong_length(delta: int) -> None:
    with HybridKEMRecipient() as bob:
        pub = bob.public_key
        ek = pub.mlkem_public[:delta] if delta < 0 else pub.mlkem_public + b"\x00"
        with pytest.raises(ValueError, match="type check"):
            encapsulate(HybridPublicKey(pub.kem_name, ek, pub.x25519_public))


@pytest.mark.parametrize("delta", [-1, +1])
def test_decapsulate_refuses_a_ciphertext_of_the_wrong_length(delta: int) -> None:
    with HybridKEMRecipient() as bob:
        ct, _ = encapsulate(bob.public_key)
        c1 = ct.mlkem_ciphertext[:delta] if delta < 0 else ct.mlkem_ciphertext + b"\x00"
        with pytest.raises(ValueError, match="type check"):
            bob.decapsulate(HybridCiphertext(c1, ct.x25519_ephemeral_public))


@pytest.mark.parametrize("kem_name", ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"])
def test_encapsulate_refuses_every_c2sp_modulus_vector(kem_name: str) -> None:
    """The same 2,595 malformed keys `check:pqc` gives the browser, through the Python path."""
    keys = gzip.decompress((VECTORS / f"{kem_name}.txt.gz").read_bytes()).decode().split()
    assert len(keys) > 500, f"only {len(keys)} vectors read for {kem_name}"
    with HybridKEMRecipient(kem_name) as bob:
        x_pub = bob.public_key.x25519_public
        accepted = []
        for i, hexkey in enumerate(keys):
            try:
                encapsulate(HybridPublicKey(kem_name, bytes.fromhex(hexkey), x_pub))
                accepted.append(i)
            except (RuntimeError, ValueError):
                pass
    assert not accepted, f"{kem_name}: {len(accepted)} malformed keys accepted, first {accepted[:5]}"


def test_a_channel_refuses_to_seal_past_its_nonce_range() -> None:
    """The MCP bridge keeps the caller below 2**40 so its nonces never meet the reply's."""
    from qsh.pqc.mcp_bridge import _RESPONDER_NONCE_BASE
    from qsh.pqc.transport import SecureChannel
    chan = SecureChannel(b"K" * 32, _counter=_RESPONDER_NONCE_BASE - 1, _limit=_RESPONDER_NONCE_BASE)
    chan.seal(b"last one allowed")
    with pytest.raises(RuntimeError, match="nonce range exhausted"):
        chan.seal(b"would reuse the responder's first nonce")

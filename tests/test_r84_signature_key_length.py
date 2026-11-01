"""ML-DSA verification rejects a public key of the wrong length before liboqs sees it.

WHY THIS EXISTS (Round 84). Advisory GHSA-w696-4vrr-x6x3 (2026-10-28): liboqs-python up to 0.16.0.1
zero-pads a short public key to the parameter set's length and verifies against that, and an ML-DSA
key of 32 bytes or less accepts forged signatures. Round 75 added the length check to the KEM only;
`signatures.verify` passed any key straight through. The second test simulates the vulnerable wrapper
(its verify returns True for anything), so it fails on the code before the fix whatever version of
liboqs-python is installed.
"""

from __future__ import annotations

import pytest

oqs = pytest.importorskip("oqs")

from qsh.pqc import signatures  # noqa: E402


def test_a_genuine_signature_still_verifies() -> None:
    kp = signatures.generate_keypair()
    try:
        sig = kp.sign(b"round 84")
        assert signatures.verify(b"round 84", sig, kp.public_key)
        assert not signatures.verify(b"round 85", sig, kp.public_key)
    finally:
        kp.free()


@pytest.mark.parametrize("bad_key", [b"", b"\x00" * 32, b"\x00" * 1951])
def test_a_short_public_key_is_rejected_even_by_a_wrapper_that_accepts_it(monkeypatch, bad_key) -> None:
    monkeypatch.setattr(oqs.Signature, "verify", lambda self, m, s, pk: True)
    assert signatures.verify(b"forged", b"\x01" * 100, bad_key) is False


def test_an_oversized_signature_is_rejected(monkeypatch) -> None:
    kp = signatures.generate_keypair()
    try:
        monkeypatch.setattr(oqs.Signature, "verify", lambda self, m, s, pk: True)
        limit = oqs.Signature(signatures.DEFAULT_SIG).details["length_signature"]
        assert signatures.verify(b"x", b"\x01" * (limit + 1), kp.public_key) is False
    finally:
        kp.free()

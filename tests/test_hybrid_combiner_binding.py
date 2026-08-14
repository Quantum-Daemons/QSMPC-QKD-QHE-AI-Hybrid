"""The hybrid combiner must bind the ML-KEM ciphertext, and nothing was making it.

WHY THIS EXISTS

Round 38 found that all four of this project's hybrid-KEM combiner sites bound the encapsulation
keys and omitted `c1`, the ML-KEM ciphertext - and that the MCP one bound nothing at all, which is
expression (16) of NIST SP 800-227, the combiner that document names as NOT preserving IND-CCA.

Section 4.6.3 defines the preserving form as

    KeyCombineCCA_H:  H(K1, K2, c1, c2, ek1, ek2, domain_sep)

and states of the reference it comes from that it "does not incorporate encapsulation keys into the
combiner, as this is not needed to achieve the IND-CCA-preserving property". The ciphertexts are the
load-bearing inputs; the encapsulation keys are optional, useful for binding the secret to party
identities.

THE POINT OF THIS FILE, WHICH IS NOT THE FIX

The fix was demonstrated at the time by a throwaway script: with the old transcript, a `c1` mauled by
one bit that still decapsulates to the same `K1` derived a byte-identical session key; with `c1`
bound it derives a different one. That demonstration was recorded as PROSE in the verification log
and the script was never committed. An adversarial review then pointed out the consequence: reverting
`encapsulate`/`decapsulate` to the pre-Round-38 transcript left every test in the suite green.
`test_hybrid_kem_agreement` passes because both sides change together; `test_hybrid_kem_distinct_
sessions` passes because the X25519 ephemeral is still in the transcript. The property the change
exists to establish was unenforced.

That is this round's own lesson applied to its own work: a demonstration is not a guard, and a
guard nobody can point at is a claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

pytest.importorskip("oqs", reason="liboqs-python is required for the hybrid KEM")

from qsh.pqc.hybrid_kem import (  # noqa: E402
    HybridKEMRecipient,
    _combine,
    _combiner_info,
    _length_tag,
    encapsulate,
)


def _parts(recipient: HybridKEMRecipient, ct):
    """The four transcript elements and the two shared secrets, as the combiner sees them."""
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey

    k1 = recipient._kem.decap_secret(ct.mlkem_ciphertext)
    k2 = recipient._x_priv.exchange(
        X25519PublicKey.from_public_bytes(ct.x25519_ephemeral_public))
    return k1, k2, ct.mlkem_ciphertext, ct.x25519_ephemeral_public, \
        recipient._mlkem_public, recipient._x_pub


def test_the_key_encapsulate_returns_is_the_one_c1_is_bound_into() -> None:
    """Pin the transcript the PRODUCTION path builds, not one this test builds.

    The first version of this file constructed `c1 + c2 + ek1 + ek2` itself and asserted that
    mauling `c1` changed `_combine`'s output. It passed on the pre-Round-38 code, because it was
    testing `_combine` given a transcript handed to it rather than testing what `encapsulate`
    actually binds - the same defect this round spent its length finding elsewhere, committed in the
    test written to prevent it. Caught by reverting the combiner and watching it stay green.

    So: derive the key both ways from the real ciphertext, and assert which one `encapsulate`
    agrees with. That fails the moment the production transcript changes shape.
    """
    with HybridKEMRecipient() as r:
        ct, actual = encapsulate(r.public_key)
        k1, k2, c1, c2, ek1, ek2 = _parts(r, ct)
        # `_combine` appends the length tag itself; passing it here too double-tags.
        info = _combiner_info(r.kem_name)

        standard = _combine(k1, k2, c1 + c2 + ek1 + ek2, 32, info)      # SP 800-227 s4.6.3 order
        pre_round_38 = _combine(k1, k2, ek1 + ek2 + c2, 32, info)       # what it used to bind

        assert actual == standard, (
            "encapsulate() does not derive its key over c1 || c2 || ek1 || ek2. All four combiner "
            "sites must agree on that transcript or backend and browser derive different keys."
        )
        assert actual != pre_round_38, (
            "encapsulate() still derives the key the pre-Round-38 transcript gives - the ML-KEM "
            "ciphertext is not being bound."
        )


def test_a_mauled_ciphertext_changes_the_key_the_recipient_derives() -> None:
    """The property section 4.6.3 is about, exercised through `decapsulate`.

    Models the case a hybrid exists to hedge: ML-KEM is the broken component, so an adversary
    supplies a different `c1` that decapsulates to the SAME `K1`. That is simulated by feeding
    `decapsulate` a ciphertext object whose `mlkem_ciphertext` differs while forcing the same K1, so
    the ONLY thing that can change the derived key is whether the transcript observes c1.
    """
    import dataclasses

    with HybridKEMRecipient() as r:
        ct, _ = encapsulate(r.public_key)
        honest = r.decapsulate(ct)

        k1_fixed = r._kem.decap_secret(ct.mlkem_ciphertext)
        mauled_ct = dataclasses.replace(
            ct, mlkem_ciphertext=bytes([ct.mlkem_ciphertext[0] ^ 0x01]) + ct.mlkem_ciphertext[1:])

        # Pin K1 so the ONLY difference between the two derivations is c1 in the transcript.
        original = r._kem.decap_secret
        try:
            r._kem.decap_secret = lambda _c: k1_fixed
            under_mauled = r.decapsulate(mauled_ct)
        finally:
            r._kem.decap_secret = original

        assert honest != under_mauled, (
            "a different ML-KEM ciphertext that decapsulates to the same shared secret produced an "
            "identical session key. That is the non-preservation SP 800-227 section 4.6.3 warns "
            "about: the combiner is not observing c1, so a broken ML-KEM is not contained by a "
            "sound X25519."
        )


def test_encapsulate_and_decapsulate_bind_the_same_bytes() -> None:
    """Both sides must rebuild the identical transcript, or the round trip silently diverges."""
    with HybridKEMRecipient() as r:
        ct, sender_key = encapsulate(r.public_key)
        assert r.decapsulate(ct) == sender_key


@pytest.mark.parametrize("kem_name", ["ML-KEM-768", "ML-KEM-1024"])
def test_the_domain_separator_names_the_parameter_set(kem_name: str) -> None:
    """A category-5 run must not derive its key under a separator asserting category 3.

    Section 4.6.3 requires domain_sep to identify the composite scheme including the "choice of
    parameter set". This was a fixed literal naming ML-KEM-768 while the recipient takes a kem_name
    and the browser's `kemImpl()` genuinely switches to ML-KEM-1024 on the category-5 path.
    """
    info = _combiner_info(kem_name)
    assert kem_name.encode() in info, f"{info!r} does not name {kem_name}"
    other = "ML-KEM-1024" if kem_name == "ML-KEM-768" else "ML-KEM-768"
    assert _combiner_info(other) != info, "both parameter sets share one separator"


def test_the_output_length_is_in_the_separator() -> None:
    """HKDF-Expand for L bytes is a prefix of the expansion for L' > L.

    `length` varies across callers - 32 for transport sessions, `max(32, (2n+7)//8)` for the QOTP
    pad - so without the length in the separator a long pad would begin with exactly the bytes a
    short session key over the same material would.
    """
    info = _combiner_info("ML-KEM-768")
    short = _combine(b"A" * 32, b"B" * 32, b"transcript", 32, info)
    long_ = _combine(b"A" * 32, b"B" * 32, b"transcript", 51, info)
    assert long_[:32] != short, (
        "the 32-byte derivation is a prefix of the 51-byte one over identical material; the output "
        "length is not separating the two derivations"
    )
    assert _length_tag(32) != _length_tag(51)

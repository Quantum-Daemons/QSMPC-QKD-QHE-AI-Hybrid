"""The hybrid KEM refuses what RFC 10024 requires a hybrid group's peer to abort on (Round 86, batch 11).

RFC 10024 (August 2026) section 4.2: a ciphertext whose length does not match the parameter set; section 4.3: an
all-zero X25519 shared secret. `qsh.pqc.hybrid_kem.check_hybrid_inputs` applies those, and holds every transcript
field to its parameter set's width (the concatenation has no length prefixes), at both production call sites; the
browser's `checkHybridInputs` does the same and `web/scripts/check_hybrid_combiner.mjs` holds it there. The combiner
derives with HKDF, not TLS's key schedule, so none of this makes it an RFC 10024 group.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.pqc.hybrid_kem import HYBRID_WIDTHS, check_hybrid_inputs  # noqa: E402


def _valid(kem: str = "ML-KEM-768") -> dict:
    w = HYBRID_WIDTHS[kem]
    return {"kem_name": kem, "ss_pq": bytes(range(32)), "ss_classical": bytes(range(1, 33)), "c1": bytes(w["c1"]),
            "c2": bytes(32), "ek1": bytes(w["ek1"]), "ek2": bytes(32)}


def test_a_valid_exchange_passes_for_both_parameter_sets() -> None:
    for kem in HYBRID_WIDTHS:
        check_hybrid_inputs(**_valid(kem))


@pytest.mark.parametrize(("change", "message"), [
    ({"c1": bytes(1087)}, "ciphertext is 1087 bytes, not 1088 (RFC 10024 section 4.2)"),
    ({"kem_name": "ML-KEM-1024", "ek1": bytes(1568)}, "ciphertext is 1088 bytes, not 1568"),
    ({"ss_classical": bytes(32)}, "all zeros (RFC 10024 section 4.3)"),
    ({"c2": bytes(31)}, "X25519 share"),
    ({"ss_pq": bytes(31)}, "ML-KEM shared secret is 31 bytes"),
    ({"kem_name": "ML-KEM-512"}, "unknown parameter set"),
])
def test_each_abort_is_refused(change: dict, message: str) -> None:
    with pytest.raises(ValueError, match="hybrid KEM refused") as e:
        check_hybrid_inputs(**{**_valid(), **change})
    assert message in str(e.value)


def test_both_production_sites_call_it() -> None:
    src = (REPO / "src" / "qsh" / "pqc" / "hybrid_kem.py").read_text()
    assert src.count("check_hybrid_inputs(") == 3, "the definition and the two call sites (encapsulate, decapsulate)"


def test_the_generator_keeps_each_vector_at_its_widths() -> None:
    import json

    vectors = json.loads((REPO / "web" / "scripts" / "vectors" / "hybrid_combiner_vectors.json").read_text())["vectors"]
    for v in vectors:
        w = HYBRID_WIDTHS[v["kem"]]
        assert (len(v["c1"]) // 2, len(v["ek1"]) // 2) == (w["c1"], w["ek1"]), v["name"]

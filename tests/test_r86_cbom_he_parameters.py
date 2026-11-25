"""Every homomorphic-encryption parameter set the CBOM names stays inside the HE standard's 128-bit limits (Round 86).

The owner's choice of 2026-11-21, the first layer of a CBOM-and-PETs evaluation: the CBOM already says which HE
parameter sets the code uses, so this reads them from the published CBOM itself (`web/public/cbom/qsh.cdx.json`, which
`tests/test_r86_cbom.py` holds to its generator) and checks each ciphertext modulus against the Homomorphic Encryption
Security Standard v1.1 (homomorphicencryption.org, November 2018), ternary secret (-1, 1), 128-bit security: Table 1
(cost model BKZ.sieve, classical) and Table 2 (BKZ.qsieve, post-quantum), both read on the rendered page on 2026-11-21.

The identifier is read as written: an `N = <n>` sets the ring dimension for the moduli that follow it, and a modulus is
written `{a, b, ...} bits`, `Q k x b bits` or `Q a+b+... bits`. On 2026-11-21 the CKKS entry's identifier stopped
mid-phrase ("..., NVIDIA FLARE's") from 2026-11-13 on, so a parameter set it meant to name was half missing from the
published CBOM; the identifiers must now each end on a parameter, and every HE entry must yield at least one set.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CBOM = REPO / "web" / "public" / "cbom" / "qsh.cdx.json"
#: The largest log2 q for a ternary secret at 128-bit security, HE Standard v1.1 (2018).
CLASSICAL_128 = {1024: 27, 2048: 54, 4096: 109, 8192: 218, 16384: 438, 32768: 881}     # Table 1, BKZ.sieve
POST_QUANTUM_128 = {1024: 25, 2048: 51, 4096: 101, 8192: 202, 16384: 411, 32768: 827}  # Table 2, BKZ.qsieve
HE_NAME = re.compile(r"\b(CKKS|BGV|BFV)\b")
TOKEN = re.compile(r"N = (?P<n>\d+)|\{(?P<set>\d+(?:, \d+)*)\} bits|Q (?P<k>\d+) x (?P<b>\d+) bits|Q (?P<sum>\d+(?:\+\d+)+) bits")


def parameter_sets(identifier: str) -> list[tuple[int | None, int, str]]:
    """(N, log2 q, the text it came from) for each modulus written in the identifier, in order."""
    out: list[tuple[int | None, int, str]] = []
    n: int | None = None
    for m in TOKEN.finditer(identifier):
        if m["n"]:
            n = int(m["n"])
        elif m["set"]:
            out.append((n, sum(int(x) for x in m["set"].split(", ")), m.group(0)))
        elif m["k"]:
            out.append((n, int(m["k"]) * int(m["b"]), m.group(0)))
        else:
            out.append((n, sum(int(x) for x in m["sum"].split("+")), m.group(0)))
    return out


def _he_entries(cbom: dict) -> dict[str, str]:
    out = {}
    for c in cbom["components"]:
        props = (c.get("cryptoProperties") or {}).get("algorithmProperties") or {}
        if HE_NAME.search(c.get("name", "")) and props.get("parameterSetIdentifier"):
            out[c["bom-ref"]] = props["parameterSetIdentifier"]
    return out


def _problems(cbom: dict) -> list[str]:
    problems = []
    entries = _he_entries(cbom)
    if not entries:
        problems.append("the CBOM names no HE parameter set")
    for ref, ident in entries.items():
        sets = parameter_sets(ident)
        if not sets:
            problems.append(f"{ref}: no modulus could be read from {ident!r}")
        if re.search(r"(\b\w+'s|,|;|:)\s*$", ident):
            problems.append(f"{ref}: the identifier stops mid-phrase: {ident[-40:]!r}")
        for n, log_q, text in sets:
            if n not in POST_QUANTUM_128:
                problems.append(f"{ref}: {text} has no ring dimension the standard's table lists (N = {n})")
                continue
            if log_q > POST_QUANTUM_128[n]:
                problems.append(f"{ref}: {text} is log2 q = {log_q} at N = {n}, past the post-quantum 128-bit limit "
                                f"{POST_QUANTUM_128[n]} (classical {CLASSICAL_128[n]})")
    return problems


def test_every_he_parameter_set_in_the_cbom_is_inside_the_standard() -> None:
    cbom = json.loads(CBOM.read_text(encoding="utf-8"))
    assert _problems(cbom) == []
    sets = [s for ident in _he_entries(cbom).values() for s in parameter_sets(ident)]
    # What was read, so a parser that silently matched less would show here: the browser's CKKS chain, the shared-key
    # aggregation, threshold CKKS and threshold BGV at N = 8192, the square students' two sets (A2 at N = 8192, B at
    # N = 16384), the satellite QCNN's tree network at N = 16384, and the committee's threshold BGV at N = 16384.
    assert sorted((n, q) for n, q, _ in sets) == [(8192, 120), (8192, 140), (8192, 200), (8192, 200), (8192, 202), (8192, 202),
                                                   (16384, 240), (16384, 260), (16384, 360)], sets


def test_every_bound_the_cbom_quotes_is_the_standards() -> None:
    """The threshold BGV entries quote their limit ("inside the HE standard's post-quantum 202 at N = 8192"): the number
    quoted must be Table 2's for that N."""
    quoted = []
    for ident in _he_entries(json.loads(CBOM.read_text(encoding="utf-8"))).values():
        for m in re.finditer(r"post-quantum (\d+) at N = (\d+)", ident):
            quoted.append((int(m[2]), int(m[1])))
            assert int(m[1]) == POST_QUANTUM_128[int(m[2])], m.group(0)
    assert sorted(quoted) == [(8192, 202), (8192, 202), (16384, 411)]


def test_a_modulus_past_the_limit_is_refused() -> None:
    cbom = json.loads(CBOM.read_text(encoding="utf-8"))
    c = next(c for c in cbom["components"] if c.get("bom-ref") == "crypto/ckks")
    c["cryptoProperties"]["algorithmProperties"]["parameterSetIdentifier"] = "N = 8192, coefficient moduli {60, 60, 60, 60} bits"
    assert any("past the post-quantum 128-bit limit 202" in p for p in _problems(cbom))


def test_an_identifier_that_stops_mid_phrase_is_refused() -> None:
    cbom = json.loads(CBOM.read_text(encoding="utf-8"))
    c = next(c for c in cbom["components"] if c.get("bom-ref") == "crypto/ckks")
    c["cryptoProperties"]["algorithmProperties"]["parameterSetIdentifier"] = (
        "browser: N = 8192, coefficient moduli {60, 40, 40, 60} bits, scale 2^40; the medical page's shared-key "
        "aggregation {60, 40, 40} bits, NVIDIA FLARE's")
    assert any("stops mid-phrase" in p for p in _problems(cbom))


@pytest.mark.parametrize("n", sorted(POST_QUANTUM_128))
def test_the_post_quantum_limit_is_below_the_classical_one(n: int) -> None:
    """A transcription check on the two tables: the quantum cost model always allows less."""
    assert POST_QUANTUM_128[n] < CLASSICAL_128[n]

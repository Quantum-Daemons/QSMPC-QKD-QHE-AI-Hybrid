"""What NIST IR 8547 says of each cryptographic asset in the CBOM, on a stated date (Round 86).

The Python twin of `web/src/lib/ir8547.ts`, which the Validations page's CBOM panel renders; the owner's choice of
2026-11-23. Each asset of `web/public/cbom/qsh.cdx.json` is read against NIST IR 8547, *Transition to Post-Quantum
Cryptography Standards* (final, October 2026, doi:10.6028/NIST.IR.8547), read on the final PDF on 2026-11-23, and
names the table or section it rests on. It is a classification by IR 8547's tables, not a certification: nothing here
is FIPS 140-3 validated. The page's sentences live in the TypeScript; this module carries the structured rows and the
standing on a date, and `tests/test_r86_pqc_readiness.py` holds the two languages to the same verdicts.

IR 8547's dates read "deprecated after 2030" and "disallowed after 2035", so the year decides: 2031-01-01 is the first
deprecated day and 2036-01-01 the first disallowed one.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

CLASSES = ("post-quantum", "quantum-vulnerable", "symmetric", "hybrid", "outside")
STANDINGS = ("post-quantum", "symmetric", "approved-quantum-vulnerable", "quantum-vulnerable", "deprecated", "disallowed",
             "allowed-composite", "not-classified")


@dataclass(frozen=True)
class Ir8547Rule:
    ref: str
    cls: str
    basis: str                      # the table or section; "none" outside IR 8547's tables
    category: int | None = None     # for a hash or XOF, collision resistance's
    preimage_category: int | None = None
    classical_bits: int | None = None
    named: bool | None = None       # None: named unless the basis is "none"
    deprecated_after: int | None = None
    disallowed_after: int | None = None
    quantum_vulnerable: bool | None = None  # None: the class says

    def __post_init__(self) -> None:
        if self.cls not in CLASSES:
            raise ValueError(f"{self.ref}: unknown class {self.cls!r}")
        if self.named is None:
            object.__setattr__(self, "named", self.basis != "none")
        if self.quantum_vulnerable is None:
            object.__setattr__(self, "quantum_vulnerable", self.cls == "quantum-vulnerable")


def _r(ref: str, cls: str, basis: str, **kw) -> Ir8547Rule:
    return Ir8547Rule(f"crypto/{ref}", cls, basis, **kw)


RULES: tuple[Ir8547Rule, ...] = (
    _r("ml-kem-768", "post-quantum", "Table 5", category=3, classical_bits=192),
    _r("ml-kem-1024", "post-quantum", "Table 5", category=5, classical_bits=256),
    _r("x25519", "quantum-vulnerable", "Section 4.1.2", named=False, deprecated_after=2030, disallowed_after=2035),
    _r("ecdh-p256", "quantum-vulnerable", "Table 4", deprecated_after=2030, disallowed_after=2035),
    _r("hkdf-sha256", "symmetric", "Section 4.1.3"),
    _r("hybrid-combiner", "hybrid", "Sections 3.2.1 and 4.1.2"),
    _r("aes-256-gcm", "symmetric", "Table 6", category=5, classical_bits=256),
    _r("chacha20-poly1305", "outside", "none"),
    _r("ml-dsa-65", "post-quantum", "Table 3", category=3, classical_bits=192),
    _r("ml-dsa-87", "post-quantum", "Table 3", category=5, classical_bits=256),
    _r("sha-256", "symmetric", "Table 8", category=2, preimage_category=5, classical_bits=128),
    _r("sha-384", "symmetric", "Table 8", category=4, preimage_category=5, classical_bits=192),
    _r("sha-512", "symmetric", "Table 8", category=5, preimage_category=5, classical_bits=256),
    _r("shake256", "symmetric", "Table 8", category=5, preimage_category=5, classical_bits=256),
    _r("hmac-sha256", "symmetric", "Section 4.1.3", named=False),
    _r("blake2s-256", "outside", "none"),
    _r("oprf-ristretto255", "outside", "none", quantum_vulnerable=True),
    _r("ckks", "outside", "none"),
    _r("ckks-threshold", "outside", "none"),
    _r("bgv-threshold", "outside", "none"),
    _r("bgv-threshold-committee", "outside", "none"),
    _r("wrk17b", "outside", "none", quantum_vulnerable=True),
    _r("shamir", "outside", "none"),
    _r("csprng", "outside", "none"),
)

BY_REF = {r.ref: r for r in RULES}


def year_of(iso_date: str) -> int:
    m = re.fullmatch(r"(\d{4})-\d{2}-\d{2}", iso_date)
    if not m:
        raise ValueError(f"not an ISO date: {iso_date}")
    return int(m[1])


def standing_on(r: Ir8547Rule, iso_date: str) -> tuple[str, str]:
    """(standing, label) on the date, as `standingOn` in the TypeScript gives them."""
    year = year_of(iso_date)
    if r.deprecated_after is not None:
        via = "" if r.named else " (section 4.1.2)"
        if r.disallowed_after is not None and year > r.disallowed_after:
            return "disallowed", f"Disallowed{via}"
        if year > r.deprecated_after:
            return "deprecated", f"Deprecated{via}"
        if r.named:
            return "approved-quantum-vulnerable", f"Listed as approved; deprecated after {r.deprecated_after}"
        return "quantum-vulnerable", f"Not named; deprecated after {r.deprecated_after} (section 4.1.2)"
    if r.cls == "post-quantum":
        return "post-quantum", "Post-quantum; no transition date"
    if r.cls == "symmetric":
        if r.category is None:
            cat = "Category 1 or above"
        elif r.preimage_category is None:
            cat = f"Category {r.category}"
        else:
            cat = f"Category {r.category} (collision), {r.preimage_category} (preimage)"
        return "symmetric", f"{cat}; no transition date"
    if r.cls == "hybrid":
        return "allowed-composite", "Allowed as a composite (section 3.2.1); not validated"
    return "not-classified", "Not classified; quantum-vulnerable" if r.quantum_vulnerable else "Not classified"


def standing_counts(iso_date: str, refs: list[str] | None = None) -> dict[str, int]:
    counts = Counter(standing_on(BY_REF[ref], iso_date)[0] for ref in (refs if refs is not None else BY_REF) if ref in BY_REF)
    return {s: counts.get(s, 0) for s in STANDINGS}

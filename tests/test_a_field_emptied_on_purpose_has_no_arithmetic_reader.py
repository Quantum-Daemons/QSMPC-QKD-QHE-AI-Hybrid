"""Emptying a field is only half a change; the other half is everything that read it.

WHY THIS EXISTS

Round 42 found a session TOTAL sitting in ETSI GS QKD 015's `skr` and `eskr`, which the spec defines
as a secret key RATE in bit/s, and set both to `null` - correctly, because this simulation has no
wall-clock model of the quantum channel. It did not look for readers.

`web/src/lib/clientSdnController.ts` still computed

    stored_key_count: Math.floor((link?.eskr ?? 0) / QKD014_KEY_SIZE_BITS)

so from Round 42 onward the GS QKD 014 status table advertises a key management entity holding
**exactly zero keys, on every run, forever**. `null ?? 0` is 0, `floor(0 / 352)` is 0, and nothing
raised, nothing logged, and no test failed.

WHY NOTHING CAUGHT IT

- `tests/test_sdn_fields_carry_the_dimension_the_spec_says.py` asserts on ASSIGNMENTS to `skr`/`eskr`
  and on `allocate()`. It never asks who READS them.
- `tests/test_sdn_qkd.py` does assert `stored_key_count > 0`, but against the **Python**
  `KeyManagementEntity` in `src/qsh/qkd/kme.py`, which is a different implementation and was not
  changed. A gate pointed at the twin of the thing that broke.
- Round 43 walked this exact panel on the deployed site. It recorded `skr (bit/s)` rendering `n/a`
  and `key budget 1,436 bit`, and did not record the `0` sitting between them. **Reading a panel for
  the thing you changed is not reading the panel.**

THE INVARIANT, STATED GENERALLY

`skr` and `eskr` are `number | null` and null is their designed value here. So any arithmetic that
reaches them through a nullish fallback silently yields the fallback - which is indistinguishable
from a real measurement of zero. That is the class, not the instance: **a field emptied on purpose
must have no arithmetic reader that can quietly substitute a number for its absence.**

The check is therefore on the READING side, which is the side Round 42 did not change, and the side
this file was written to add.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TS = REPO / "web" / "src" / "lib" / "clientSdnController.ts"
PANEL = REPO / "web" / "src" / "components" / "SdnControlPlanePanel.tsx"

#: The fields Round 42 emptied on purpose. Their declared type is `number | null`.
EMPTIED = ("skr", "eskr")

_COMMENT = re.compile(r"/\*[\s\S]*?\*/|//[^\n]*")


def _code_only(text: str) -> str:
    """Comments blanked, line count preserved.

    The comments in this file quote `eskr` while explaining the Round 42 decision, so a scan that
    reads them would find the defect in the prose describing it. Round 42's own frame gate shipped
    exactly that and was repaired the same way.
    """
    return _COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def test_the_emptied_fields_are_still_declared_nullable() -> None:
    """If they stop being null this check has no subject, and that is a decision to make openly."""
    src = _code_only(TS.read_text())
    for field in EMPTIED:
        assert re.search(rf"\b{field}:\s*number\s*\|\s*null", src), (
            f"`{field}` is no longer declared `number | null`. If a real rate now fills it, this "
            "gate's premise is gone - delete it deliberately rather than letting it pass vacuously."
        )


def test_no_arithmetic_reads_an_emptied_field_through_a_fallback() -> None:
    """`null ?? 0` in a division is a measurement of zero that nobody measured."""
    src = _code_only(TS.read_text())
    offenders: list[str] = []
    for field in EMPTIED:
        # `x?.eskr ?? 0`, `(l.skr || 0)`, `Number(link.eskr ?? 0)` - the fallback is the tell.
        for m in re.finditer(rf"[\w?.\[\]]*\.{field}\s*(\?\?|\|\|)\s*[-\d]", src):
            line = src.count("\n", 0, m.start()) + 1
            offenders.append(f"{TS.name}:{line}: {m.group(0).strip()!r}")
    assert not offenders, (
        "these expressions substitute a number for a field that is null BY DESIGN:\n  "
        + "\n  ".join(offenders)
        + "\n\n`skr`/`eskr` carry null because this simulation has no wall-clock model of the "
        "quantum channel. A nullish fallback turns that absence into a value - `stored_key_count` "
        "read `(link?.eskr ?? 0) / 352` and advertised zero stored keys on every run from Round 42 "
        "onward. Derive from a quantity that is actually a bit count, such as `key_budget_bits`."
    )


def test_stored_key_count_comes_from_a_bit_denominated_quantity() -> None:
    """The specific consumer, pinned where it lives.

    Kept beside the general check because a class-level gate going green tells you nothing about
    whether THIS field was repaired or merely deleted.
    """
    src = _code_only(TS.read_text())
    m = re.search(r"stored_key_count:\s*([^,\n]+)", src)
    assert m, "`stored_key_count` is no longer emitted; the GS QKD 014 status table has lost a field"
    expr = m.group(1)
    for field in EMPTIED:
        # WORD BOUNDARY. `skr` is a substring of `eskr`, so a plain `in` reported the wrong field as
        # the culprit - it named `skr` for an expression that reads `eskr`. A gate that misattributes
        # sends the reader to the wrong line, which is its own small version of this round's subject.
        assert not re.search(rf"\b{field}\b", expr), (
            f"`stored_key_count` is derived from `{field}`, which is null here: {expr.strip()!r}. "
            "A key COUNT divides a bit count by a key size; deriving it from a rate field was the "
            "dimensional error one layer below the one Round 42 displayed."
        )
    assert "key_budget_bits" in expr, (
        f"`stored_key_count` is not derived from `key_budget_bits`: {expr.strip()!r}. That is the "
        "quantity this run actually produced, in bits, and the one the panel already renders with "
        "its unit."
    )


def test_the_panel_renders_the_count_it_is_given() -> None:
    """An independent surface, so the two cannot be satisfied by the same edit."""
    src = PANEL.read_text()
    assert "stored_key_count" in src, (
        "the panel no longer renders `stored_key_count`; it is the field a reader uses to judge "
        "whether the KME has key material at all"
    )

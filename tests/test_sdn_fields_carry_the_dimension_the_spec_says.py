"""A spec field name vouches for the quantity in it, so the quantity must match.

WHY THIS EXISTS

`SdnControlPlanePanel` rendered `skr 1915.0` and `eskr 1915.0`. ETSI GS QKD 015 defines both as a
secret key RATE, and this repository's own Python mirror says so in the type it publishes -
`src/qsh/qkd/sdn_controller.py`: ``skr: float = 0.0  #: secret key rate, bit/s`` - and logs
``SKR {skr:.1f} bit/s``. The browser was passing `secure_key_bits`, a session TOTAL of 1915 bits,
into a parameter called `skrBitsPerSec`.

The caller had reasoned about it correctly and still got there. `UseCasePage` comments that
`secure_key_bits` is what an allocation budget is denominated in and that `metrics.skr` is per-pulse
and is not - then handed the total to a parameter whose name asserts a rate. One layer down,
`allocate()` subtracted granted bits from `eskr`, so the rate field was serving as the budget ledger:
the wrong dimension was not a display slip, it was the data model.

The checklist already carried the sentence that condemns this, about these same two quantities:
*"`secure_key_bits` must equal `skr` times the pulse count, NOT `skr` itself ... Dimensional
mismatch is the check: if two quantities must be equal, they must first have the same units."* It
was written about a different pair of surfaces and nothing applied it here. (Since Round 80 the two
are no longer EQUAL - `secure_key_bits` is a block's composable finite key, below `skr` times the
pulses - and the dimensional point stands: one is bits, the other bits per pulse.)

WHY NO RATE IS COMPUTED INSTEAD

Bits per second needs a pulse repetition rate. `quantumSim.ts` computes
`secretKeyRate = (secureFraction * nSift) / nPulses` - per pulse - and `pulses` is a count; there is
no wall-clock model of the quantum channel anywhere. A repetition rate is a property of apparatus,
and `README.md` states that no QKD device exists here, calling that "a boundary of the deliverable,
not a gap in it". Surveyed 2026-08-15: TNO-Quantum `communication.qkd_key_rate` (Apache-2.0) is
per-pulse, QKDNetSim is GPL-2.0 and already excluded from the commercial-OK set, SeQUeNCe would
still require asserting a clock. No library supplies a repetition rate because none can.

So the fields stay named as the spec names them and carry `null`, and the bits are rendered in a
column that says `bit`.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TS = REPO / "web" / "src" / "lib" / "clientSdnController.ts"
PY = REPO / "src" / "qsh" / "qkd" / "sdn_controller.py"
PANEL = REPO / "web" / "src" / "components" / "SdnControlPlanePanel.tsx"


def test_the_python_mirror_still_documents_the_unit() -> None:
    """The reference the browser must not contradict. If this moves, the pair must move together."""
    src = PY.read_text()
    assert re.search(r"skr:\s*float.*#:.*bit/s", src), (
        "sdn_controller.py no longer documents `skr` as bit/s; the dimensional reference this test "
        "compares the browser against has moved, so re-derive it rather than deleting the check"
    )


def test_the_python_ledger_is_denominated_in_bits_too() -> None:
    """The SAME assertion as the browser's, against the file the browser is compared to.

    ROUND 44. Round 42 removed `eskr -= granted_bits` from `clientSdnController.ts` and wrote this
    file to keep it out. It then read `PY` for one thing only - the `#: bit/s` unit comment - so
    `src/qsh/qkd/sdn_controller.py` went on budgeting against `eskr` and accumulating granted BITS
    into `expected_consumption`, both documented bit/s, for two more rounds.

    That file is the DIMENSIONAL REFERENCE this gate cites when it judges the browser. A reference
    that is wrong makes the comparison worthless, so "it has no production caller" was never a reason
    to leave it - and "we fixed the language the defect was reported in" is the boundary-with-one-side
    this project keeps finding.

    Both fields are now read-only properties, so the assignment cannot be written at all; this
    asserts the ledger arithmetic that remains.
    """
    src = PY.read_text()
    alloc = src[src.index("    def allocate("):]
    alloc = alloc[: alloc.index("\n    def ", 10)] if "\n    def " in alloc[10:] else alloc
    assert "key_budget_bits" in alloc, (
        "the Python allocate() no longer draws down a bit-denominated budget"
    )
    for rate_field in ("eskr", "expected_consumption"):
        assert f"link.{rate_field} =" not in alloc and f"link.{rate_field} +=" not in alloc, (
            f"the Python allocate() writes `{rate_field}`, a bit/s field, from a bit count - the "
            "exact operation Round 42 removed from the browser and left standing here"
        )


def test_the_browser_never_puts_a_bit_count_in_a_rate_field() -> None:
    """The defect, asserted on the assignment rather than on the rendered string."""
    src = TS.read_text()
    body = re.sub(r"/\*[\s\S]*?\*/", "", src)          # comments explain the defect; do not match them
    body = "\n".join(line.split("//")[0] for line in body.split("\n"))

    assert "skrBitsPerSec" not in body, (
        "`skrBitsPerSec` is back. The caller passes `secure_key_bits`, a total, so a parameter named "
        "for a rate makes the field name vouch for the wrong dimension."
    )
    # OBJECT-LITERAL assignments only. The first version matched `skr: number | null` in the
    # interface and reported the type annotation as a bit count - a false positive in the check
    # written to catch a false quantity. The declaration is the thing that makes `null` legal.
    body = re.sub(r"skr:\s*number\s*\|\s*null;", "", body)
    body = re.sub(r"eskr:\s*number\s*\|\s*null;", "", body)
    bad = re.findall(r"\bskr:\s*(?!null\b)(\w+)", body) + re.findall(r"\beskr:\s*(?!null\b)(\w+)", body)
    assert not bad, (
        f"skr/eskr assigned from {sorted(set(bad))}. This simulation has no wall-clock model of the "
        "quantum channel, so it can produce no bit/s figure; the ETSI rate fields carry null and the "
        "bits are rendered in their own column. Filling them again puts a total under a spec field "
        "name that says rate."
    )


def test_the_allocation_ledger_is_denominated_in_bits() -> None:
    """The layer below the display, which is where the dimension actually lived.

    `allocate()` subtracted granted bits from `eskr`. A gate that only checked what the panel printed
    would have passed a data model that decrements a rate.
    """
    body = re.sub(r"/\*[\s\S]*?\*/", "", TS.read_text())
    body = "\n".join(line.split("//")[0] for line in body.split("\n"))
    alloc = body[body.index("export function allocate"):]
    alloc = alloc[: alloc.index("\n}")]
    assert "key_budget_bits" in alloc, "allocate() no longer draws down the bit-denominated budget"
    assert "eskr" not in alloc, "allocate() is decrementing a rate field again"


def test_the_panel_renders_the_bits_with_a_unit() -> None:
    """A number with no unit is the readable form of the same defect."""
    src = PANEL.read_text()
    assert re.search(r"key_budget_bits[^\n]*bit", src), (
        "the key budget renders without a unit; `skr 1915.0` with no unit is what a reader took for "
        "bits per second in the first place"
    )

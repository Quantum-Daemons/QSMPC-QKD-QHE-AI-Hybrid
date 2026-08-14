"""An agreement rate must never be published without the part of its denominator that carries no signal.

WHY THIS EXISTS

`encrypted_agreement` is the headline claim of this project's private-inference figure: the encrypted
computation reproduces the plaintext one. On genomics it is 23% noise. Seven of that page's thirty
fixed rows carry no effect allele at any of the eight PGS loci, so the plaintext model scores them
exactly 0 on both classes; each flips on ~1e-9 of encryption noise, independently, every run. The
same page printed 3, 5 and 6 mismatches across identical runs, and the rate ranged 0.83 to 0.97 by
chance. Over the 23 rows the model actually decides, agreement is 1.00 - which is both the stronger
claim and the true one.

So three companions have to travel with every agreement rate: `encrypted_undecidable_rows`,
`encrypted_decidable_rows`, `encrypted_agreement_over_decidable`.

WHY A COUNT-PARITY TEST RATHER THAN A PROSE ROW

Round 38 shipped that disclosure **four times** and got the scope wrong on three of them:

  1. First it went into the finance/genomics/medical batch and not into telecom's, which has its own
     copy of the flip logic. Which page you opened decided whether you were told.
  2. Then into the metrics table and not into the chart caption, so a reader looking at the figure
     got the count without the context.
  3. Then into telecom's metrics and caption but not its LOG.
  4. And the caption was gated on `undecidableRows > 0`, which hid it on telecom - where **zero** is
     the informative value, because it is what makes that page's 1.00 a different claim from
     genomics' 1.00.

Every one of those is the same defect as the one being fixed: a disclosure that is conditional on
something other than whether there is anything to disclose. Fixing instances one at a time did not
stop it recurring, so this asserts the invariant instead.

Counts rather than positions, deliberately: a new batch path that emits an agreement rate without the
companions moves one count and not the others, and the assertion fails naming the gap. It does not
care where in the file the calls sit or how the surrounding branches are written.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "web" / "src" / "lib" / "clientCompute.ts"

#: The rate, and the three figures that say what it is a rate over.
AGREEMENT = "encrypted_agreement"
COMPANIONS = (
    "encrypted_agreement_rows",
    "encrypted_undecidable_rows",
    "encrypted_decidable_rows",
    "encrypted_agreement_over_decidable",
)


def _emit_count(key: str, text: str) -> int:
    """How many `metric(stage, "<key>", ...)` call sites exist, matching the key EXACTLY.

    Exact-match matters: `encrypted_agreement` is a prefix of three of its own companions, so a
    substring count would report the rate as emitted five times and pass by accident.
    """
    return len(re.findall(rf'metric\(\s*stage\s*,\s*"{re.escape(key)}"\s*,', text))


def test_the_agreement_metric_is_still_emitted() -> None:
    """A parse that finds nothing would pass every assertion below for the wrong reason."""
    n = _emit_count(AGREEMENT, SRC.read_text())
    assert n >= 2, (
        f"found {n} `encrypted_agreement` emissions; this project has at least two batch paths "
        "(the finance/genomics/medical corpus batch and the telecom held-out split). Did the "
        "metric move, or did the regex stop matching?"
    )


def test_every_agreement_rate_travels_with_its_denominator() -> None:
    text = SRC.read_text()
    rate = _emit_count(AGREEMENT, text)
    gaps = {c: _emit_count(c, text) for c in COMPANIONS if _emit_count(c, text) != rate}
    assert not gaps, (
        f"`encrypted_agreement` is emitted from {rate} call sites, but "
        + ", ".join(f"`{k}` from {v}" for k, v in gaps.items())
        + ". A path that publishes an agreement rate without the count of rows the plaintext model "
        "cannot decide publishes a number that is partly a coin toss, and on genomics that is 23% "
        "of the denominator. Emit all four from the same block, unconditionally - the count is a "
        "property of the corpus and the model, not of the run, so `if (mismatches > 0)` is the "
        "wrong shape and so is `> 0`."
    )


def test_the_disclosure_is_not_gated_on_being_non_zero() -> None:
    """Zero undecidable rows is the informative case, not the case to suppress.

    Telecom reads 0 of 512 and that is exactly what makes its 1.00 mean something. The chart caption
    gated this on `> 0` and rendered nothing there, which is how a page with a genuinely clean sweep
    ended up making a weaker-looking claim than one whose rate is a coin toss.
    """
    chart = (REPO / "web" / "src" / "components" / "VerificationChart.tsx").read_text()
    assert "undecidableRows" in chart, "the caption no longer mentions the undecidable count"
    # The `> 0` test may appear as a WORDING branch; it must not be the sole gate on rendering
    # anything at all, so a zero-branch has to exist beside it.
    assert re.search(r"undecidableRows\s*!=\s*null", chart), (
        "the caption's undecidable disclosure is not guarded on presence. If it is guarded on "
        "`> 0` alone, a page with no ties renders nothing - and no ties is the strongest thing "
        "this figure can report."
    )
    assert re.search(r"None</strong> of these", chart), (
        "the caption has no zero-undecidable branch. Telecom reads 0 of 512, and suppressing that "
        "makes its clean sweep indistinguishable from a page that simply did not say."
    )


def test_no_plot_field_is_set_but_never_rendered() -> None:
    """Every field `EncryptedAgreementPlot` declares must be READ by the component that owns it.

    This is the assertion the first version of this file was missing, and the miss cost a real
    regression. Removing the tie verdict from the caption also deleted the ciphertext-cap sentence -
    the removed block's closing `) : null}` merged with the tie block's - so `rowsOffered` was left
    declared in the interface, set by four producers, and read by nothing. TypeScript cannot see it:
    an optional interface field that no one reads is not an error. The result was that telecom's
    figure reported agreement 1.00 over 512 rows without saying the split held 600, silently
    reverting a fix an earlier round had made for exactly that reading.

    A field that a producer bothers to compute and pass is a field someone decided the reader needed.
    If the component stops reading it, either the component regressed or the field should go - and
    both are worth failing over.
    """
    chart = (REPO / "web" / "src" / "components" / "VerificationChart.tsx").read_text()

    m = re.search(r"export interface EncryptedAgreementPlot \{(.*?)\n\}", chart, re.S)
    assert m, "EncryptedAgreementPlot no longer parses - did the interface move or get renamed?"
    body = m.group(1)
    # Field names at the interface's top level, comments and doc blocks stripped.
    fields = re.findall(r"^\s{2}(\w+)\??:", body, re.M)
    assert len(fields) >= 6, f"only parsed {fields}; the interface should declare at least six fields"

    # Everything after the interface is the component. A field is "read" if it appears as `plot.<f>`.
    component = chart[m.end():]
    unread = [f for f in fields if f != "kind" and f"plot.{f}" not in component]
    assert not unread, (
        f"{unread} are declared on EncryptedAgreementPlot and never read as `plot.<field>` in the "
        "component. Either a producer is computing something the figure silently drops - which is "
        "how the ciphertext-cap disclosure disappeared from the caption while remaining in the "
        "metrics and the log - or the field is dead and should be removed. tsc will not tell you: "
        "an unread optional field is not a type error."
    )

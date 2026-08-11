"""The XAI panel's caveat matches whether the page it is on has an encrypted inference stage.

WHY THIS EXISTS. ``ShapPanel`` rendered one paragraph unconditionally:

    These bars explain the bundled linear model. The class this page reports comes from the
    encrypted CKKS + MPC stage, which evaluates a different model - so read the bars as "what
    drives this model on this input", NOT as an attribution of the reported class.

That is true on finance, genomics and medical. It is **false on telecom and satellite**, whose entries
in ``FLOWS`` contain no ``private_inference`` stage at all: on those two pages the explained model is
the model that produced the reported class, so the bars are precisely the attribution the paragraph
tells the reader they are not. The panel disclaimed a correct attribution on two of five pages.

It survived several rounds of review because **the error points toward caution**. A panel that
under-claims reads as rigour, and this repository's checklist carried the same generalisation - it
enumerated three use cases and then stated the conclusion for five - so the instrument that should
have caught it asserted the same thing.

The fix could have keyed on ``artifacts.primary_inference`` being absent, since only the
``private_inference`` stage writes it. That works today and only by coincidence of which stage happens
to write which artifact. Two shipped defects came from exactly that kind of implicit coupling: the
satellite page scoring a fibre channel because ``config.usecase`` was read but never emitted, and the
free-space model then reading the fibre page's distance keys. So the flag is declared explicitly at
each explanation site, and this test is what stops the declaration and the flow list from drifting
apart - the same guard shape as ``tests/test_run_config_keys.py`` and ``tests/test_federated_scope.py``,
and for the same reason: a claim that must agree with another file is checked, not trusted.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMPUTE = REPO / "web" / "src" / "lib" / "clientCompute.ts"
PANEL = REPO / "web" / "src" / "components" / "ShapPanel.tsx"
USECASE_PAGE = REPO / "web" / "src" / "pages" / "UseCasePage.tsx"

#: Use cases with no page of their own. `demo` is the generic fallback flow, reachable only by an
#: explicit usecase override, so there is no rendered surface for a paragraph to be wrong on.
NO_RENDERED_PAGE = {"demo"}


def _flows() -> dict[str, list[str]]:
    """The stage sequence per use case, parsed from the `FLOWS` literal."""
    text = COMPUTE.read_text()
    block = re.search(r"const FLOWS: Record<string, string\[\]> = \{(.*?)\n\};", text, re.S)
    assert block, "FLOWS not found - did the declaration move or change shape?"
    flows: dict[str, list[str]] = {}
    for line in block.group(1).splitlines():
        m = re.match(r'\s*([a-z_]+):\s*\[(.*)\]', line)
        if m:
            flows[m.group(1)] = re.findall(r'"([^"]+)"', m.group(2))
    assert flows, "parsed no flows out of the FLOWS literal"
    return flows


def _flag_assignments() -> list[bool]:
    """Every `artifacts.explanation_is_of_reported_class = <bool>` in source order."""
    code = "\n".join(re.sub(r"//.*$", "", ln) for ln in COMPUTE.read_text().splitlines())
    return [v == "true" for v in
            re.findall(r"artifacts\.explanation_is_of_reported_class\s*=\s*(true|false)", code)]


def test_the_flag_is_set_at_every_explanation_site() -> None:
    """An explanation with no flag renders the panel's default branch, which may be the wrong one."""
    code = "\n".join(re.sub(r"//.*$", "", ln) for ln in COMPUTE.read_text().splitlines())
    sites = len(re.findall(r"artifacts\.explanation_provenance\s*=", code))
    flags = len(_flag_assignments())
    assert flags == sites, (
        f"{sites} places set `explanation_provenance` but only {flags} set "
        "`explanation_is_of_reported_class`. Every site that produces an explanation must say "
        "whether it explains the model that produced the reported class - an unset flag falls to "
        "the panel's else-branch, which asserts there IS an encrypted stage evaluating something "
        "else, and that is false on any page whose flow has no private_inference."
    )


def test_the_counts_match_the_flows_that_lack_private_inference() -> None:
    """One `true` per page with no encrypted stage, one `false` per page with one."""
    flows = {k: v for k, v in _flows().items() if k not in NO_RENDERED_PAGE}
    without = sorted(k for k, stages in flows.items() if "private_inference" not in stages)
    assert without == ["satellite_comms", "telecom_ids"], (
        f"the set of use cases with no private_inference stage changed to {without}. That is the "
        "condition the panel's paragraph branches on, so ShapPanel and this test both need "
        "updating - and check whether the new page's explanation site sets the flag."
    )
    assignments = _flag_assignments()
    assert assignments.count(True) == len(without), (
        f"{len(without)} use cases have no private_inference stage ({without}) but "
        f"{assignments.count(True)} sites set the flag true. A page that decides without an "
        "encrypted stage and does not set it will tell visitors its correct attribution is not one."
    )
    assert assignments.count(False) >= 1, (
        "no site sets the flag false. The pages WITH an encrypted stage need the original caveat: "
        "their bars explain a different model from the one that produced the reported class."
    )


def test_the_panel_branches_rather_than_asserting_one_case() -> None:
    """The regression itself, pinned: the caveat must not be unconditional again."""
    panel = PANEL.read_text()
    assert "isOfReportedClass" in panel, (
        "ShapPanel no longer takes `isOfReportedClass`. If the paragraph became unconditional "
        "again, that is the defect this file exists for: it disclaims a correct attribution on "
        "telecom and satellite."
    )
    assert re.search(r"\{isOfReportedClass \?", panel), (
        "the prop is present but the paragraph is not branched on it. Reading the flag and then "
        "rendering one paragraph is the same defect with an unused variable."
    )
    assert "explanation_is_of_reported_class" in USECASE_PAGE.read_text(), (
        "UseCasePage stopped passing the artifact, so the prop is permanently undefined and the "
        "panel silently takes the else-branch on every page. An optional prop that is never "
        "supplied does not raise - it just always picks one side."
    )

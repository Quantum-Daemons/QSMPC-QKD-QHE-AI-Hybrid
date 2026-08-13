"""The federated stage is injected only where it has real data and somewhere to be drawn.

WHY THIS EXISTS. The QFL toggle was removed and its state pinned to ``useState(true)`` with no writer,
so ``buildConfig`` set ``qfl`` on every run of every use case and ``runClientPipeline`` injected a
``federated_training`` stage into finance, genomics, telecom, satellite and the generic demo. Three of
those had neither of the two things such a stage needs:

    no render target   ``OrchestrationGraph``'s default flow has no ``federated_training`` node and
                       ``TwoPartyDiagram``'s federated step is medical-only, so the run advanced to a
                       stage no figure contained: the animation stalled on nothing and every diagram
                       row dimmed with nothing highlighted.
    no real data       ``realFederatedSet`` maps only medical and finance to a bundled sample. The
                       other three trained on a synthetic two-cluster toy, and that accuracy was then
                       handed to the agent panel as "federated accuracy" with no qualifier - on pages
                       that never mention federation, while the demo banner told visitors federated
                       training was the medical page's feature.

The fix adds ``FEDERATED_USECASES``, which is a hand-kept list of what exists elsewhere - exactly the
kind of claim that rots. This test reads both sides. It is the same guard shape as
``tests/test_qcnn_usecases.py``, for the same reason: a set that must agree with another file is
checked, not trusted.

AND THEN IT ROTTED ANYWAY, because "both sides" was one side. Everything above under "no render
target" was DESCRIBED here and asserted nowhere - the only reader was ``_has_real_federated_data``.
Finance was later admitted to the set on that data half alone, while ``USECASE_SPECS.finance_aml``
had no federated step and ``OrchestrationGraph.FLOWS`` had no ``finance_aml`` key at all, so it fell
through to ``default``, which has no such node either. Measured on the deployed site: the stage
started at 3.999 s, reported an accuracy of 0.25, and no ``svg`` on the page contained the string
"federat". CI was green throughout, and ``usePipelineRun``'s own comment asserted in prose that
"its diagram renders the step".

A docstring that names a failure mode is not a guard against it.
``test_every_federated_use_case_has_a_render_target`` is the half that was missing.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNHOOK = REPO / "web" / "src" / "lib" / "usePipelineRun.ts"
REALDATA = REPO / "web" / "src" / "lib" / "realData.ts"
DIAGRAM = REPO / "web" / "src" / "components" / "TwoPartyDiagram.tsx"
GRAPH = REPO / "web" / "src" / "components" / "OrchestrationGraph.tsx"


def _declared() -> set[str]:
    m = re.search(r"export const FEDERATED_USECASES = new Set\(\[(.*?)\]\)", RUNHOOK.read_text(), re.S)
    assert m, "FEDERATED_USECASES not found - did the declaration move?"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def _has_real_federated_data() -> set[str]:
    """Use cases `realFederatedSet` can load a bundled sample for."""
    text = REALDATA.read_text()
    m = re.search(r"realFederatedSet[^{]*\{(.*?)\n\}", text, re.S)
    assert m, "realFederatedSet not found"
    # The body maps `usecase: "/data/.../sample.json"`.
    return set(re.findall(r'^\s*([a-z_]+):\s*"/data/', m.group(1), re.M))


def _span(text: str, start: int, opener: str, closer: str) -> tuple[int, int]:
    """Indices of the balanced ``opener``..``closer`` pair beginning at or after ``start``."""
    i = text.index(opener, start)
    depth, j = 0, i
    while j < len(text):
        if text[j] == opener:
            depth += 1
        elif text[j] == closer:
            depth -= 1
            if depth == 0:
                return i, j
        j += 1
    raise AssertionError(f"unbalanced {opener!r} from index {start}")


def _blocks(text: str, header: str, opener: str, closer: str) -> dict[str, str]:
    """Per-use-case bodies of a top-level ``Record<string, ...>`` literal.

    Delimiter-counted rather than regex-matched to the close: the step bodies hold template literals
    with their own braces, so a non-greedy ``.*?`` stops at the first one. The outer object is
    bracketed first so a same-shaped key LATER in the file cannot be mistaken for a use case.
    """
    # From the `= {`, not from the header: `Record<string, { id: string; label: string }[]>` puts an
    # inline type literal between the two, and bracketing that returns the type instead of the value.
    outer_open, outer_close = _span(text, text.index("= {", text.index(header)), "{", "}")
    body = text[outer_open:outer_close]
    out: dict[str, str] = {}
    for m in re.finditer(r"^  ([a-z_]+): " + re.escape(opener), body, re.M):
        i, j = _span(body, m.end() - 1, opener, closer)
        out[m.group(1)] = body[i:j]
    assert out, f"no use-case blocks found in {header!r} - did the declaration move?"
    return out


def _diagram_specs() -> dict[str, str]:
    return _blocks(DIAGRAM.read_text(), "export const USECASE_SPECS", "{", "}")


def _graph_flows() -> dict[str, str]:
    return _blocks(GRAPH.read_text(), "const FLOWS", "[", "]")


def test_every_federated_use_case_has_a_render_target() -> None:
    """The half that was prose. A stage with no figure is a run advancing into nothing."""
    declared = _declared()
    specs, flows = _diagram_specs(), _graph_flows()

    # The diagram's federated step is a shared const so medical and finance cannot disagree about
    # one function; accept either it or an inline stageId, so extracting a second one still passes.
    def drawn_in_diagram(uc: str) -> bool:
        body = specs.get(uc, "")
        return "FEDERATED_STEP" in body or 'stageId: "federated_training"' in body

    missing_diagram = sorted(uc for uc in declared if not drawn_in_diagram(uc))
    missing_graph = sorted(
        uc for uc in declared if 'id: "federated_training"' not in flows.get(uc, "")
    )
    assert not missing_diagram and not missing_graph, (
        "a use case runs federated_training with nowhere to draw it.\n"
        f"  no step in TwoPartyDiagram USECASE_SPECS: {missing_diagram or 'none'}\n"
        f"  no node in OrchestrationGraph FLOWS:      {missing_graph or 'none'}\n"
        "OrchestrationGraph falls back to `default` for an unknown key, and `default` has no "
        "federated node - so a missing key here is silent on the page, not an error."
    )


def test_the_diagram_reads_the_aggregation_it_was_told_rather_than_asserting_one() -> None:
    """The federated label must not name a property of its own choosing.

    It read "pairwise masking (demo keys)" while ``clientQFL`` summed plaintext updates: no masks,
    no pairwise seeds, no key material. The label now renders ``qfl_aggregation``, which the stage
    emits from the value the function returns, so the figure cannot outrun the algorithm.
    """
    text = DIAGRAM.read_text()
    step = re.search(r"const FEDERATED_STEP: Step = \{(.*?)\n\};", text, re.S)
    assert step, "FEDERATED_STEP not found - did it move or get inlined again?"
    assert 'f(m, "qfl_aggregation")' in step.group(1), (
        "the federated step no longer renders the measured aggregation. Any hand-written "
        "description here is a claim about code in another file, which is how the last one became "
        "false."
    )
    compute = (REPO / "web" / "src" / "lib" / "clientCompute.ts").read_text()
    assert 'metric(stage, "qfl_aggregation", q.aggregation)' in compute, (
        "clientCompute no longer publishes qfl_aggregation, so the diagram renders an em dash "
        "where the aggregation should be."
    )


def test_federated_scope_matches_the_use_cases_with_real_data() -> None:
    declared, real = _declared(), _has_real_federated_data()
    assert declared == real, (
        "FEDERATED_USECASES disagrees with the use cases realFederatedSet has a sample for.\n"
        f"  declared but no real data: {sorted(declared - real) or 'none'}\n"
        f"  has real data but not declared: {sorted(real - declared) or 'none'}\n"
        "Declaring a use case without a sample trains a federated round on a synthetic toy and "
        "reports its accuracy as a result; omitting one that has a sample hides real work."
    )


def test_the_flag_is_gated_rather_than_unconditional() -> None:
    """`buildConfig` must not set `qfl` for every use case again."""
    body = re.search(r"const buildConfig = \(\) => \{(.*?)\n  \};", RUNHOOK.read_text(), re.S)
    assert body, "buildConfig not found"
    line = [ln for ln in body.group(1).splitlines() if "c.qfl" in ln]
    assert line, "buildConfig no longer sets c.qfl - if that is deliberate, delete this test"
    assert "runsFederated" in line[0], (
        f"c.qfl is set without the scope predicate: {line[0].strip()!r}. Ungated, the stage is "
        "injected into use cases that have no federated data and no figure to draw it in."
    )

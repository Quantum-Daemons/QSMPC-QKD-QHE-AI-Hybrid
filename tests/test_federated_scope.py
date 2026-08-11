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
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNHOOK = REPO / "web" / "src" / "lib" / "usePipelineRun.ts"
REALDATA = REPO / "web" / "src" / "lib" / "realData.ts"


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

"""The lending page's wiring (Round 86).

`npm run check:lending` holds the market to closed forms and `npm run check:lending-private` holds the private check to the clear answer. Neither can see how the page is
assembled, so this holds:

- the flow runs no QKD and no model, and the canvas and the stage-order graph draw exactly that flow;
- a page with no quantum channel offers no control that needs one, and signs "pqc-only";
- the page states the condition its privacy depends on: with public thresholds the check protects nothing;
- the close factor's boundary is the one in Aave's contract;
- it is offered only by the build that can run it.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web" / "src"
FLOW = ["pqc_handshake", "lending_market", "private_liquidation", "result", "sign_result"]


def _read(rel: str) -> str:
    return (WEB / rel).read_text()


def test_the_flow_has_no_quantum_stage_and_no_model_stage() -> None:
    m = re.search(r"defi_lending: \[([^\]]*)\]", _read("lib/clientCompute.ts"))
    assert m, "FLOWS has no defi_lending entry"
    assert re.findall(r'"(\w+)"', m.group(1)) == FLOW


def test_the_canvas_and_the_graph_draw_that_flow() -> None:
    specs = _read("lib/useCaseFlows.ts")
    spec = specs[specs.index("defi_lending: {") : specs.index("finance_aml: {")]
    drawn = list(dict.fromkeys(re.findall(r'stageId: "(\w+)"', spec)))
    assert drawn == [s for s in FLOW if s != "result"], f"the canvas draws {drawn}"
    assert '"q-arrow"' not in spec, "the lending canvas has no quantum lane to draw"
    graph = _read("components/OrchestrationGraph.tsx")
    nodes = graph[graph.index("defi_lending: [\n") :]
    assert re.findall(r'id: "(\w+)"', nodes[: nodes.index("],")]) == FLOW
    assert "defi_lending: []," in graph, "the graph would draw a full-deployment stage this flow does not have"


def test_a_page_with_no_quantum_channel_offers_no_control_that_needs_one() -> None:
    controls = _read("components/PipelineControls.tsx")
    assert 'quantum && sel(demo ? "Quantum channel"' in controls
    assert re.search(r"\{quantum && <label[^\n]*sel-optiqkd", controls)
    hook = _read("lib/usePipelineRun.ts")
    assert 'runsAnyQkd(usecase) ? qkd : "pqc-only"' in hook, "the run and the record must not name a quantum channel this flow does not have"
    panel = _read("components/ParameterPanel.tsx")
    block = panel[panel.index('if (usecase === "defi_lending") {') :]
    assert "qkd_" not in block[: block.index("return (")], "the lending parameters include a QKD field"


def test_the_page_states_when_the_check_protects_nothing() -> None:
    assert "On a public chain they are public, and then Bob can compute his margin himself" in _read("components/LendingPanels.tsx")
    assert "public, and then Bob can compute M himself" in _read("lib/lending/privateCheck.ts")
    assert "Carol, a separate key holder" in _read("lib/useCases.ts")
    assert "the pool's thresholds kept from him" in _read("lib/useCases.ts")


def test_both_arrangements_say_what_they_depend_on() -> None:
    panels = _read("components/LendingPanels.tsx")
    assert "Two parties must not collude" in panels, "Carol's custody has to state its non-collusion assumption"
    assert "no answer counts as a default" in panels, "the bilateral mode has to say Bob can refuse"
    assert "no primary source for a confidential liquidation running in production" in panels
    assert re.search(r"held === \"borrower\" \? BILATERAL_LIMITS : KEYHOLDER_LIMITS", panels), "the limits have to follow the run's own arrangement"


def test_carol_is_the_default_and_the_control_reaches_the_run() -> None:
    defaults = (REPO / "src" / "qsh" / "config" / "defaults.py").read_text()
    assert 'custody: str = "keyholder"' in defaults
    panel = _read("components/ParameterPanel.tsx")
    assert 'id="pp-custody"' in panel and 'value="keyholder"' in panel and 'value="borrower"' in panel
    model = _read("lib/lending/model.ts")
    assert "p.lending_custody" in model, "the control has to reach the simulation's configuration"
    assert "parties: (c) => bilateral(c)" in _read("lib/useCaseFlows.ts"), "the canvas has to name who holds the key in this run"


def test_the_close_factor_boundary_is_the_contracts() -> None:
    """`healthFactor > CLOSE_FACTOR_HF_THRESHOLD ? DEFAULT : MAX` in Aave's LiquidationLogic: 0.95 exactly is the full close."""
    assert "hf <= c.full_close_at_or_below_hf ? 1 : c.close_factor" in _read("lib/lending/model.ts")


def test_it_is_offered_only_by_the_build_that_can_run_it() -> None:
    cases = _read("lib/useCases.ts")
    entry = cases[cases.index('id: "defi_lending"') :]
    assert "clientOnly: true" in entry
    assert "ALL_USECASES.filter((u) => !u.clientOnly || isClientDemo())" in cases
    assert "defi_lending" not in (REPO / "api" / "run_manager.py").read_text(), "the server has no such pipeline"

"""The lending page's wiring (Round 86).

`npm run check:lending` holds the market to closed forms and `npm run check:lending-private` holds the private check to the clear answer. Neither can see how the page is
assembled, so this holds:

- the flow runs no QKD and no model, and the canvas and the stage-order graph draw exactly that flow;
- a page with no quantum channel offers no control that needs one, and signs "pqc-only";
- the page states the condition its privacy depends on: with public thresholds the check protects nothing;
- the close factor's boundary is the one in Aave's contract;
- it is offered only by the build that can run it;
- the committee's arrangement states what it rests on, shares once per ciphertext, and its circuit is pinned in both places that run it.
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
    assert re.search(r"held === \"borrower\" \? BILATERAL_LIMITS : held === \"committee\" \? COMMITTEE_LIMITS : KEYHOLDER_LIMITS", panels), \
        "the limits have to follow the run's own arrangement"
    assert "Who must not collude" in panels and "The pool and any two members together could decrypt" in panels, "the committee has to state its non-collusion assumption"
    assert "Passive members, and no robustness" in panels, "the committee has to say its members are assumed to follow the protocol"


def test_carol_is_the_default_and_the_control_reaches_the_run() -> None:
    defaults = (REPO / "src" / "qsh" / "config" / "defaults.py").read_text()
    assert 'custody: str = "keyholder"' in defaults
    panel = _read("components/ParameterPanel.tsx")
    assert 'id="pp-custody"' in panel and 'value="keyholder"' in panel and 'value="borrower"' in panel and 'value="committee"' in panel
    model = _read("lib/lending/model.ts")
    assert "p.lending_custody" in model, "the control has to reach the simulation's configuration"
    assert 'p.lending_custody === "committee"' in model, "the committee's value has to reach the run too"
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


def test_the_committee_shares_once_per_ciphertext() -> None:
    """Lattigo's SECURITY.md: a member must not share twice for the same input, whatever the active set (Okada et al., ePrint 2025/409)."""
    go = (REPO / "wasm" / "qsh-fedthresh" / "agg" / "committee.go").read_text()
    assert "c.used[digest]" in go and "sha256.Sum256(c1)" in go, "the members have to refuse a c1 they have shared for"
    assert "a fresh encryption of zero" in go, "the pool has to re-randomise every evaluated ciphertext, or two hours at the same prices give the same c1"
    assert "addUniformFlood(" in go and "leastFloodBits(CommitteeBits" in go, "each share has to be flooded, sized for the committee's own worst case"
    tests = (REPO / "wasm" / "qsh-fedthresh" / "agg" / "committee_test.go").read_text()
    assert "TestCommitteeRefusesASecondShareForTheSameCiphertext" in tests


def test_the_sign_circuit_is_pinned_where_it_runs() -> None:
    rust = (REPO / "wasm" / "qsh-mpc" / "src" / "lib.rs").read_text()
    harness = (REPO / "web" / "scripts" / "check_lending_private.mjs").read_text()
    pinned = re.search(r"assert_eq!\(o\.and_ops, (\d+),", rust)
    assert pinned, "the crate's test has to pin the sign circuit's size"
    assert f"o.andGates === {pinned.group(1)}" in harness, "the page's check and the crate's test have to pin the same circuit"

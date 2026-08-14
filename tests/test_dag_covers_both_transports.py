"""The DAG is drawn once and describes two different runs; it must cover both.

WHY THIS EXISTS

`OrchestrationGraph.tsx` renders on `/orchestration` in BOTH modes - the client-only public demo and
the full deployment behind the API. Its `FLOWS` map was reconciled with `clientFlow()` in an earlier
round and is exactly right for the demo. Against the backend it was wrong for every use case:

    finance_aml       backend runs `explainability`; graph draws `federated_training`, which
                      `standard_stages()` does not contain at all
    genomics, demo    backend runs `explainability` - no node
    medical_fl        backend runs `explainability` - no node
    telecom_ids       backend runs `qkd_optimize` - no node
    satellite_comms   backend runs `qkd_optimize` and `telemetry_anomaly` - no nodes

So a full-mode visitor watched the graph go dark for a stage that was running, while a node was drawn
for a stage that could never light. The previously-fixed "diagram shows a stage the pipeline does not
run" defect was live the whole time on the transport nobody looked at.

This is the round's subject in one file: a check that names a boundary and verifies one side. The
earlier reconciliation compared `FLOWS` against `clientFlow()` and stopped there.

WHY THE STAGE NAMES ARE READ FROM THE CLASSES

`stage.name` is a class attribute, so a rename that misses this map would otherwise be invisible
here too. The map from class to name is parsed from the source rather than hand-written, which is
why `telemetry_anomaly` - defined inside the satellite use case, not in `stages.py` - is picked up
without this file knowing where it lives.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TSX = REPO / "web" / "src" / "components" / "OrchestrationGraph.tsx"

#: Files that define a `Stage` subclass, so `ClassName -> stage.name` can be resolved.
_STAGE_SOURCES = [
    REPO / "src" / "qsh" / "orchestration" / "stages.py",
    REPO / "src" / "qsh" / "usecases" / "satellite_comms" / "__init__.py",
    REPO / "src" / "qsh" / "usecases" / "telecom_ids" / "__init__.py",
    REPO / "src" / "qsh" / "usecases" / "medical_fl" / "__init__.py",
]

#: Where each use case's backend stage list is built, and the line the list starts on.
_BACKEND_LISTS = {
    "finance_aml": (REPO / "src" / "qsh" / "orchestration" / "pipeline.py", "def standard_stages"),
    "genomics": (REPO / "src" / "qsh" / "orchestration" / "pipeline.py", "def standard_stages"),
    "demo": (REPO / "src" / "qsh" / "orchestration" / "pipeline.py", "def standard_stages"),
    "medical_fl": (REPO / "src" / "qsh" / "usecases" / "medical_fl" / "__init__.py", "stages = ["),
    "telecom_ids": (REPO / "src" / "qsh" / "usecases" / "telecom_ids" / "__init__.py", "stages = ["),
    "satellite_comms": (REPO / "src" / "qsh" / "usecases" / "satellite_comms" / "__init__.py", "stages = ["),
}


def _class_to_stage_name() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in _STAGE_SOURCES:
        out.update(re.findall(r"class (\w+)\(Stage\):.*?name = \"([a-z_]+)\"", path.read_text(), re.S))
    assert len(out) >= 8, f"only {len(out)} Stage classes resolved; the class/name convention changed"
    return out


def _backend_stages(usecase: str) -> list[str]:
    path, anchor = _BACKEND_LISTS[usecase]
    text = path.read_text()
    at = text.index(anchor)
    window = text[at: at + 1400]
    names = _class_to_stage_name()
    out: list[str] = []
    for cls in re.findall(r"(\w+Stage)\(\)", window):
        resolved = names.get(cls)
        assert resolved, f"{cls} has no resolvable stage.name; the parse is incomplete, not the code"
        if resolved not in out:
            out.append(resolved)
    return out


def _graph_flow(usecase: str, which: str) -> list[str]:
    """`which` is the const name: FLOWS (client) or BACKEND_EXTRA (backend-only stages)."""
    src = TSX.read_text()
    block = re.search(rf"const {which}[^=]*=\s*\{{(.*?)\n\}};", src, re.S)
    assert block, f"{which} not found in OrchestrationGraph.tsx"
    body = block.group(1)
    entry = re.search(rf"\b{usecase}:\s*\[(.*?)\]", body, re.S)
    if entry is None and which == "FLOWS":
        entry = re.search(r"\bdefault:\s*\[(.*?)\]", body, re.S)
    if entry is None:
        return []
    return re.findall(r'id:\s*"([a-z_]+)"', entry.group(1)) or re.findall(r'"([a-z_]+)"', entry.group(1))


def test_the_parse_finds_stages_on_both_sides() -> None:
    """Guard against a parse that silently yields nothing to compare."""
    assert len(_backend_stages("finance_aml")) >= 6
    assert len(_graph_flow("finance_aml", "FLOWS")) >= 6


def test_every_backend_stage_has_a_node() -> None:
    """A stage that runs and lights nothing makes the graph read as a stalled run."""
    missing = {}
    for usecase in _BACKEND_LISTS:
        drawn = set(_graph_flow(usecase, "FLOWS")) | set(_graph_flow(usecase, "BACKEND_EXTRA"))
        gap = [s for s in _backend_stages(usecase) if s not in drawn]
        if gap:
            missing[usecase] = gap
    assert not missing, (
        "the backend runs stages the DAG has no node for:\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in missing.items())
        + "\n\nOn the full deployment the graph goes dark while these run. Add a node, or if the "
        "stage is deliberately not drawn, say so in the file and add it to this test's exceptions "
        "with the reason - silence here reads as coverage."
    )


def _drawn_in_order(usecase: str) -> list[str]:
    """`flowFor()`'s result, reproduced from the two maps and the `before` field."""
    client = _graph_flow(usecase, "FLOWS")
    extra_src = TSX.read_text()
    block = re.search(r"const BACKEND_EXTRA[^=]*=\s*\{(.*?)\n\};", extra_src, re.S)
    assert block, "BACKEND_EXTRA not found"
    entry = re.search(rf"\b{usecase}:\s*\[(.*?)\]", block.group(1), re.S) \
        or re.search(r"\bdefault:\s*\[(.*?)\]", block.group(1), re.S)
    pairs = re.findall(r'id:\s*"([a-z_]+)"[^}]*?before:\s*"([a-z_]+)"', entry.group(1)) if entry else []
    out = list(client)
    for node, before in pairs:
        at = out.index(before) if before in out else max(0, len(out) - 1)
        out.insert(at, node)
    return out


def test_the_drawn_order_is_the_order_the_backend_runs() -> None:
    """A set comparison cannot see a diagram that runs backwards.

    The first version of this file compared SETS, and every backend-only node was spliced in front
    of `result` regardless of where it actually runs. Both were wrong in the same direction, so
    neither could reveal the other: all six use cases drew `explainability` after
    `quantum_comparison` (it runs before) or `qkd_optimize` near the end (it runs third). On the full
    deployment the visitor watched a later node light and the run then jump backwards - which reads
    as a stalled pipeline, the exact failure the sibling test's docstring names.

    An ordered artefact needs an ordered check. Restricting to the stages the backend actually runs
    keeps client-only nodes (`data`, and finance's injected `federated_training`) out of the
    comparison rather than forcing them into a backend list that has no place for them.
    """
    wrong = []
    for usecase in _BACKEND_LISTS:
        ran = _backend_stages(usecase)
        drawn = [s for s in _drawn_in_order(usecase) if s in ran]
        if drawn != ran:
            wrong.append(f"{usecase}:\n      drawn: {' -> '.join(drawn)}\n      ran  : {' -> '.join(ran)}")
    assert not wrong, (
        "the DAG draws stages in an order the backend does not run them:\n    "
        + "\n    ".join(wrong)
        + "\n\nEach BACKEND_EXTRA entry carries a `before` naming the node it precedes; a node with "
        "the wrong `before` lights out of sequence and the run appears to go backwards."
    )


def test_no_node_is_drawn_for_a_stage_neither_transport_runs() -> None:
    """The reverse direction: a node that can never light.

    `federated_training` was drawn for finance because the CLIENT injects it, while
    `standard_stages()` never has it - so the same node is correct on one transport and impossible
    on the other. That is what `BACKEND_EXTRA` and the client-only note exist to disambiguate; this
    asserts nothing is drawn that neither side runs.
    """
    client_src = (REPO / "web" / "src" / "lib" / "clientCompute.ts").read_text()
    stray = {}
    for usecase in _BACKEND_LISTS:
        backend = set(_backend_stages(usecase))
        for stage in _graph_flow(usecase, "FLOWS") + _graph_flow(usecase, "BACKEND_EXTRA"):
            if stage in backend or stage == "data":
                continue
            # Client-only stages must at least exist in the client pipeline's vocabulary.
            if f'"{stage}"' not in client_src:
                stray.setdefault(usecase, []).append(stage)
    assert not stray, (
        f"nodes drawn for stages neither transport runs: {stray}. A node that can never light is a "
        "diagram asserting a step that does not happen."
    )

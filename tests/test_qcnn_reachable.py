"""A published QCNN must be REACHABLE from the stages its use case actually runs.

`tests/test_qcnn_usecases.py` checks the data: the declared set matches the files on disk, the
weights load, and the shipped copy agrees with the trained artifact. All three passed while the
telecom QCNN option was inert, because none of them asks the question that matters to a visitor -
does selecting it change anything?

The gap is a half-stated invariant. The comment above `QCNN_USECASES` says a backend option is shown
"when the artefact implementing it exists", and that is one of two conditions. The other is that a
stage the use case RUNS reads the artefact. Telecom satisfied the first and not the second:
`FLOWS.telecom_ids` contains `qml_ids` and not `private_inference`, and the QCNN branch lived in the
`private_inference` case. So `ParameterPanel` offered a QCNN option that no code path could honour,
which is the same shape as the genomics failure the comment itself recounts - option offered, silent
fallback to CKKS, two panels announcing that a quantum circuit had decided the class.

Nothing local caught it. tsc was clean (the branch simply was not there to fail), every Python gate
was green, and the weights were served with HTTP 200 and the right content-type. It took running the
deployed page and noticing that `qcnn_weights.json` was never fetched.

This test asserts the second condition, in the same source-derived style as the rest of the suite:
no hand-kept list, both facts parsed from the modules that define them.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

_WEB = Path(__file__).resolve().parents[1] / "web" / "src" / "lib"
_COMPUTE = _WEB / "clientCompute.ts"
_PIPELINE = _WEB / "usePipelineRun.ts"

#: The call every QCNN branch must make. Fetching the weights is the irreducible step - a branch
#: that does not load them cannot be running a trained circuit whatever else it prints.
_LOADER = "loadQcnnWeights("


def _flows() -> dict[str, list[str]]:
    """`FLOWS` from clientCompute.ts, as a dict."""
    src = _COMPUTE.read_text(encoding="utf-8")
    m = re.search(r"const FLOWS: Record<string, string\[\]> = \{(.*?)\n\};", src, re.S)
    assert m, "FLOWS table not found - clientCompute.ts changed shape and this parser is now blind"
    out: dict[str, list[str]] = {}
    for key, body in re.findall(r"(\w+):\s*\[([^\]]*)\]", m.group(1)):
        out[key] = [s.strip().strip('"') for s in body.split(",") if s.strip()]
    assert len(out) >= 5, f"parsed only {len(out)} flows; the table has more than that"
    return out


def _qcnn_usecases() -> set[str]:
    """`QCNN_USECASES` from usePipelineRun.ts."""
    src = _PIPELINE.read_text(encoding="utf-8")
    m = re.search(r"QCNN_USECASES = new Set\(\[([^\]]*)\]\)", src)
    assert m, "QCNN_USECASES not found - usePipelineRun.ts changed shape"
    got = {s.strip().strip('"') for s in m.group(1).split(",") if s.strip()}
    assert got, "QCNN_USECASES parsed empty, which would make every assertion below vacuous"
    return got


def _stage_blocks() -> dict[str, str]:
    """Each `case "<stage>": { ... }` body from the runStage switch, keyed by stage name.

    Split on the case labels rather than by matching braces: the bodies contain template literals,
    regexes and brace characters inside strings, and a brace counter gets those wrong in exactly the
    quiet way this test exists to prevent.
    """
    src = _COMPUTE.read_text(encoding="utf-8")
    starts = [(m.group(1), m.start()) for m in re.finditer(r'^      case "([a-z_]+)":', src, re.M)]
    assert len(starts) >= 6, f"found {len(starts)} case labels; the switch has more than that"
    bounds = [p for _, p in starts] + [len(src)]
    return {name: src[bounds[i]:bounds[i + 1]] for i, (name, _) in enumerate(starts)}


def _stages_that_run_a_qcnn() -> set[str]:
    return {name for name, body in _stage_blocks().items() if _LOADER in body}


@pytest.mark.parametrize("usecase", sorted(_qcnn_usecases()))
def test_a_published_qcnn_is_reachable_from_a_stage_the_use_case_runs(usecase: str) -> None:
    """Shipping weights and registering the use case is not enough; a stage must read them."""
    flows, qcnn_stages = _flows(), _stages_that_run_a_qcnn()
    assert usecase in flows, f"{usecase} declares a QCNN but has no flow"
    reachable = sorted(set(flows[usecase]) & qcnn_stages)
    assert reachable, (
        f"{usecase} is in QCNN_USECASES, so ParameterPanel offers a QCNN backend option, but none of "
        f"the stages it runs ({', '.join(flows[usecase])}) calls {_LOADER!r}. Stages that do: "
        f"{', '.join(sorted(qcnn_stages)) or '(none)'}. The option would be inert: selecting it "
        f"changes nothing and the page falls back to CKKS while claiming a circuit decided the class."
    )


def test_the_check_can_fail() -> None:
    """Anti-tautology: a use case with no QCNN stage must come out unreachable.

    Without this, the test above passing tells you only that the parsers returned something.
    `satellite_comms` runs neither `private_inference` nor `qml_ids`, and ships no weights, so it is
    the honest negative control. If it ever gains a QCNN the assertion below should be moved, not
    deleted - the suite needs some use case that demonstrates the check discriminates.
    """
    flows, qcnn_stages = _flows(), _stages_that_run_a_qcnn()
    assert qcnn_stages, "no stage calls the loader at all; every reachability assertion is vacuous"
    negative = "satellite_comms"
    assert negative in flows and negative not in _qcnn_usecases()
    assert not (set(flows[negative]) & qcnn_stages), (
        f"{negative} now reaches a QCNN stage, so it no longer demonstrates that this check can "
        f"return false; pick another use case that ships no weights")


def test_telecom_specifically_runs_qml_ids_and_not_private_inference() -> None:
    """The exact configuration that produced the inert control, pinned so it cannot come back.

    Telecom is the only QCNN use case whose flow omits `private_inference`, which is why it was the
    one that broke. Pinning it means a future edit that moves the telecom QCNN branch back into
    `private_inference` fails here rather than on the deployed site.
    """
    flow = _flows()["telecom_ids"]
    assert "qml_ids" in flow and "private_inference" not in flow, (
        f"telecom's flow changed to {flow}; the reachability requirement above still holds but the "
        f"reasoning in this file's docstring needs rewriting")
    assert _LOADER in _stage_blocks()["qml_ids"]


def test_every_use_case_with_weights_on_disk_is_declared() -> None:
    """The reverse direction, cheap to state here: weights on disk with no registration are dead.

    `test_qcnn_usecases.py` owns the forward direction. This catches the mirror-image mistake -
    training and committing a circuit, then never adding the use case, so the artefact ships in the
    bundle and nothing can ever select it.
    """
    root = Path(__file__).resolve().parents[1] / "web" / "public" / "data"
    on_disk = {p.parent.name for p in root.glob("*/qcnn_weights.json")}
    assert on_disk, f"no qcnn_weights.json under {root}; this test is not looking where they live"
    declared = {u.split("_")[0] if u != "medical_fl" else "medical" for u in _qcnn_usecases()}
    orphans = sorted(on_disk - declared)
    assert not orphans, (
        f"qcnn_weights.json ships for {orphans} but no use case declares it, so the bundle carries "
        f"weights that no control can select")


def test_the_shipped_weights_name_their_readout_convention() -> None:
    """A multiclass readout that does not record its bit order cannot be reproduced from the file.

    Reading the bits in another order permutes the class map while leaving accuracy, macro F1 and
    every loss curve numerically identical, so the convention has to travel WITH the weights rather
    than living only in the trainer.
    """
    root = Path(__file__).resolve().parents[1] / "web" / "public" / "data"
    multiclass = []
    for p in sorted(root.glob("*/qcnn_weights.json")):
        w = json.loads(p.read_text(encoding="utf-8"))
        readout = w.get("readout") or {}
        if int(readout.get("readout_qubits") or 0) > 0:
            multiclass.append(p)
            assert readout.get("bit_order"), f"{p} measures multiple qubits but names no bit order"
            assert readout.get("class_index"), f"{p} names no class-index rule"
            n = int(readout.get("n_classes") or 0)
            assert len(w.get("class_names") or []) == n, (
                f"{p} declares {n} classes but lists {len(w.get('class_names') or [])} names")
    assert multiclass, "no multiclass QCNN found; this test is asserting nothing"

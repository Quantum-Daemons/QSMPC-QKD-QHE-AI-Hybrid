"""The QCNN trainer writes its artifact's keys in the order the shipped artifacts hold them (Round 86).

WHY THIS EXISTS. `scripts/train/qcnn_classical_surrogate.py` added `classical_surrogate` to the four shipped QCNN artifacts
without retraining them, last before `note`, while the trainer wrote the same key after `classical_lr_accuracy`. The values
agreed and the bytes did not, so the command each artifact records no longer regenerated it byte for byte. Only
`test_r84_qcnnfin_four_columns.py`'s slow test could see that, and CI runs `-m "not slow"`.

WHAT IS ASSERTED. The top-level keys of the dict the trainer writes, read from its source, are each shipped artifact's keys
in the same order. A key added to one and not the other, or moved in one, fails here without a training run.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "scripts" / "train" / "qcnn_variational.py"
ARTIFACTS = ("models/qcnn_medical/weights.json", "models/qcnn_finance/weights.json",
             "models/qcnn_telecom/weights.json", "models/qcnn_telecom_q7/weights.json")


def _written_keys() -> list[str]:
    """The keys, in order, of the one dict literal in the trainer that holds `classical_surrogate`: the artifact it writes."""
    dicts = [node for node in ast.walk(ast.parse(TRAINER.read_text(encoding="utf-8")))
             if isinstance(node, ast.Dict)
             and any(isinstance(k, ast.Constant) and k.value == "classical_surrogate" for k in node.keys)]
    assert len(dicts) == 1, f"expected one artifact dict in the trainer, found {len(dicts)}"
    keys = dicts[0].keys
    assert all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in keys), "a computed key cannot be checked here"
    return [k.value for k in keys]


EXPORT = REPO / "scripts" / "export_finance_qcnn_amlworld.py"
#: Written by scripts/export_finance_qcnn_amlworld.py from a trained artifact since Round 86's batch 6, not by the trainer.
EXPORTED = {"models/qcnn_finance/weights.json"}


def _keys_assigned(path: Path, target: str) -> list[str]:
    """Constant keys of `target["k"] = ...` assignments in a script, in source order."""
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == target
                        and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str)):
                    out.append((node.lineno, t.slice.value))
    return [k for _, k in sorted(out)]


def _exported_keys() -> list[str]:
    """The finance artifact's keys as its two writers produce them: the trainer's dict, the keys its `--select-on ap` run
    adds, then the export's `served.update({...})` keys it does not already hold and the ones it assigns after."""
    keys = _written_keys() + [k for k in _keys_assigned(TRAINER, "artifact") if k in ("test_auprc", "classical_lr_auprc", "selection")]
    upd = [n for n in ast.walk(ast.parse(EXPORT.read_text(encoding="utf-8")))
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "update"
           and isinstance(n.func.value, ast.Name) and n.func.value.id == "served"]
    assert len(upd) == 1 and isinstance(upd[0].args[0], ast.Dict), "expected one served.update({...}) in the export"
    keys += [k.value for k in upd[0].args[0].keys if k.value not in keys]
    return keys + [k for k in _keys_assigned(EXPORT, "served") if k not in keys]


@pytest.mark.parametrize("path", ARTIFACTS)
def test_the_trainer_writes_the_keys_the_artifact_holds_in_its_order(path: str) -> None:
    shipped = list(json.loads((REPO / path).read_text(encoding="utf-8")))
    if path in EXPORTED:
        assert shipped == _exported_keys(), (
            f"{path} and its writers, the trainer and the export, order its keys differently")
        return
    assert shipped == _written_keys(), (
        f"{path} and the trainer order their keys differently, so the recorded command cannot regenerate the file")

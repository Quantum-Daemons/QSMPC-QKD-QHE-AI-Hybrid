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


@pytest.mark.parametrize("path", ARTIFACTS)
def test_the_trainer_writes_the_keys_the_artifact_holds_in_its_order(path: str) -> None:
    shipped = list(json.loads((REPO / path).read_text(encoding="utf-8")))
    assert shipped == _written_keys(), (
        f"{path} and the trainer order their keys differently, so the recorded command cannot regenerate the file")

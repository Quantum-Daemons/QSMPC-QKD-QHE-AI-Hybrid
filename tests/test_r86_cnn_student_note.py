"""The medical CNN student's published note says what the browser does with it (Round 86, fact-check C3).

Its weights.json note ended "The browser runs conv and both dense layers under CKKS and the ReLU/argmax under MPC, the
hybrid this project describes." The page and the model card say otherwise. Each layer's output is decrypted, and the
ReLU, the average pool and the argmax run in plaintext in the client; the page's MPC argmax runs on the Wisconsin
regression. The note also still used the "FHE-feasible" label that Round 86 retired.

The note is now a constant in qsh.models.distil, and the shipped file is held to it, so a retrain writes the same text.
"""

from __future__ import annotations

import json
from pathlib import Path

from qsh.models import distil

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "models" / "medical_breastmnist_student" / "weights.json"
CKKS = REPO / "web" / "src" / "lib" / "clientFHE.ts"


def test_the_shipped_note_is_the_generators() -> None:
    assert json.loads(WEIGHTS.read_text())["note"] == distil.CNN_STUDENT_NOTE


def test_the_note_says_what_the_browser_does() -> None:
    note = distil.CNN_STUDENT_NOTE
    assert "ReLU/argmax under MPC" not in note and "FHE-feasible" not in note
    assert "the ReLU, the average pool and the argmax run in plaintext in the client" in note
    # The code it describes: the decryption stays, and ReLU and pooling are applied in the clear.
    src = CKKS.read_text()
    assert "ReLU and pooling are applied in the clear here" in src

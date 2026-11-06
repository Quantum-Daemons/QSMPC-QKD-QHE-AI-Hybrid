"""The telecom quantum comparison is made like for like, and every surface states it that way (Round 86).

The record set the quantum models' macro F1 against the forest's, calling it "the same 8-class problem". Both quantum
models are binary, each on its own smaller draw, so their macro F1 averages two classes and the forest's eight.
`scripts/measure_telecom_quantum_like_for_like.py` scores a forest on each quantum model's own split and labels, after
re-fitting the kernel SVC to its recorded figures and confirming that the QNN's are the constant predictor's. This
holds the record, the trainer, the model panel and the card's note to that measurement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _latest() -> dict:
    found = sorted((REPO / "docs" / "measurements").glob("telecom-quantum-like-for-like-*.json"))
    assert found, "no telecom-quantum-like-for-like measurement is committed"
    return json.loads(found[-1].read_text())


def _record() -> dict:
    return json.loads((REPO / "models" / "metrics.json").read_text())["telecom_ids.qml"]


def test_the_record_carries_the_measured_comparison() -> None:
    t, rec = _latest(), _record()
    lfl = rec["quantum_comparison_like_for_like"]
    for side, split in (("qnn", t["qnn_split"]), ("qsvc", t["qsvc_split"])):
        assert lfl[side]["quantum_macro_f1"] == split["quantum"]["macro_f1"]
        assert lfl[side]["forest_macro_f1"] == split["forest_trained_binary"]["macro_f1"]
        assert lfl[side]["constant_attack_macro_f1"] == split["constant_attack"]["macro_f1"]
        assert lfl[side]["test_rows"] == split["test_rows"]
    # The verdict is the like-for-like one, and it is what the record stores.
    derived = any(lfl[s]["quantum_macro_f1"] > lfl[s]["forest_macro_f1"] for s in ("qnn", "qsvc"))
    assert rec["quantum_beats_classical"] is derived is t["quantum_beats_classical"]


def test_the_qnn_is_the_constant_predictor_and_the_record_says_binary() -> None:
    t, rec = _latest(), _record()
    q = t["qnn_split"]
    assert q["quantum"]["is_the_constant_attack_predictor"] is True
    assert abs(rec["quantum_qnn_macro_f1"] - q["constant_attack"]["macro_f1"]) < 5e-6
    # 7 of every 8 test rows are attacks: the constant predictor's three figures are 7/8, 14/15 and 7/15.
    assert q["test_attack_rows"] * 1 == q["test_normal_rows"] * 7
    assert abs(q["constant_attack"]["accuracy"] - 7 / 8) < 1e-6
    assert abs(q["constant_attack"]["macro_f1"] - 7 / 15) < 1e-6


def test_the_trainer_compares_like_with_like() -> None:
    src = (REPO / "scripts" / "train" / "telecom_ids.py").read_text()
    assert "classical_on_quantum_split" in src and '"quantum_comparison_like_for_like"' in src
    assert "on the same 8-class problem." not in src
    assert 'beat = max(qnn.macro_f1, qsvc.macro_f1) > rf_metrics["macro_f1"]' not in src


def test_the_panel_derives_the_verdict_from_the_like_for_like_splits() -> None:
    panel = (REPO / "web" / "src" / "components" / "ModelPanel.tsx").read_text()
    assert "rec.quantum_comparison_like_for_like" in panel
    assert "the best quantum macro F1 here" not in panel


def test_the_card_note_states_the_measured_forest() -> None:
    sys.path.insert(0, str(REPO / "src"))
    from qsh.models import registry

    t, note = _latest(), registry.get("telecom_ids.qml").notes
    forest_svc = t["qsvc_split"]["forest_trained_binary"]["macro_f1"]
    forest_qnn = t["qnn_split"]["forest_trained_binary"]["macro_f1"]
    assert f"scores {forest_svc:.1f} and {forest_qnn:.3f}" in note, note
    assert f"telecom-quantum-like-for-like-{t['measured']}.json" in note
    assert "on identical data" not in note

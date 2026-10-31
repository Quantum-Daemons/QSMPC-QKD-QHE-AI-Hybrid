"""The medical diagnosis means the same class whichever private-inference backend produced it.

WHY THIS EXISTS (Round 84). The QCNN trainer labels malignant 1 (y = target == 0) while the medical
use case keeps scikit-learn's WDBC convention, malignant 0, which the CKKS branch reports. The stage
wrote the QCNN's raw index to `private_prediction`, so `private_diagnosis` read 0 for a malignant
patient on CKKS and 1 for the same verdict on QCNN. The fix maps the artifact's index into the use
case's convention; the raw index stays under `qhe_prediction`.
"""

from __future__ import annotations

import pytest

from qsh.qhe import qcnn_trained


@pytest.mark.parametrize("artifact_cls, expected_case_cls", [(1, 0), (0, 1)])
def test_a_qcnn_verdict_is_reported_in_the_use_cases_convention(monkeypatch, artifact_cls, expected_case_cls) -> None:
    if qcnn_trained.load_weights("medical_fl") is None:
        pytest.skip("no trained medical QCNN artifact")
    monkeypatch.setattr(qcnn_trained, "infer", lambda x, w: (0.0, artifact_cls, 0.0))
    from qsh.usecases import medical_fl

    out = medical_fl.run_demo({"inference": "qcnn", "seed": 1})
    assert out["private_diagnosis_backend"] == "qcnn-trained"
    assert out["private_diagnosis"] == expected_case_cls, (
        f"the QCNN's class {artifact_cls} (1 = malignant in the artifact) must be reported as "
        f"{expected_case_cls} in WDBC's convention (0 = malignant), the one the CKKS branch uses")

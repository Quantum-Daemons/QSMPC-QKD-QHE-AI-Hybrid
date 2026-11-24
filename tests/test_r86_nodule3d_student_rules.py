"""The 3D nodule student's script applies its pre-registration's rules (Round 86, batch 4).

`references/medical-nodule3d-square-student-preregistration.md` fixes the label, the split and the network before any
label was counted. `scripts/train/medical_nodule3d_student.py` is where they run, so this checks them there, on
records written here, before the script meets LIDC-IDRI's:

- the script refuses to run unless the pre-registration's SHA-256 is the one it records, and it is the file's;
- the label: the median rating above 3 malignant, below 3 benign, exactly 3 dropped, with medians of an even number of
  ratings falling between two;
- the split: by patient, stratified on the patient's majority label with a tie counted malignant, 0.7n and 0.1n
  rounded half up per stratum, the same on every call, no patient in two parts;
- the folds: five, stratified the same way, covering every training patient once;
- the folded float64 network the CKKS pass is compared with gives the PyTorch module's logits in evaluation mode.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "train" / "medical_nodule3d_student.py"


def _script():
    spec = importlib.util.spec_from_file_location("medical_nodule3d_student", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_it_runs_only_under_the_committed_preregistration() -> None:
    s = _script()
    assert s.sha256(s.PREREG) == s.PREREG_SHA256
    s.check_preregistration()


@pytest.mark.parametrize("ratings, label", [
    ([5, 5, 5, 4], 1), ([1, 2, 2], 0), ([3], None), ([3, 3, 4, 2], None),
    ([2, 3, 4, 4], 1),          # median 3.5
    ([2, 2, 3, 4], 0),          # median 2.5
    ([4], 1), ([1, 5], None),   # median 3: dropped, whatever the spread
])
def test_the_label_is_the_median_rating_with_3_dropped(ratings, label) -> None:
    assert _script().label_of(ratings) == label


def _records(n_patients: int = 60, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_patients):
        for _ in range(int(rng.integers(1, 4))):
            out.append({"patient": f"LIDC-IDRI-{i:04d}", "malignancy": rng.integers(1, 6, 4).tolist()})
    return out


def test_the_split_is_by_patient_stratified_and_the_same_every_time() -> None:
    s = _script()
    records = _records()
    parts = s.make_split(records)
    assert parts == s.make_split(records)
    every = [p for k in parts for p in parts[k]]
    assert len(every) == len(set(every)), "a patient is in two parts"
    labels: dict[str, list[int]] = {}
    for r in records:
        y = s.label_of(r["malignancy"])
        if y is not None:
            labels.setdefault(r["patient"], []).append(y)
    assert set(every) == set(labels), "the split is over the patients with a labelled nodule, all of them"
    for stratum in (0, 1):
        members = {p for p, ys in labels.items() if int(2 * sum(ys) >= len(ys)) == stratum}
        n = len(members)
        got = [len(members & set(parts[k])) for k in ("train", "val", "test")]
        assert got == [(7 * n + 5) // 10, (n + 5) // 10, n - (7 * n + 5) // 10 - (n + 5) // 10], (stratum, got)
    # A tie is malignant: a patient with one benign and one malignant nodule is in the malignant stratum.
    tie = [{"patient": "T", "malignancy": [1, 1, 1]}, {"patient": "T", "malignancy": [5, 5, 5]}]
    only = s.make_split(tie)
    assert only == {"train": ["T"], "val": [], "test": []}


def test_the_folds_cover_every_training_patient_once_and_are_stratified() -> None:
    s = _script()
    patients = [f"P{i:03d}" for i in range(50) for _ in range(2)]
    y = np.array([int(i % 5 == 0) for i in range(50) for _ in range(2)])
    fold = s.folds_of(patients, y)
    assert set(fold) == set(patients) and set(fold.values()) == set(range(s.FOLDS))
    sizes = np.bincount(list(fold.values()), minlength=s.FOLDS)
    assert sizes.max() - sizes.min() <= 1, sizes
    malignant = {p for p, v in zip(patients, y) if v}
    per_fold = np.bincount([fold[p] for p in malignant], minlength=s.FOLDS)
    assert per_fold.max() - per_fold.min() <= 1, per_fold


def test_the_folded_float64_network_is_the_modules_in_evaluation_mode() -> None:
    torch = pytest.importorskip("torch")
    s = _script()
    torch.manual_seed(0)
    model = s.student("square")
    with torch.no_grad():                     # batch norm statistics as a trained network has them, not the defaults
        model.bn.running_mean.copy_(torch.tensor([0.1, -0.2, 0.05, 0.3]))
        model.bn.running_var.copy_(torch.tensor([0.5, 1.5, 0.8, 2.0]))
        model.bn.weight.copy_(torch.tensor([1.2, 0.7, -0.4, 0.9]))
        model.bn.bias.copy_(torch.tensor([0.0, 0.1, -0.1, 0.2]))
    model.eval()
    crop = np.random.default_rng(1).uniform(-1, 1, (s.SIDE,) * 3)
    with torch.no_grad():
        want = model(torch.tensor(crop[None, None], dtype=torch.float32)).double().numpy()[0]
    got = s.plain_logits(crop, s.folded(model))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
    # The control is the same network with a ReLU in place of the square.
    control = s.student("control")
    control.load_state_dict(model.state_dict())
    control.eval()
    with torch.no_grad():
        relu = control(torch.tensor(crop[None, None], dtype=torch.float32)).double().numpy()[0]
    assert not np.allclose(relu, want), "the control arm computes the square arm's network"

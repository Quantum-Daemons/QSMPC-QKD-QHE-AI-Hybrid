"""The chest X-ray federation's analysis (Round 86, `scripts/train/medical_cxr_fl.py`), without the data or a GPU.

The figures in `docs/measurements/cxr-fl-2026-11-14.json` rest on a few pieces of plain arithmetic, held here:

- duplicates: a pixel-identical group across the official split loses its train_val copies, any other group all but one;
- hospitals: patient-level (no patient in two hospitals, or in a hospital and its validation), every train_val image placed
  once, no test image placed, deterministic in the seed, and the AP shares ordered as the targets are;
- the AUROC equals scikit-learn's, ties included, and its patient-bootstrap interval contains it;
- the operating point reaches the asked sensitivity on the set it is chosen on, and no higher threshold does;
- the CAM scoring: a peak inside the box is a hit, outside is not unless within Zhang et al.'s 15 px, and the box's area
  share is what a uniform map scores; the centre baseline points at the centre.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "train" / "medical_cxr_fl.py"


@pytest.fixture(scope="module")
def fl():
    spec = importlib.util.spec_from_file_location("medical_cxr_fl_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _prep(tmp: Path) -> tuple[Path, Path]:
    prep, raw = tmp / "prep", tmp / "raw"
    prep.mkdir()
    raw.mkdir()
    rows = [  # position, image, patient, view, split, effusion
        ("a.png", 1, "PA", "test", 1), ("b.png", 2, "PA", "train_val", 1),   # a and b identical, across the split
        ("c.png", 3, "AP", "train_val", 0), ("d.png", 3, "AP", "train_val", 0),  # c and d identical, one patient
        ("e.png", 4, "AP", "test", 0), ("f.png", 5, "PA", "train_val", 0),
    ]
    with open(prep / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "image", "patient", "view", "split", "effusion", "age", "sex", "follow_up", "pixel_sha256", "effusion_box"])
        for i, (name, pat, view, split, y) in enumerate(rows):
            w.writerow([i, name, pat, view, split, y, 50, "F", 0, "x", int(name == "a.png")])
    with open(prep / "duplicates.json", "w") as f:
        json.dump({"groups": [["a.png", "b.png"], ["c.png", "d.png"]], "cross_split": [["a.png", "b.png"]], "images": 6}, f)
    with open(raw / "BBox_List_2017.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Image Index", "Finding Label", "Bbox [x", "y", "w", "h]"])
        w.writerow(["a.png", "Effusion", "100", "200", "300", "400"])
        w.writerow(["e.png", "Mass", "1", "1", "1", "1"])
    return prep, raw


def test_duplicates_keep_the_test_copy_and_one_of_each_other_group(fl, tmp_path: Path) -> None:
    d = fl.load(*_prep(tmp_path))
    assert d["names"][int(np.flatnonzero(d["drop"])[0])] == "b.png"
    assert [d["names"][i] for i in np.flatnonzero(d["drop"])] == ["b.png", "d.png"]
    assert d["boxes"] == {0: (100.0, 200.0, 300.0, 400.0)}, "only Effusion boxes are kept"


def _synthetic(n_patients: int = 600, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    patient, ap, test, y = [], [], [], []
    for p in range(n_patients):
        k = int(rng.integers(1, 6))
        share = rng.choice([0.0, 0.0, 1.0, 0.5])
        is_test = rng.random() < 0.2
        for _ in range(k):
            patient.append(p)
            ap.append(rng.random() < share if share not in (0.0, 1.0) else bool(share))
            test.append(is_test)
            y.append(float(rng.random() < 0.15))
    n = len(patient)
    return dict(n=n, patient=np.array(patient), ap=np.array(ap), test=np.array(test), y=np.array(y, np.float32),
                drop=np.zeros(n, bool), boxes={})


def test_hospitals_are_patient_level_complete_and_deterministic(fl) -> None:
    d = _synthetic()
    tr, va, k = fl.hospitals(d, 1)
    assert k in fl.SHARPNESS
    placed = np.concatenate(tr + va)
    assert len(placed) == len(set(placed.tolist())), "an image placed twice"
    assert set(placed.tolist()) == set(np.flatnonzero(~d["test"]).tolist()), "a train_val image unplaced, or a test image placed"
    owner = {}
    for h, part in enumerate(tr + va):
        for p in np.unique(d["patient"][part]):
            assert p not in owner, f"patient {p} in two places"
            owner[p] = h
    tr2, va2, k2 = fl.hospitals(d, 1)
    assert k2 == k and all(np.array_equal(a, b) for a, b in zip(tr + va, tr2 + va2, strict=True))
    shares = [d["ap"][t].mean() for t in tr]
    assert shares == sorted(shares), f"AP shares {shares} not ordered as the targets"


def test_the_auroc_is_scikit_learns_and_its_interval_contains_it(fl) -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(800) < 0.3).astype(int)
    p = np.round(rng.normal(y * 0.8, 1.0), 1)          # rounded, so there are ties
    patients = rng.integers(0, 200, 800)
    r = fl.auc_boot(y, p, patients, n_boot=200, seed=1)
    assert r["auroc"] == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert r["lo"] <= r["auroc"] <= r["hi"] and r["hi"] - r["lo"] < 0.2
    assert fl.auc_boot(np.zeros(5), np.arange(5.0), np.arange(5)) is None


def test_the_operating_point_reaches_its_sensitivity_and_no_higher_threshold_does(fl) -> None:
    rng = np.random.default_rng(4)
    y = (rng.random(500) < 0.2).astype(int)
    p = rng.normal(y, 1.0)
    thr = fl.operating_point(y, p, 0.8)
    sens = ((p >= thr) & (y == 1)).sum() / (y == 1).sum()
    assert sens >= 0.8
    higher = np.sort(p[y == 1])
    nxt = higher[higher > thr]
    if len(nxt):
        assert ((p >= nxt[0]) & (y == 1)).sum() / (y == 1).sum() < 0.8


def test_the_cam_scoring(fl) -> None:
    boxes = {0: (512.0, 512.0, 256.0, 256.0), 1: (0.0, 0.0, 128.0, 128.0)}   # on the 1024 px image
    cam = np.zeros((2, 224, 224))
    cam[0, 140, 140] = 1.0      # (140, 140) at 224 is (640, 640) at 1024: inside box 0
    cam[1, 200, 200] = 1.0      # far from box 1
    cam[1, 30, 30] = 0.5        # (30, 30) at 224 is (137, 137) at 1024: 9 px outside box 1, within the 15 px tolerance
    r = fl.score_cams(cam, np.array([0, 1]), boxes, np.array([True, False]))
    assert r["AP"]["pointing_hit_rate"] == 1.0 and r["PA"]["pointing_hit_rate"] == 0.0
    assert r["AP"]["mass_in_box"] == 1.0 and r["PA"]["mass_in_box"] == 0.0
    assert r["AP"]["box_area_share"] == pytest.approx((56 / 224) ** 2)
    cam[1, 200, 200] = 0.0      # now the maximum is the near miss
    r = fl.score_cams(cam, np.array([0, 1]), boxes, np.array([True, False]))
    assert r["PA"]["pointing_hit_rate"] == 0.0 and r["PA"]["pointing_hit_rate_15px"] == 1.0


def test_the_centre_baseline_points_at_the_centre(fl) -> None:
    boxes = {0: (448.0, 448.0, 128.0, 128.0), 1: (0.0, 0.0, 100.0, 100.0)}
    r = fl.centre_baseline(boxes, np.array([0, 1]), np.array([False, False]))
    assert r["all"]["pointing_hit_rate"] == 0.5


def test_the_card_shows_the_record_and_credits_nih() -> None:
    record = json.loads((REPO / "docs" / "measurements" / "cxr-fl-2026-11-14.json").read_text())
    card_text = (REPO / "web" / "public" / "data" / "medical" / "cxr_fl.json").read_text()
    card = json.loads(card_text)
    for key in ("summary", "data", "attribution", "recipe", "untrained_view_probe_auroc"):
        assert card[key] == record[key], f"the card's {key} is not the record's"
    att = card["attribution"]
    assert att["provider"] == "NIH Clinical Center" and att["download"] == "https://nihcc.app.box.com/v/ChestXray-NIHCC"
    assert "CVPR 2017" in att["citation"]
    assert "/home/" not in card_text and "/home/" not in json.dumps(record), "a local path reached a shipped file"
    assert card["data"]["boxes"]["all_in_test_split"] and card["data"]["boxes"]["effusion_boxes"] == 153
    seeds = {name: arm["seeds"] for name, arm in card["summary"].items()}
    assert seeds.pop("FedAvg ConvNeXt-atto") == 1 and set(seeds.values()) == {3}
    ap = [h["ap_share"] for h in card["data"]["hospitals"]]
    assert ap == sorted(ap)

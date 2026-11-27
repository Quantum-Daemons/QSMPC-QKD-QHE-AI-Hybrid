"""The 5G-NIDD models under leakage-aware protocols (Round 86, batch 9; checklist MS33), held to their pre-registration.

`references/telecom-nidd-clean-protocols-preregistration.md` fixed the four splits, the input sets, the single-feature
check and the arms before any model was fitted under them; `scripts/train/telecom_nidd_clean_protocols.py` writes one
record, `docs/measurements/telecom-nidd-clean-protocols-2026-11-27.json`, an arm at a time. The flows are not in CI, so
what is held here is what the committed files say about each other, and the code on synthetic rows:

  - the record ran under the committed pre-registration (its SHA-256 is the file's and the script's);
  - the single-feature check's flags are exactly its values at or above the pre-registered 0.99;
  - each split's parts add up as the pre-registration says (P0, P2 and P3 cover every row once; P1 tests on the other
    station's rows);
  - the splits and the check's threshold, on synthetic rows, do what the pre-registration says (disjoint parts, P2
    time-ordered in each group, cuts rounded down, ties on one side of a threshold, absent classes left out).
"""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "train" / "telecom_nidd_clean_protocols.py"
PROTOCOLS = REPO / "scripts" / "train" / "nidd_protocols.py"
RECORD = REPO / "docs" / "measurements" / "telecom-nidd-clean-protocols-2026-11-27.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.cache
def _m():
    return _load("telecom_nidd_clean_protocols", SCRIPT)


def _r() -> dict:
    return json.loads(RECORD.read_text())


def test_the_record_ran_under_the_committed_preregistration() -> None:
    r, m = _r(), _m()
    assert hashlib.sha256((REPO / r["preregistration"]["file"]).read_bytes()).hexdigest() == r["preregistration"]["sha256"]
    assert r["preregistration"]["sha256"] == m.PREREG_SHA256
    assert r["classes"][0] == "Benign" and len(r["classes"]) == 9


def test_the_flags_are_the_values_at_the_preregistered_threshold() -> None:
    r, m = _r(), _m()
    assert m.FLAG_F1 == 0.99
    for split, v in r["single_feature_check"].items():
        values = v["validation_f1_at_least_0_5"]
        expected = {n: {c: f for c, f in d.items() if f >= m.FLAG_F1} for n, d in values.items()}
        expected = {n: d for n, d in expected.items() if d}
        assert v["flagged"] == expected, split
        assert all(0.5 <= f <= 1.0 for d in values.values() for f in d.values()), split


def test_each_split_adds_up() -> None:
    r = _r()
    n = r["rows"]
    for name in ("P0", "P2", "P3"):
        assert sum(r["parts"][name]) == n, name
    a, b = r["parts"]["P1_train_1_test_2"], r["parts"]["P1_train_2_test_1"]
    # Train and validate on one station, test on every row of the other: the two directions' test parts are the stations.
    assert a[0] + a[1] == b[2] and b[0] + b[1] == a[2] and a[2] + b[2] == n


def test_the_splits_on_synthetic_rows() -> None:
    P = _load("nidd_protocols", PROTOCOLS)
    P._self_test()
    assert P.cuts(10) == (7, 8) and P.cuts(3) == (2, 2) and P.cuts(1) == (0, 0)


def test_the_threshold_and_the_scores_on_synthetic_rows() -> None:
    m = _m()
    rng = np.random.default_rng(1)
    x = np.r_[rng.normal(0, 1, 500), rng.normal(5, 1, 500)]
    t = np.r_[np.zeros(500, bool), np.ones(500, bool)]
    f1, _, sign = m.best_threshold(x, t)
    assert f1 > 0.99 and sign == 1
    assert m.best_threshold(-x, t)[2] == -1
    # Two tied values: the cut falls between them, never inside a run of equal values.
    assert m.best_threshold(np.repeat([0.0, 1.0], 500), t)[0] == 1.0
    s = m.scores(np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, 1, 2]), 4)
    assert s["classes_absent"] == [3] and abs(s["macro_f1"] - np.mean([2 / 3, 0.8, 1.0])) < 1e-6

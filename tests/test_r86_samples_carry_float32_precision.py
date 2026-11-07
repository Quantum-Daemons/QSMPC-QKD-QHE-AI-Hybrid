"""An evaluation sample carries its features at the precision its graph reads them, float32 (Round 86).

The samples were written with every feature rounded to 6 decimals. That keeps a value of order one, and leaves a value
of order 1e-5 with one or two significant digits. OPS-SAT-AD has hundreds of such values: 8 of its 529 test decisions
flipped. The page then reproduced 0.9509 for a graph whose accuracy on the partition is 0.9660, and the sample's note
blamed the export, which in fact reproduces the scikit-learn model on every row.

Two of the samples are a whole published partition, so their expected figures must be their records'. That holds
the samples to the records without the corpus, which CI does not have. The helper is checked to round-trip, and the
builders that ship raw features are checked to use it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import prep_model_inputs as P  # noqa: E402


def test_float32_shortest_reads_back_as_the_same_float32() -> None:
    rng = np.random.default_rng(0)
    a = np.concatenate([rng.normal(size=500) * 10.0 ** rng.integers(-9, 6, size=500), [0.0, 1.0584846e-05, 255.0]])
    out = P.float32_shortest(a)
    assert np.array_equal(out.astype(np.float32), a.astype(np.float32))
    # The value that started this: 6 decimals made it 1.1e-05.
    assert out[-2] != round(float(np.float32(1.0584846e-05)), 6) and np.float32(out[-2]) == np.float32(1.0584846e-05)


def test_the_whole_partition_samples_reproduce_their_records() -> None:
    m = json.loads((REPO / "models" / "metrics.json").read_text())
    teacher = json.loads((REPO / "models" / "satellite_opssat" / "eval_sample.json").read_text())
    student = json.loads((REPO / "models" / "satellite_opssat_student" / "eval_sample.json").read_text())
    assert len(teacher["labels"]) == len(student["labels"]) == m["satellite_comms.opssat"]["n_test_segments"] == 529
    assert teacher["expected"]["accuracy"] == m["satellite_comms.opssat"]["accuracy"], (
        "the OPS-SAT sample is the whole test partition, so it must reproduce the record's accuracy")
    assert round(student["expected"]["minority_class_f1"], 6) == m["satellite_comms.opssat_student"]["student_metric"]
    assert "blamed the export" in teacher["expected"]["note"] and "before export, which predicts" not in teacher["expected"]["note"]


def test_the_builders_that_ship_raw_features_write_float32_precision() -> None:
    src = (REPO / "scripts" / "prep_model_inputs.py").read_text()

    def body(name: str) -> str:
        start = src.index(f"def {name}(")
        return src[start:src.index("\ndef ", start + 1)]

    for name in ("build_finance", "build_satellite_opssat", "build_telecom", "build_opssat_student"):
        assert "float32_shortest(" in body(name), f"{name} no longer writes its features at float32 precision"
    assert ".round(6)" not in body("build_satellite_opssat") and ".round(6)" not in body("build_telecom")

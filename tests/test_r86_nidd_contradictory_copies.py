"""One attacker's flood is in 5G-NIDD twice under contradictory labels, and nothing published reads the copy (Round 86).

The published file holds the same 281,529 flows labelled Benign in the first base station's file and UDPFlood
in the second's. On those rows no model reading flow statistics can pass accuracy 0.7685, and every higher score
was reading which file a row came from. `load_5g_nidd` removes the Benign copy.

Nothing here needs the raw file except the last test: the rule is exercised on made-up rows, and the published
sample and the record are tracked.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsh.models import datasets as D

REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "models" / "telecom_nidd" / "eval_sample.json"
PUBLISHED_ROWS = 1_215_890


def _rows(*rows: tuple) -> pd.DataFrame:
    """Rows of (attack type, source, destination port, bytes); the other key columns are fixed."""
    out = []
    for kind, src, port, size in rows:
        out.append({"StartTime": "03:49.0", "LastTime": "03:49.0", "SrcAddr": src, "DstAddr": "10.0.0.1", "Sport": "1234",
                    "Dport": port, "Proto": "udp", "TotPkts": 1, "TotBytes": size, "Dur": 0.0, "Attack Type": kind})
    return pd.DataFrame(out)


def test_a_benign_row_recording_a_flow_also_labelled_as_an_attack_is_marked() -> None:
    df = _rows(("Benign", "10.0.0.7", "80", 42), ("UDPFlood", "10.0.0.7", "80", 42))
    assert D.nidd_contradictory_copies(df).tolist() == [True, False]


def test_flows_without_a_port_still_match_each_other() -> None:
    df = _rows(("Benign", "10.0.0.7", None, 42), ("ICMPFlood", "10.0.0.7", None, 42))
    assert D.nidd_contradictory_copies(df).tolist() == [True, False]


def test_a_flow_differing_in_one_key_column_is_left_alone() -> None:
    df = _rows(("Benign", "10.0.0.7", "80", 43), ("UDPFlood", "10.0.0.7", "80", 42),
               ("Benign", "10.0.0.8", "80", 42), ("Benign", "10.0.0.7", "81", 42))
    assert not D.nidd_contradictory_copies(df).any()


def test_two_benign_copies_are_not_a_contradiction() -> None:
    df = _rows(("Benign", "10.0.0.7", "80", 42), ("Benign", "10.0.0.7", "80", 42))
    assert not D.nidd_contradictory_copies(df).any()


def test_no_feature_vector_in_the_published_sample_is_under_two_classes() -> None:
    d = json.loads(SAMPLE.read_text())
    rows = np.ascontiguousarray(np.asarray(d["samples"], dtype=np.float64))
    labels = np.asarray(d["labels"], dtype=np.int64)
    keys = rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()
    _, group = np.unique(keys, return_inverse=True)
    per_group = np.unique(group * len(d["class_names"]) + labels) // len(d["class_names"])
    shared = len(per_group) - len(np.unique(per_group))
    assert shared == 0, f"{shared} feature vectors in the published sample appear under more than one class"


def test_the_record_is_the_cleaned_fit_and_keeps_the_rows_as_published_beside_it() -> None:
    m = json.loads((REPO / "models" / "metrics.json").read_text())["telecom_ids.nidd"]
    assert m["n_samples"] == PUBLISHED_ROWS - D.NIDD_CONTRADICTORY_COPIES
    assert m["flow_level_ceiling"]["accuracy"] > 0.9999
    as_published = m["contradictory_copies"]["as_published"]
    assert as_published["n_samples"] == PUBLISHED_ROWS
    # The ceiling on those rows is the arithmetic of the copies, which is what identified them.
    assert as_published["flow_level_ceiling"]["accuracy"] == pytest.approx(1 - D.NIDD_CONTRADICTORY_COPIES / PUBLISHED_ROWS, abs=2e-5)
    assert as_published["accuracy"] < as_published["flow_level_ceiling"]["accuracy"] < m["aggregate_accuracy"]


def test_the_loader_removes_exactly_the_copies_on_the_real_file() -> None:
    try:
        D.nidd_fields_preserved_csv()
    except FileNotFoundError:
        pytest.skip("the raw 5G-NIDD file is not on this machine")
    x, y, _, classes = D.load_5g_nidd()
    assert len(y) == PUBLISHED_ROWS - D.NIDD_CONTRADICTORY_COPIES
    counts = dict(zip(classes, np.bincount(y).tolist(), strict=True))
    assert counts["Benign"] == 477_737 - D.NIDD_CONTRADICTORY_COPIES and counts["UDPFlood"] == 457_340
    _, y_kept, _, _ = D.load_5g_nidd(with_contradictory_copies=True)
    assert len(y_kept) == PUBLISHED_ROWS

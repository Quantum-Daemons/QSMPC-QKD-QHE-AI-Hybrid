"""No published model reads a capture time as a feature (Round 86).

`telecom_ids.nidd` read `IdleTime` from Round 50 to Round 86 and scored accuracy 0.9829. The column held
17 values, all multiples of 128, which as Unix times fall in the two days the corpus was captured, each
dominated by one attack scenario. It was argued to be a flow statistic because 17 values cannot address a
row, and every leak check passed it, because one-vs-rest AUC cannot tell a capture time from a statistic.
Reading the values could, so this reads them, for every published evaluation sample.

A column is a capture time here when every value in it lies in the range of plausible Unix times, 2001 to
2033. No statistic of a flow or a transaction falls there on every row: the TCP base sequence numbers in
5G-NIDD range over 32 bits and are not flagged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SAMPLES = sorted(REPO.glob("models/*/eval_sample.json"))
LOW, HIGH = 1e9, 2e9


def _unix_time_columns(rows: np.ndarray, names: list[str]) -> list[str]:
    finite = np.where(np.isfinite(rows), rows, 0.0)
    return [names[j] for j in range(finite.shape[1]) if LOW < finite[:, j].min() and finite[:, j].max() < HIGH]


def _tabular(path: Path) -> tuple[np.ndarray, list[str]] | None:
    d = json.loads(path.read_text())
    names, rows = d.get("feature_names"), d.get("samples")
    if not names or not rows or not isinstance(rows[0], list) or len(rows[0]) != len(names) or isinstance(rows[0][0], list):
        return None
    return np.asarray(rows, dtype=float), list(names)


def test_there_are_tabular_samples_to_check() -> None:
    assert sum(_tabular(p) is not None for p in SAMPLES) >= 2


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.parent.name)
def test_no_published_sample_has_a_unix_time_column(path: Path) -> None:
    table = _tabular(path)
    if table is None:
        pytest.skip("an image or a sequence sample, with no named tabular columns")
    rows, names = table
    found = _unix_time_columns(rows, names)
    assert not found, f"{path.parent.name} reads {found}, whose values all lie in the range of Unix times: a capture time names the session, not the event"


def test_the_check_fires_on_the_column_that_got_through() -> None:
    rows = np.array([[1663055360.0, 42.0], [1663276160.0, 84.0], [1663272960.0, 1392.0]])
    assert _unix_time_columns(rows, ["IdleTime", "TotBytes"]) == ["IdleTime"]
    wide = np.array([[0.0], [4.29e9], [1.66e9]])
    assert _unix_time_columns(wide, ["SrcTCPBase"]) == [], "a column that merely reaches Unix-time values is not one"


def test_the_nidd_loader_refuses_to_read_it_back() -> None:
    src = (REPO / "scripts" / "train" / "telecom_nidd.py").read_text()
    assert "assert not set(D.NIDD_CAPTURE_TIME) & set(feature_names)" in src
    datasets = (REPO / "src" / "qsh" / "models" / "datasets.py").read_text()
    features = datasets[datasets.index("NIDD_FEATURES = ("):]
    assert "IdleTime" not in features[: features.index(")")]

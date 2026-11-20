"""F1 by the receiving account's in-degree, in SALT-GNN's bins (Round 86, the owner's choice of 2026-11-15).

`docs/measurements/finance-degree-f1-2026-11-20.json` is computed offline by `scripts/research/finance_degree_f1.py` from
IBM's Multi-PNA+EU and Multi-GIN+EU runs' test and validation scores, five seeds per protocol. Here: the record holds
together without the data (SALT-GNN's six bins, rows and positives adding up to the test window, five runs whose mean is
the mean shown, ordered intervals, the paper's published rows as recorded), each of those refusals fires on a seeded
fault, and where the runs' dumps are on this machine the whole record is recomputed and must agree.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "research"))

import finance_degree_f1 as D  # noqa: E402

RECORD_PATH = REPO / "docs" / "measurements" / "finance-degree-f1-2026-11-20.json"
RECORD = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
#: Where the runs' dumps and IBM's formatted HI-Small file are on the machine that ran them (outside the repository).
DATA = Path(os.environ.get("QSH_DEGREE_F1_DATA", REPO.parent / "qsh-gnn-work"))


def test_the_record_holds_together() -> None:
    assert D.check(RECORD) == []
    assert RECORD["test_rows"] == 1_523_504 and RECORD["test_positives"] == 2_321


def test_a_renamed_bin_is_refused() -> None:
    r = copy.deepcopy(RECORD)
    r["bins_by_window"]["all"][4]["bin"] = "50-100"
    assert any("not SALT-GNN's" in p for p in D.check(r))


def test_a_changed_count_is_refused() -> None:
    r = copy.deepcopy(RECORD)
    r["bins_by_window"]["batch"][2]["rows"] += 1
    assert any("do not add up to the test window" in p for p in D.check(r))


def test_a_changed_run_is_refused() -> None:
    r = copy.deepcopy(RECORD)
    r["families"]["pna_ibm"]["by_window"]["all"]["0.5"]["bins"][4]["f1_runs"][0] += 0.01
    assert any("the mean is not the runs' mean" in p for p in D.check(r))


def test_a_changed_published_row_is_refused() -> None:
    r = copy.deepcopy(RECORD)
    r["published"]["rows"][4]["f1"][4][0] = 0.761
    assert any("SALT-GNN's Table 2" in p for p in D.check(r))


def test_every_family_has_both_windows_and_both_thresholds() -> None:
    for key, fam in RECORD["families"].items():
        assert set(fam["by_window"]) == {"all", "batch"}, key
        for w in ("all", "batch"):
            assert set(fam["by_window"][w]) == {"0.5", "validation"}, (key, w)
        assert fam["native_window"] == D.FAMILIES[key][2]
        for run in fam["runs"]:
            assert run["scored"] + run["unscored"] == RECORD["test_rows"], run


def test_recomputed_from_the_runs_where_they_are() -> None:
    transactions = DATA / "aml_data" / "Small_HI" / "formatted_transactions.csv"
    if not transactions.exists() or not list(DATA.glob("pna_project_fitbn_seed1")):
        pytest.skip(f"the runs' dumps are not on this machine ({DATA}): the record is checked for consistency only, "
                    "not recomputed; set QSH_DEGREE_F1_DATA to the folder that holds them")
    fresh = D.run(transactions, DATA, bootstrap=RECORD_BOOTSTRAP, seed=RECORD_SEED)
    assert fresh == RECORD, "the record no longer follows from the runs' scores"


RECORD_BOOTSTRAP = 1000
RECORD_SEED = 0


def test_the_page_serves_the_record_byte_for_byte() -> None:
    """web/public/data/finance/degree-f1.json is what the IBM tab fetches; the script writes it with --web-copy."""
    served = REPO / "web" / "public" / "data" / "finance" / "degree-f1.json"
    assert served.read_bytes() == RECORD_PATH.read_bytes()
    component = (REPO / "web" / "src" / "components" / "DegreeF1Table.tsx").read_text(encoding="utf-8")
    assert "data/finance/degree-f1.json" in component and "<StageBadge kind=\"offline\" />" in component

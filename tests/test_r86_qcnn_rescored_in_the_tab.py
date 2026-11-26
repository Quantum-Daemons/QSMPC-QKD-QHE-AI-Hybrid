"""The trained QCNN's held-out figure is re-scored in the tab, on the trainer's own test rows (Round 86).

WHY THIS EXISTS

Every QCNN figure the pages printed came from the weights file, and the log said so: "recorded at training time ... not re-scored in this
tab". The CKKS and ONNX tiers are re-scored on their held-out rows; the circuit's headline was the one a visitor could not reproduce (audit
of 2026-11-08, lead P3-O1). `scripts/data/build_qcnn_heldout.py` now ships each use case's test rows, and the page runs the circuit over
them (`web/src/lib/qcnnRescore.ts`). This file holds the Python half of the evidence; `web/scripts/check_qcnn_rescore.mjs` the browser half.

WHAT IS ASSERTED

1. The shipped rows are what the builder produces from the trainer's `load()` and `shuffled_split()`, and the trainer's own circuit
   reproduces every recorded test metric from them (`--check` exits 0 only then, and writes nothing).
2. The browser check runs where the other gates run: npm, the local gate and CI.
3. What the page shows comes from the re-score: the circuit card's button and the CKKS panel's row, and DATA_LICENSES.md lists the files.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PUBLIC = REPO / "web" / "public" / "data"


def test_the_shipped_rows_are_the_trainer_s_split_and_reproduce_every_recorded_figure() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "data" / "build_qcnn_heldout.py"), "--check"],
                       capture_output=True, text=True, env=env, cwd=REPO, timeout=900)
    assert r.returncode == 0, f"build_qcnn_heldout.py --check failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    # Medical and telecom: finance's file is written by scripts/export_finance_qcnn_amlworld.py since Round 86's batch 6,
    # because its corpus is not in the repository; the test below holds it to its figures.
    assert r.stdout.count("as built") == 2 and r.stdout.count("rows reproduce") == 3, r.stdout


def test_the_finance_sample_reproduces_the_figures_it_records() -> None:
    """Finance since Round 86's batch 6: the page's 8,192 test-window transactions, a sample of the circuit's 1,523,504, at
    its eight inputs. The re-score is held to `expected`, the trainer's circuit on these rows at the served threshold, and
    this recomputes it with the Python tier's contraction; the copy the Python pipeline reads is the page's, byte for byte."""
    from sklearn.metrics import accuracy_score, f1_score

    sys.path.insert(0, str(REPO / "src"))
    from qsh.qhe import qcnn_tree

    web = PUBLIC / "finance" / "qcnn_heldout.json"
    assert web.read_bytes() == (REPO / "data" / "finance" / "qcnn_heldout.json").read_bytes()
    held, w = json.loads(web.read_text()), json.loads((PUBLIC / "finance" / "qcnn_weights.json").read_text())
    assert held["feature_names"] == w["feature_names"] and held["input_space"] == w["input_space"] == "raw"
    X, y = np.asarray(held["rows"], dtype=np.float64), np.asarray(held["labels"])
    pred = (qcnn_tree.readout(w, X) * float(w["scale"]) + float(w["bias"]) > 0).astype(int)
    assert round(float(f1_score(y, pred)), 6) == held["expected"]["test_minority_f1"]
    assert round(float(accuracy_score(y, pred)), 6) == held["expected"]["test_accuracy"]
    assert w["n_test"] > len(y) == 8192


@pytest.mark.parametrize("data_dir", ["finance", "medical", "telecom"])
def test_each_file_carries_its_licence_and_says_which_split_it_is(data_dir: str) -> None:
    held = json.loads((PUBLIC / data_dir / "qcnn_heldout.json").read_text())
    assert held["licence"] and held["licence_url"].startswith("https://") and held["attribution"]
    assert len(held["rows"]) == len(held["labels"]) > 0
    if "expected" in held:
        # A sample of the split, not the split (finance since Round 86's batch 6): it says so, and whose figures are whose.
        assert "test-window transactions" in held["split"] and "never trained, chose a checkpoint or a threshold on them" in held["split"]
        assert "whole test window" in held["split"] and "weights file's" in held["split"]
        for ref in held["reads"]:
            assert json.loads((PUBLIC / ref).read_text())["n_test"] > len(held["labels"]), ref
        return
    assert "shuffled with seed" in held["split"] and "never trained or chose a checkpoint on them" in held["split"]
    for ref in held["reads"]:
        assert json.loads((PUBLIC / ref).read_text())["n_test"] == len(held["labels"]), ref


def test_the_browser_check_runs_where_the_other_gates_run() -> None:
    assert '"check:qcnn-rescore"' in (REPO / "web" / "package.json").read_text()
    assert "check:qcnn-rescore" in (REPO / "scripts" / "ci_local.sh").read_text()
    assert "check:qcnn-rescore" in (REPO / ".github" / "workflows" / "ci.yml").read_text()


def test_the_page_shows_the_re_score_and_the_licences_list_the_files() -> None:
    card = (REPO / "web" / "src" / "components" / "QcnnCircuitCard.tsx").read_text()
    assert "rescoreTrainedQcnn(usecase, weights)" in card and "Re-score on the held-out rows" in card
    panel = (REPO / "web" / "src" / "components" / "QcnnCkksPanel.tsx").read_text()
    assert "re-scored in this tab" in panel and "rescore.matchesRecord" in panel
    page = (REPO / "web" / "src" / "pages" / "UseCasePage.tsx").read_text()
    assert "rescore={(artifacts.qcnn_rescore" in page
    licences = (REPO / "DATA_LICENSES.md").read_text()
    for data_dir in ("finance", "medical", "telecom"):
        assert f"web/public/data/{data_dir}/qcnn_heldout.json" in licences, data_dir

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
    assert r.stdout.count("as built") == 3 and r.stdout.count("rows reproduce") == 4, r.stdout


@pytest.mark.parametrize("data_dir", ["finance", "medical", "telecom"])
def test_each_file_carries_its_licence_and_says_which_split_it_is(data_dir: str) -> None:
    held = json.loads((PUBLIC / data_dir / "qcnn_heldout.json").read_text())
    assert held["licence"] and held["licence_url"].startswith("https://") and held["attribution"]
    assert "shuffled with seed" in held["split"] and "never trained or chose a checkpoint on them" in held["split"]
    assert len(held["rows"]) == len(held["labels"]) > 0
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

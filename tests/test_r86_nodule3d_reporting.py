"""The 3D nodule student reported at each step (Round 86, batch 8; checklist MS29), held to its pre-registration.

`references/medical-nodule3d-reporting-preregistration.md` (with its addendum 5fef91e) asked for the teacher on the test
part, AUPRC beside every AUC, and the AUC of each CKKS run's decrypted logits, as Lee et al. report theirs; nothing is
adopted by these figures. `scripts/train/medical_nodule3d_report.py` wrote the record and
`scripts/export_nodule3d_reporting.py` the page's file. The crops are not in CI, so what is held here is what the
committed files say about each other:

  - the record ran under the committed pre-registration, on the committed student record (both by SHA-256);
  - each seed's AUC is the student record's within the addendum's two pairs, and each arm's mean and standard deviation
    are its five seeds';
  - the shipped seed is the student record's export seed, at its recorded AUC; each CKKS run's decrypted logits give the
    same AUC and AUPRC as its own plaintext, within the student record's 1e-3 logit tolerance, on every test crop;
  - the page file is exactly what its exporter writes from the record, and the 3D card mounts the section.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "docs" / "measurements" / "medical-nodule3d-reporting-2026-11-27.json"
STUDENT = REPO / "docs" / "measurements" / "medical-nodule3d-square-student.json"
SITE = REPO / "web" / "public" / "data" / "medical" / "nodule3d_reporting.json"
SCRIPT = REPO / "scripts" / "train" / "medical_nodule3d_report.py"


def _r() -> dict:
    return json.loads(RECORD.read_text())


def test_the_record_ran_under_the_committed_preregistration_and_student_record() -> None:
    r = _r()
    assert hashlib.sha256((REPO / r["preregistration"]["file"]).read_bytes()).hexdigest() == r["preregistration"]["sha256"]
    assert f'PREREG_SHA256 = "{r["preregistration"]["sha256"]}"' in SCRIPT.read_text()
    assert r["of_record"]["file"] == str(STUDENT.relative_to(REPO))
    assert hashlib.sha256(STUDENT.read_bytes()).hexdigest() == r["of_record"]["sha256"]


def test_each_seed_is_the_student_record_within_two_pairs_and_the_means_are_the_seeds() -> None:
    r, s = _r(), json.loads(STUDENT.read_text())
    t = r["test_part"]
    pair = 1.0 / (t["malignant"] * (t["crops"] - t["malignant"]))
    for arm in ("square", "control"):
        seeds = {k: v for k, v in r["students"][arm].items() if k != "mean"}
        assert sorted(seeds) == sorted(s["arms"][arm]), arm
        for k, v in seeds.items():
            assert v["recorded_test_auc_gpu"] == s["arms"][arm][k]["test_auc"]
            assert abs(v["auc"] - v["recorded_test_auc_gpu"]) <= 2 * pair + 1e-9, (arm, k)
            assert v["auprc_ci95"][0] <= v["auprc"] <= v["auprc_ci95"][1]
        m = r["students"][arm]["mean"]
        aucs, aps = [v["auc"] for v in seeds.values()], [v["auprc"] for v in seeds.values()]
        assert m["auc"] == pytest.approx(np.mean(aucs), abs=1e-6) and m["auc_sd"] == pytest.approx(np.std(aucs, ddof=1), abs=1e-6)
        assert m["auprc"] == pytest.approx(np.mean(aps), abs=1e-6) and m["auprc_sd"] == pytest.approx(np.std(aps, ddof=1), abs=1e-6)


def test_the_shipped_seed_and_its_encrypted_runs() -> None:
    r, s = _r(), json.loads(STUDENT.read_text())
    assert r["shipped_seed"]["seed"] == int(s["criteria"]["export_seed"])
    assert abs(r["shipped_seed"]["folded_float64"]["auc"] - s["export_bootstrap"]["auc"]) <= 1e-6
    assert sorted(r["encrypted"]) == ["A2", "B"]
    for name, e in r["encrypted"].items():
        assert e["crops"] == r["test_part"]["crops"], name
        assert e["max_abs_logit_error"] <= 1e-3, name
        for k in ("auc", "auprc"):
            assert abs(e["decrypted"][k] - e["plaintext_same_run"][k]) <= 1e-6, (name, k)
            assert abs(e["plaintext_same_run"][k] - r["shipped_seed"]["folded_float64"][k]) <= 1e-6, (name, k)


def test_the_page_file_is_what_the_exporter_writes() -> None:
    spec = importlib.util.spec_from_file_location("export_nodule3d_reporting", REPO / "scripts" / "export_nodule3d_reporting.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert json.dumps(mod.build(), indent=1) + "\n" == SITE.read_text()


def test_the_3d_card_mounts_the_section() -> None:
    card = (REPO / "web" / "src" / "components" / "Nodule3dCard.tsx").read_text()
    section = (REPO / "web" / "src" / "components" / "Nodule3dReportingSection.tsx").read_text()
    assert "<Nodule3dReportingSection" in card and "data/medical/nodule3d_reporting.json" in section

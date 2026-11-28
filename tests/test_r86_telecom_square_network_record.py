"""The telecom square network's pre-registered run, held to its own rules and to what the docs say of it (Round 86).

`references/telecom-nidd-square-student-preregistration.md` was committed before anything was trained, and
`scripts/train/telecom_nidd_square_student.py` refuses to run unless the file hashes to the SHA-256 it was frozen at.
The record (`docs/measurements/telecom-nidd-square-student-2026-11-28.json`) states each criterion's verdict; this
recomputes every verdict from the record's own figures by the pre-registration's rules, so a verdict written by hand
or by a changed script fails here, and holds the figures `docs/models-and-benchmarks.md` copies to the record.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "references" / "telecom-nidd-square-student-preregistration.md"
RECORD = REPO / "docs" / "measurements" / "telecom-nidd-square-student-2026-11-28.json"
RERUN = REPO / "docs" / "measurements" / "telecom-nidd-clean-protocols-2026-11-27.json"
DOC = REPO / "docs" / "models-and-benchmarks.md"
SEEDS = ("5", "6", "7", "8", "9")


def _record() -> dict:
    return json.loads(RECORD.read_text())


def test_the_record_was_written_under_the_frozen_preregistration() -> None:
    r = _record()
    assert r["preregistration"]["sha256"] == hashlib.sha256(PREREG.read_bytes()).hexdigest()
    script = (REPO / "scripts" / "train" / "telecom_nidd_square_student.py").read_text()
    assert f'PREREG_SHA256 = "{r["preregistration"]["sha256"]}"' in script


def test_the_control_is_the_reruns_linear_student() -> None:
    r, rerun = _record(), json.loads(RERUN.read_text())
    assert r["control"]["test"] == rerun["student"]["F30/P2"]["test"]
    assert r["control"]["l2"] == rerun["student"]["F30/P2"]["l2"]
    assert r["data"]["parts"] == rerun["parts"]["P2"]


def test_each_verdict_follows_from_the_records_figures_by_the_preregistrations_rules() -> None:
    r = _record()
    arms, c = r["arms"]["square"], r["criteria"]
    assert sorted(arms) == list(SEEDS)
    control = r["control"]["test"]["flamby_metric"]
    test = [arms[s]["test"]["flamby_metric"] for s in SEEDS]
    assert c["1_accuracy"] == (sum(test) / len(test) >= control + 0.015 and min(test) >= control + 0.010)
    assert c["2_no_collapse"] == all(arms[s]["test"]["macro_f1"] >= 0.90 and min(arms[s]["test_class_f1"].values()) > 0
                                     for s in SEEDS)
    assert r["export_seed"] == int(max(SEEDS, key=lambda s: (arms[s]["validation"]["flamby_metric"], -int(s))))
    for name, s in r["ckks"]["sets"].items():
        assert c["3_fidelity"][name] == (s["max_abs_error"] <= 1e-3 and s["decisions_differ_on_the_other_flows"] == 0)
        assert s["rows"] == 2048 and s["plaintext_matches_numpy_max_abs"] <= 1e-9
    for name, s in r["browser"]["sets"].items():
        assert len(s["runs"]) == 3
        meets = all(run["total_ms"] <= 30_000 and run["wasm_memory_bytes"] <= 1.5e9 and run["max_abs_error"] <= 1e-3
                    and run["decisions_differ_on_flows_clear_of_twice_the_error"] == 0 for run in s["runs"])
        assert c["4_page"][name] == meets == s["criterion_4"]
    offered = [n for n in ("A2", "B") if c["3_fidelity"][n] and c["4_page"][n]]
    assert c["sets_offered"] == offered
    assert c["adopted"] == bool(c["1_accuracy"] and c["2_no_collapse"] and offered)


def test_the_docs_paragraph_is_the_record() -> None:
    r, doc = _record(), DOC.read_text()
    arms = r["arms"]["square"]
    test = [arms[s]["test"]["flamby_metric"] for s in SEEDS]
    b = r["export_bootstrap"]
    ckks, browser = r["ckks"]["sets"], r["browser"]["sets"]
    expected = [
        f"FLamby metric of {r['control']['test']['flamby_metric']:.4f}",
        f"Seeds 5 to 9 score {min(test):.4f} to {max(test):.4f} (mean {sum(test) / len(test):.4f})",
        f"the exported seed, {r['export_seed']}, best on validation",
        f"interval of {b['low']:.4f} to {b['high']:.4f}",
        f"$`{ckks['A2']['max_abs_error'] / 1e-4:.1f} \\times 10^{{-4}}`$ under A2",
        f"$`{ckks['B']['max_abs_error'] / 1e-5:.1f} \\times 10^{{-5}}`$ under B",
        f"the slowest takes {browser['A2']['slowest_total_ms'] / 1000:.1f} s",
        f"and {browser['B']['slowest_total_ms'] / 1000:.1f} s under B",
        f"the worker's memory {browser['A2']['largest_wasm_memory_bytes'] / 1e9:.2f} and "
        f"{browser['B']['largest_wasm_memory_bytes'] / 1e9:.2f} GB",
    ]
    flat = " ".join(doc.split())
    for phrase in expected:
        assert phrase in flat, phrase
    assert min(min(arms[s]["test_class_f1"].values()) for s in SEEDS) > 0.95
    assert all(arms[s]["float64"]["test_decisions_differing_from_the_device"] == 0 for s in SEEDS)

"""The CNN student under MPC on all 156 test images has a record, and what the page and the docs say of it is the record.

Round 86, batch 12 (the owner's choice of 2026-11-28, "record the 156-image MPC run"): the Models panel's note, and the
Hub card it generates, say the student's class under MPC "matched the clear network on all 156 test images", and the
models doc gave a table for that run, measured on 2026-11-08, which left no record; `web/scripts/check_hybrid_cnn.mjs`
ran one image. `--all` now runs every image of the published sample through the module the page runs (the real
node-seal and Polytune, each image held to the one-image checks) and writes
`docs/measurements/medical-cnn-mpc-<date>.json`. Held here: the record covers the published sample's every image and
every image passed; the note's claim is the record's; and every figure of the doc's table is the record's.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "models" / "medical_breastmnist_student" / "eval_sample.json"
DOC = REPO / "docs" / "models-and-benchmarks.md"


def _record() -> tuple[Path, dict]:
    found = sorted((REPO / "docs" / "measurements").glob("medical-cnn-mpc-*.json"))
    if not found:
        pytest.fail("no record of the CNN student under MPC on all 156 images (check_hybrid_cnn.mjs --all writes it)")
    return found[-1], json.loads(found[-1].read_text())


def _section() -> str:
    doc = DOC.read_text()
    start = doc.index("#### The CNN student under MPC")
    return doc[start:doc.index("\n#### ", start + 10)]


def test_the_record_covers_every_image_and_every_image_passed() -> None:
    _, r = _record()
    sample = json.loads(SAMPLE.read_text())
    assert r["images"] == len(sample["samples"]) == 156
    assert r["class_matches_the_clear_network"] == r["images"]
    assert r["every_image_passed_the_checks"] is True
    assert len(r["and_gates_per_image"]) == 1 and len(r["relus_per_image"]) == 1
    assert r["largest_logit_error"] < 2e-3 and r["statistical_distance_log2"]["max"] < -20


def test_the_models_panel_note_is_the_record() -> None:
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from qsh.models import registry

    _, r = _record()
    note = registry.get("medical_fl.student").notes
    claim = re.search(r"its class matched the clear network on all (\d+) test images", note)
    assert claim, "the note no longer states the claim this test holds"
    assert int(claim.group(1)) == r["images"] == r["class_matches_the_clear_network"]


def test_the_docs_table_is_the_record() -> None:
    path, r = _record()
    s = " ".join(_section().split())
    expected = [
        path.name,
        f"the same on {r['class_matches_the_clear_network']} of {r['images']}",
        f"{r['accuracy']:.4f}",
        f"{r['malignant_detected']} of {r['malignant_images']}",
        f"$`{r['largest_logit_error'] / 1e-4:.1f} \\times 10^{{-4}}`$",
        f"the smallest logit gap is {r['smallest_logit_gap']:.3f}",
        f"$`2^{{{r['statistical_distance_log2']['max']:.1f}}}`$ to $`2^{{{r['statistical_distance_log2']['min']:.1f}}}`$",
        f"{r['and_gates_per_image'][0]:,}: {r['relus_per_image'][0]} ReLUs",
        f"{r['gc_megabytes_per_image']:.1f} MB in {r['batches_per_image'][0]} batches",
        f"{r['ckks_rotations_per_image'][0]} rotations and {r['conjugations_per_image'][0]} conjugations",
        f"{r['ckks_megabytes_per_image']:.1f} MB of ciphertexts, {r['key_megabytes']:.1f} MB of keys",
        f"{r['wall_clock_s_median']:.1f} s: garbled circuits {r['gc_s_median']:.1f} s, CKKS {r['ckks_s_median']:.2f} s",
    ]
    for phrase in expected:
        assert phrase in s, phrase

"""The student weights the page's CKKS tier reads from its own bundle are the pinned, published models (Round 86).

WHY THIS EXISTS

The held-out CKKS figure on finance, telecom and satellite (`heldOutCorpus` in web/src/lib/realData.ts) reads a
weights file bundled with the page, named by `web/public/data/<use case>/student_heldout.json`, not the file on the
Hub. Nothing compared the two: the page re-scores the PUBLISHED graph and then encrypts a bundled copy, and a copy
left behind by a republish would have encrypted another model under the published one's name. Finance's and
telecom's copies are byte-identical files; satellite's is a derived form (the scaler the published ONNX graph applies
moved into `input`, as `scripts/data/build_heldout_payloads.py` writes it), so for it the model itself is compared.

WHAT IS ASSERTED, on tracked files only, so it runs everywhere: each payload names its student; the student's
weights.json in its model directory is the file pinned for its revision (its SHA-256 in models/hub_file_hashes.json);
and the bundled copy is that file, byte for byte, or for satellite the same W, b, expansion and scaler.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PUBLIC = REPO / "web" / "public"
PINS = json.loads((REPO / "models" / "hub_file_hashes.json").read_text())


def _payload(usecase: str) -> tuple[dict, Path, Path]:
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from qsh.models import registry

    held = json.loads((PUBLIC / "data" / usecase / "student_heldout.json").read_text())
    key = held["model_key"]
    published = REPO / registry.ARTIFACT_DIRS[key] / "weights.json"
    pinned = PINS[key]["files"]["weights.json"]
    assert hashlib.sha256(published.read_bytes()).hexdigest() == pinned, (
        f"{key}: models/.../weights.json is not the file pinned at {PINS[key]['revision'][:8]}")
    return held, published, PUBLIC / held["weights"].lstrip("/")


@pytest.mark.parametrize("usecase", ["finance", "telecom"])
def test_the_bundled_copy_is_the_published_file(usecase: str) -> None:
    _, published, bundled = _payload(usecase)
    assert bundled.read_bytes() == published.read_bytes(), f"{bundled.relative_to(REPO)} differs from the pinned file"


def test_the_satellite_copy_is_the_published_model() -> None:
    _, published, bundled = _payload("satellite")
    pub, copy = json.loads(published.read_text()), json.loads(bundled.read_text())
    assert copy["W"] == pub["W"] and copy["b"] == pub["b"], "another model's coefficients"
    assert copy["student_key"] == pub["student_key"] and copy["n_parameters"] == pub["n_parameters"]
    client = pub["ckks_client"]
    assert copy["input"]["preprocessing"] == client["preprocessing"]
    assert copy["input"]["mean"] == client["mean"] and copy["input"]["scale"] == client["scale"], (
        "the bundled scaler is not the one the published file gives a CKKS client")
    assert copy["input"]["expansion"] == pub["input"]["expansion"]

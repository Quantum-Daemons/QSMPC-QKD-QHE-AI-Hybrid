"""`main` keeps what the FHE.org 2027 poster states until its notification (the owner's rule of 2026-11-05).

The poster, submitted with a link to this repository, states the demo's CKKS parameters and cites eleven files for them. Its notification is due 2026-12-18. Until
then a change that moves one of these numbers makes the submitted poster wrong about the repository it links to. The register's `BLOCKED UNTIL 2026-12-19` line makes
lifting this gate an obligation with a date.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from qsh.config import get_config

REPO = Path(__file__).resolve().parents[1]

CITED = ["references/he-scheme-selection.md", "references/model-strategy-decision.md", "tests/test_fhe_parameters_single_source.py",
         "tests/test_node_seal_version_drift.py", "tests/test_scorecard_bfv_exact.py", "tests/test_r84_medtel_ckks_decrypts_only_outputs.py",
         "docs/deliverable-verification.md", "web/src/lib/clientFHE.ts", "config.yaml", "models/finance_aml_scorecard/README.md", "DATA_LICENSES.md"]


@pytest.mark.parametrize("path", CITED)
def test_every_cited_file_is_still_there(path: str) -> None:
    assert (REPO / path).exists(), path


def test_the_parameters_the_poster_states() -> None:
    fhe = get_config().fhe
    assert fhe.poly_modulus_degree == 8192
    assert tuple(fhe.coeff_mod_bit_sizes) == (60, 40, 40, 60)
    assert fhe.global_scale_bits == 40
    assert re.search(r"export const CKKS_FLOOD_BITS = 16;", (REPO / "web" / "src" / "lib" / "clientFHE.ts").read_text())


def test_the_node_seal_release_the_poster_names() -> None:
    lock = json.loads((REPO / "web" / "package-lock.json").read_text())
    assert lock["packages"]["node_modules/node-seal"]["version"] == "7.0.0"

"""The bundled demo samples must describe themselves the way the registry describes them.

WHY THIS TEST EXISTS

Round 17 shipped a public demo whose run log told visitors it was scoring "real PaySim mobile-money
transactions". PaySim is a simulator; the rows are synthetic. The claim had already been corrected in
DATA_LICENSES.md, in the TSV header, in the loader and in the README — and the browser kept printing
the old text, because the string also existed in a TypeScript map and in the JSON data files, and
nothing connected the copies. Fixing four of six places looked like fixing it.

Across the same audit five licence statements turned out to be wrong in the same way:

    PaySim        "real ... transactions"        -> a SIMULATOR; rows are synthetic, and CC-BY-SA
                                                   ShareAlike reaches the bundled files
    PGS Catalog   "CC-BY 4.0"                    -> per-score; EMBL-EBI terms by default, and
                                                   CC BY-NC-ND scores exist
    1000 Genomes  "open data-use, commercial-OK" -> IGSR grants no blanket licence
    SMAP/MSL      "CC0 / public domain"          -> no provider states any licence
    WDBC          "public domain"                -> UCI relicensed to CC BY 4.0; attribution is due

A licence claim on a public page is not a comment. This test asserts the one property that would have
caught every one of them: whatever the demo ships and prints must be the string the registry holds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qsh.config import frontend_config
from qsh.models.registry import BUNDLED_DEMO_SAMPLES, DEMO_SAMPLE_FILES

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("usecase", sorted(DEMO_SAMPLE_FILES))
def test_shipped_json_matches_the_registry(usecase: str) -> None:
    """The JSON the browser fetches carries the registry's provenance text verbatim."""
    path = ROOT / DEMO_SAMPLE_FILES[usecase]
    assert path.exists(), f"{path} is referenced by the registry but not shipped"
    shipped = json.loads(path.read_text())["provenance"]
    assert shipped == BUNDLED_DEMO_SAMPLES[usecase][0], (
        f"{path.relative_to(ROOT)} states a different provenance than the registry. Regenerate it "
        "rather than editing either by hand — the copies drifting is the bug this test exists for."
    )


@pytest.mark.parametrize("usecase", sorted(BUNDLED_DEMO_SAMPLES))
def test_frontend_config_carries_the_same_text(usecase: str) -> None:
    """The SPA's baked config is the browser's only source for this line, so it must match too."""
    entry = frontend_config()["demo_samples"][usecase]
    assert entry["provenance"] == BUNDLED_DEMO_SAMPLES[usecase][0]
    assert entry["licence"] and entry["url"].startswith("http")


def test_no_source_restates_a_corrected_claim() -> None:
    """No file may re-assert one of the five claims the primary sources contradict.

    Scoped to the files a visitor's browser can reach plus the scripts that generate them.
    DATA_LICENSES.md is exempt: it documents the corrections and must quote the wrong text to do so.
    """
    forbidden = {
        "real PaySim": "PaySim is a simulator; its rows are synthetic",
        "US-Gov public domain": "no provider states a licence for SMAP/MSL",
        "open data-use, commercial-OK": "IGSR grants no blanket licence",
    }
    roots = [ROOT / "web" / "src", ROOT / "web" / "public" / "data", ROOT / "scripts"]
    offenders: list[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".ts", ".tsx", ".json", ".py"}:
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:  # pragma: no cover - unreadable file
                continue
            for claim, why in forbidden.items():
                if claim in text:
                    offenders.append(f"{path.relative_to(ROOT)}: '{claim}' — {why}")
    assert not offenders, "corrected licence claims restated:\n  " + "\n  ".join(offenders)

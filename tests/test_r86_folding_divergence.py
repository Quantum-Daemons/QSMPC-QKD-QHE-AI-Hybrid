"""The genomics page's encrypted predictor of 3D folding divergence follows its pre-registration (Round 86, batch 6; D1).

WHY THIS EXISTS (the owner's option D1 of 2026-11-26; references/genomics-3d-divergence-preregistration.md and its two
addenda, every one committed before any figure)

The predictor is taught by Akita's predictions from Gilbertson et al.'s data (Zenodo 13900918, CC BY 4.0) and shown
only if the pre-registered criteria hold. This file holds the record to the rules that choose its windows and decide
the card, and, where the card ships, the payload to the record, its licence and its genotype budget: no individual
beyond the 30 the genomics payload already serves.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RECORD = json.loads((REPO / "docs" / "measurements" / "genomics-3d-divergence-2026-11-26.json").read_text())
SITE_PATH = REPO / "web" / "public" / "data" / "genomics" / "folding_divergence.json"


def test_the_record_is_under_the_committed_preregistration_and_its_addenda() -> None:
    pre = RECORD["preregistration"]
    text = (REPO / pre["file"]).read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == pre["sha256"]
    assert "## Addendum, 2026-11-26, before any figure" in text and "## Addendum 2, 2026-11-26, before any figure" in text
    c = RECORD["criteria"]
    assert (c["r2_floor"], c["min_windows"], c["ckks_tolerance"]) == (0.5, 6, 1e-4)


def test_twelve_windows_none_overlapping_at_most_ten_variants_each() -> None:
    w = RECORD["windows"]
    assert len(w) == 12
    for i, a in enumerate(w):
        assert 1 <= a["variants"] <= 10
        for b in w[i + 1:]:
            assert a["chr"] != b["chr"] or abs(a["start"] - b["start"]) >= 1_048_576, (a, b)
    counts = RECORD["inputs"]
    assert counts["individuals"] == 2457 and counts["test"] == counts["validation"] == 369 and counts["fit"] == 1719


def test_the_card_decision_is_the_criteria_applied() -> None:
    w = RECORD["windows"]
    for x in w:
        assert x["qualifies"] == (x["test_r2"] >= 0.5 and x["test_r2"] > x["super_population_r2"]), x
    q = [i for i, x in enumerate(w) if x["qualifies"]]
    assert RECORD["qualifying_windows"] == q and RECORD["card_shown"] == (len(q) >= 6)
    shown = RECORD["card_shown"] and RECORD["ckks_python"]["criterion_met"]
    assert SITE_PATH.exists() == shown, "the page's payload exists exactly when both criteria hold"


@pytest.mark.skipif(not SITE_PATH.exists(), reason="the criteria did not hold, so the page ships no payload")
def test_the_payload_is_the_records_and_adds_no_individual() -> None:
    site = json.loads(SITE_PATH.read_text())
    q = RECORD["qualifying_windows"]
    assert [(x["chr"], x["start"]) for x in site["windows"]] == [(RECORD["windows"][i]["chr"], RECORD["windows"][i]["start"]) for i in q]
    W, b, C = np.asarray(site["W"]), np.asarray(site["b"]), np.asarray(site["carriers"])
    assert W.shape == (len(q), len(site["variants"])) and b.shape == (len(q),)
    # Block-diagonal: each window's model reads its own variants only.
    for k, row in enumerate(W):
        own = {j for j, v in enumerate(site["variants"]) if v["window"] == k}
        assert {j for j in np.flatnonzero(row)} <= own
    prs = json.loads((REPO / "web" / "public" / "data" / "genomics" / "pgs000783_breast_cancer.json").read_text())
    assert site["sample_ids"] == prs["sample_ids"], "the card serves individuals the genomics payload does not"
    assert C.shape == (30, len(site["variants"])) and set(np.unique(C)) <= {0, 1}
    assert len(site["teacher"]) == 30 and all(len(t) == len(q) for t in site["teacher"])
    for fact in ("10.5281/zenodo.13900918", "https://creativecommons.org/licenses/by/4.0/", "msae209", "Akita",
                 "Nature Methods 17:1111-1117", "doi:10.1038/nature15393"):
        assert fact in site["attribution"], fact
    assert "not a disease risk" in site["what"] and "20190312" in site["genotype_caveat"]


@pytest.mark.skipif(not SITE_PATH.exists(), reason="the criteria did not hold, so the page ships no payload")
def test_the_licence_file_counts_the_cards_genotype_cells() -> None:
    site = json.loads(SITE_PATH.read_text())
    cells = 30 * len(site["variants"])
    text = (REPO / "DATA_LICENSES.md").read_text()
    assert "folding_divergence.json" in text and f"{cells:,}" in text


def test_the_page_shows_nothing_new_when_the_criteria_fail() -> None:
    """Criterion 1 needs six windows; the run of 2026-11-26 gave four, so the page carries no card and no payload."""
    page = (REPO / "web" / "src" / "pages" / "UseCasePage.tsx").read_text()
    if RECORD["card_shown"] and RECORD["ckks_python"]["criterion_met"]:
        pytest.skip("the criteria held, so the card is the page's to show")
    assert not SITE_PATH.exists() and "FoldingDivergenceCard" not in page
    assert not (REPO / "web" / "src" / "components" / "FoldingDivergenceCard.tsx").exists()
    assert RECORD["ckks_python"]["criterion_met"], "the encrypted path itself met its criterion"

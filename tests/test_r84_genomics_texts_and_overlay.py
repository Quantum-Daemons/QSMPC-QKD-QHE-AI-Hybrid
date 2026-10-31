"""The genomics page's words and its gnomAD overlay say what is true.

WHY THIS EXISTS (Round 84, scratchpad audit83/page-finance-genomics.md findings 4, 6, 7, 9, 10 and 13,
and audit83/datasets-licences-2d.md findings 10, 12, 15 and 17)

  - The cohorts' canvas label said their counts "never leave", beside a caption saying one step sums
    both shares of every cohort; the agentic prompt asked why the quantum secure sum "is safer", a
    property the page disclaims; realData.ts called genomics "commercial-OK".
  - Three texts said no per-sample genotype is, or may be, published, while the browser payload
    serves 30 people's dosages.
  - The ceiling was not the Catalog's own maximum for the score in use; for PGS000072 it is the one
    AUROC among its three Catalog performance records (PPM002038, 0.637; read 2026-10-31).
  - The gnomAD overlay plotted variants whose coordinate lookup hit another record. It now reads each
    variant by rsID from the same Ensembl record the dosages come from, for the allele they count.

WHAT IS ASSERTED: each of those is gone, the overlay is exactly what its generator writes, and every
point it keeps is the score's effect allele, checked against the frequency the score publishes.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
OVERLAY = REPO / "web" / "public" / "data" / "genomics" / "real_2d.json"
PAYLOAD = REPO / "web" / "public" / "data" / "genomics" / "pgs000072_breast_cancer.json"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def generated() -> dict:
    return _module(REPO / "scripts" / "prep_real_data_2d.py", "prep_real_data_2d").genomics()


def test_the_overlay_is_what_its_generator_writes(generated: dict) -> None:
    assert generated, "the genomics overlay generator returned nothing"
    assert json.loads(OVERLAY.read_text()) == generated, (
        "web/public/data/genomics/real_2d.json is not the generator's output: regenerate it with "
        "scripts/prep_real_data_2d.py")


def test_the_overlay_carries_gnomads_terms_and_link() -> None:
    d = json.loads(OVERLAY.read_text())
    assert "not to attempt to reidentify participants" in d["provenance"]
    assert "re-identify participants" in d["attribution"]
    assert "https://gnomad.broadinstitute.org/" in d["attribution"], "gnomAD asks for a link back"
    assert d["licence"] == "CC0-1.0" and "publicdomain/zero/1.0" in d["licence_url"]


def test_every_plotted_point_is_the_scores_effect_allele() -> None:
    p2d = _module(REPO / "scripts" / "prep_real_data_2d.py", "prep_real_data_2d_points")
    cache = json.loads((REPO / "data" / "genomics" / "gnomad_af_pgs000072.json").read_text())
    payload = json.loads(PAYLOAD.read_text())
    d = json.loads(OVERLAY.read_text())
    r = p2d._round
    expected = []
    for key in payload["keys"]:
        rec = cache.get(key)
        if not rec or rec.get(p2d.GNOMAD_X) is None or rec.get(p2d.GNOMAD_Y) is None:
            continue
        # One record, one allele: gnomAD's European frequency agrees with 1000 Genomes' for it.
        if abs(rec[p2d.GNOMAD_Y] - rec["g1k_eur"]) <= p2d.GNOMAD_AF_TOLERANCE:
            expected.append((r(rec[p2d.GNOMAD_X]), r(rec[p2d.GNOMAD_Y])))
    assert sorted(tuple(pt[:2]) for pt in d["points"]) == sorted(expected)
    # Every variant the cache resolves is a scored one, and none the dosages leave out is plotted.
    unscored = {u["key"] for u in payload["variants_unscored"]}
    assert all(cache[k] is None for k in unscored) and not unscored & set(payload["keys"])
    assert d["y"]["label"].endswith("gnomAD nfe") and d["x"]["label"].endswith("gnomAD eas")
    assert "non-Finnish European (nfe)" in d["modifications"] and f"{len(d['points'])} of the score's" in d["modifications"]


def test_the_genotype_texts_say_what_is_served() -> None:
    from qsh.models.registry import DS_1000G

    p = json.loads(PAYLOAD.read_text())
    notes = DS_1000G.notes
    assert "no per-sample genotype is ever published" not in notes
    assert f"serves {p['n_individuals_shipped']} of the {p['n_individuals_available']:,} individuals" in notes
    weights = json.loads((REPO / "models" / "genomics_prs" / "weights.json").read_text())["note"]
    gp = _module(REPO / "scripts" / "train" / "genomics_prs.py", "genomics_prs")
    assert weights == gp.WEIGHTS_NOTE, "weights.json's note is not the trainer's"
    assert "none may be published" not in weights and "30 individuals in the browser payload" in weights


def test_the_ceiling_is_the_catalogs_current_maximum() -> None:
    from qsh.models.registry import M_PRS

    assert M_PRS.baseline == M_PRS.ceiling == 0.637
    for fact in ("PPM002038", "the only one", "own three Catalog performance records", "read 2026-10-31"):
        assert fact in M_PRS.ceiling_source, f"{fact!r} missing from the ceiling source"
    for fact in ("PPM002038", "only 162 of the original 187 SNPs", "AUC at 5 years", "WITH COVARIATES",
                 "doi:10.1038/s41467-020-19600-4"):
        assert fact in M_PRS.baseline_source, f"{fact!r} missing from the baseline source"
    for stale in ("PGS000004", "PPM001647", "PPM038141", "0.633"):
        assert stale not in M_PRS.baseline_source + M_PRS.ceiling_source, stale


def test_the_registry_says_the_full_score_runs() -> None:
    from qsh.models.registry import get

    notes = get("genomics.prs").notes
    assert "sign test" not in notes and "75th-percentile" not in notes
    assert "the full score, in both tiers" in notes and "90th percentile" in notes


def _block(src: str, start: str) -> str:
    at = src.index(start)
    return src[at: src.index("\n  },", at) + 1]


def test_the_page_text_says_what_runs() -> None:
    from qsh.usecases.genomics import HIGH_RISK_PERCENTILE

    uc = _block((REPO / "web" / "src" / "lib" / "useCases.ts").read_text(), 'id: "genomics"')
    blurb = re.search(r'blurb: "(.*?)",\n', uc).group(1)
    prompt = re.search(r'agenticPrompt: "(.*?)",\n', uc).group(1)
    for stale in ("sign test", "8 of the 313", "partial polygenic", "classical equivalent"):
        assert stale not in blurb, f"{stale!r} is still in the genomics blurb"
    assert f"{HIGH_RISK_PERCENTILE:.0f}th percentile" in blurb and "every variant in one ciphertext" in blurb
    assert "safer" not in prompt and "why" not in prompt.lower(), f"the prompt presupposes a property: {prompt}"
    flows = (REPO / "web" / "src" / "lib" / "useCaseFlows.ts").read_text()
    party = re.search(r'genomics: \{.*?\n    a: \{ name: "Cohorts", sub: "([^"]*)" \}', flows, re.S)
    assert party and "never leave" not in party.group(1) and "shares mod p" in party.group(1), party


def test_no_source_line_calls_genomics_commercial_ok() -> None:
    ts = (REPO / "web" / "src" / "lib" / "realData.ts").read_text()
    assert "backed by a commercial-OK dataset (medical, genomics)" not in ts
    assert "genomics, whose PGS000072 weights are CC BY 4.0" in ts
    cc = (REPO / "web" / "src" / "lib" / "clientCompute.ts").read_text()
    assert "Use REAL commercial-OK data" not in cc
    py = (REPO / "src" / "qsh" / "usecases" / "genomics" / "__init__.py").read_text()
    assert "exact CKKS dot-product" not in py and "top quartile = the clinical risk cut" not in py

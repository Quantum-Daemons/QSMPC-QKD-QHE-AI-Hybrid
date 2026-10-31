"""The genomics page's words and its gnomAD overlay say what is true.

WHY THIS EXISTS (Round 84, scratchpad audit83/page-finance-genomics.md findings 4, 6, 7, 9, 10 and 13,
and audit83/datasets-licences-2d.md findings 10, 12, 15 and 17)

  - The cohorts' canvas label said their counts "never leave", beside a caption saying one step sums
    both shares of every cohort; the agentic prompt asked why the quantum secure sum "is safer", a
    property the page disclaims; realData.ts called genomics "commercial-OK".
  - Three texts said no per-sample genotype is, or may be, published, while the browser payload
    serves 30 people's dosages.
  - The PGS000004 ceiling read 0.668 over 28 records; the Catalog now holds 32 AUROC records, the
    highest 0.70 (Shang et al. 2024, added 2026-08-26), European 0.5877-0.70 over 23 (read at the
    REST API on 2026-10-31).
  - The gnomAD overlay plotted three variants whose lookup hit another record (frequencies of 0.0003,
    0.068 and 0 against published 0.370, 0.776 and 0.038) and one at the other allele's frequency.

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
PAYLOAD = REPO / "web" / "public" / "data" / "genomics" / "pgs000004_prs313.json"


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
    cache = json.loads((REPO / "data" / "genomics" / "gnomad_af_prs313.json").read_text())
    published = p2d._effect_allele_frequencies(REPO / "data" / "genomics" / "pgs000004_breast_cancer.tsv")
    pts = {tuple(p[:2]) for p in json.loads(OVERLAY.read_text())["points"]}
    r = p2d._round
    # The three lookups that landed on another record, and the reference-allele variant unflipped.
    for key in ("10:38523626:A", "1:110198129:C", "2:217955896:G"):
        rec = cache[key]
        assert (r(rec["afr"]), r(rec["nfe"])) not in pts, f"{key} is still plotted from a mismatched record"
    flip = cache["9:136146597:T"]
    assert flip["effect_is_alt"] is False
    assert (r(flip["afr"]), r(flip["nfe"])) not in pts, "9:136146597:T is plotted at the other allele's frequency"
    assert (r(1 - flip["afr"]), r(1 - flip["nfe"])) in pts
    # Every kept record agrees with the frequency the score publishes for its effect allele.
    kept = 0
    for key, rec in cache.items():
        if not rec:
            continue
        afr = rec["afr"] if rec["effect_is_alt"] else 1 - rec["afr"]
        nfe = rec["nfe"] if rec["effect_is_alt"] else 1 - rec["nfe"]
        if (r(afr), r(nfe)) in pts:
            kept += 1
            assert abs(nfe - published[key]) <= p2d.GNOMAD_AF_TOLERANCE, key
    assert kept == len(pts)
    d = json.loads(OVERLAY.read_text())
    assert d["y"]["label"].endswith("gnomAD nfe") and d["x"]["label"].endswith("gnomAD afr")
    assert "non-Finnish European (nfe)" in d["modifications"] and f"{len(pts)} of the score's" in d["modifications"]


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

    assert M_PRS.ceiling == 0.70
    src = M_PRS.ceiling_source
    for fact in ("PPM038141", "all 32 AUROC records", "0.5877-0.70 over 23 entries", "PPM005171",
                 "PPM005174", "doi:10.1016/j.xhgg.2024.100302", "2026-08-26", "0.6573 (PPM000650"):
        assert fact in src, f"{fact!r} missing from the ceiling source"
    for stale in ("0.668 (PPM012913", "28 AUROC", "over 22 entries"):
        assert stale not in src
    for path in ("scripts/train/genomics_prs.py", "scripts/prep_real_data_2d.py"):
        assert "28 Catalog AUROC" not in (REPO / path).read_text(), path


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
    assert "genomics, whose terms are under review" in ts
    cc = (REPO / "web" / "src" / "lib" / "clientCompute.ts").read_text()
    assert "Use REAL commercial-OK data" not in cc
    py = (REPO / "src" / "qsh" / "usecases" / "genomics" / "__init__.py").read_text()
    assert "exact CKKS dot-product" not in py and "top quartile = the clinical risk cut" not in py

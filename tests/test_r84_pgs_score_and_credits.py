"""The genomics score is PGS000783, CC BY 3.0 IGO, and its credits travel with every file that ships it.

ROUND 86, BATCH 6 (the owner's choice of 2026-11-26). The score became PGS000783 (CC_Breast_IV), the 162 of
PGS000072's variants that reached P < 5e-8 in the BCAC meta-analysis, weighted by beta/SE (Kachuri et al., Nat Commun
11:6084, 2020, CC BY 3.0 IGO on the article page). PGS000072 is retired and its files are kept. Every assertion below
that named PGS000072 as the score now names PGS000783; the history that follows is Round 84's.

WHY THIS EXISTS (Round 84, the owner's decision of 2026-10-31; scratchpad
audit83/pgs-licence-and-alternatives.md)

The page scored PGS000004 (PRS313), whose weights appear only in the supplementary data of an article
licensed CC BY-NC-ND 4.0, so they were not cleared for commercial use. It now scores PGS000072, the
breast-cancer score of Graff et al., Nat Commun 12:970 (2021), whose version of record is CC BY 4.0
(Crossref `vor` and `tdm`; Europe PMC "cc by"; read 2026-10-31). CC BY 4.0 asks for credit, a link to
the licence and a note of changes; the PGS Catalog asks to be cited by the paper its REST `/rest/info`
names (Lambert, Wingfield et al., Nat Genet 56:1989-1994, 2024); IGSR asks for the 1000 Genomes
consortium paper (Nature 526:68-74, 2015). None of the three reached the page before.

WHAT IS ASSERTED

  - no PGS000004 file is tracked, and every genomics weight file in the tree is PGS000072's (or the
    retired PGS000001's);
  - the bundled scoring file is the Catalog's file unchanged under one added first line, which
    states the source, the checksums, the licence and the citations;
  - the weights the payload and `weights.json` carry are the scoring file's, unrounded;
  - the registry's dataset record, the payload and the overlay carry the credits, the CC BY 4.0
    link and the change note; the overlay says its gnomAD frequencies came through Ensembl;
  - the registry's metric is PGS000072's published AUROC, with the record's own caveats.

On the base commit every test here fails: the files named do not exist and the score is PGS000004.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCORING = REPO / "data" / "genomics" / "pgs000783_breast_cancer.tsv"
PAYLOAD = REPO / "web" / "public" / "data" / "genomics" / "pgs000783_breast_cancer.json"
WEIGHTS = REPO / "models" / "genomics_prs" / "weights.json"
OVERLAY = REPO / "web" / "public" / "data" / "genomics" / "real_2d.json"

#: sha256 of the Catalog's scoring file as downloaded on 2026-11-26 (decompressed), i.e. of every
#: line of the bundled file after the one this project added. PGS000072's, downloaded 2026-10-31, was
#: f3626961a7efdb38197ac733e124be1eefb368c73e1447e8da0f12e11db24dde.
CATALOG_TEXT_SHA256 = "c045c0c420cc4180818cb4c5fa865e4f04e0ba4f306f52bc5e6417cad723b6c6"
KACHURI_DOI = "10.1038/s41467-020-19600-4"
BCAC_DOI = "10.1038/nature24284"
CATALOG_DOI = "10.1038/s41588-024-01937-x"
G1K_DOI = "10.1038/nature15393"
CC_BY_3_IGO = "https://creativecommons.org/licenses/by/3.0/igo/"


def _tracked() -> list[str]:
    return subprocess.run(["git", "ls-files"], cwd=REPO, check=True, capture_output=True,
                          text=True).stdout.split()


def _scoring_rows() -> dict[str, dict[str, str]]:
    rows, header = {}, None
    for line in SCORING.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if header is None:
            header = parts
            continue
        row = dict(zip(header, parts, strict=True))
        rows[row["rsID"]] = row
    return rows


def test_no_pgs000004_file_is_tracked_and_every_weight_file_is_a_cleared_score() -> None:
    tracked = _tracked()
    assert not [f for f in tracked if "pgs000004" in f.lower() or "prs313" in f.lower()], (
        "a PGS000004 (PRS313) file is still in the tree; its weights' source is CC BY-NC-ND 4.0")
    for rel in ("data/genomics/pgs000004_breast_cancer.tsv", "web/public/data/genomics/pgs000004_prs313.json"):
        assert not (REPO / rel).exists(), f"{rel} is still on disk"
    # Every scoring file and weight vector under the genomics paths names its score.
    for rel in (f for f in tracked if f.startswith("data/genomics/pgs") and f.endswith(".tsv")):
        ids = [ln.split("=", 1)[1] for ln in (REPO / rel).read_text().splitlines() if ln.startswith("#pgs_id=")]
        assert ids in ([], ["PGS000072"], ["PGS000783"]), f"{rel} is the scoring file of {ids}"
    assert json.loads(PAYLOAD.read_text())["score_id"] == "PGS000783"
    assert json.loads(WEIGHTS.read_text())["score_id"] == "PGS000783"


def test_the_bundled_scoring_file_is_the_catalogs_unchanged_under_one_added_line() -> None:
    first, rest = SCORING.read_text().split("\n", 1)
    assert hashlib.sha256(rest.encode()).hexdigest() == CATALOG_TEXT_SHA256, (
        "the scoring file below its first line is no longer the Catalog's download")
    assert first.startswith("# PGS000783") and CATALOG_TEXT_SHA256 in first
    for fact in ("CC BY 3.0 IGO", CC_BY_3_IGO, KACHURI_DOI, BCAC_DOI, CATALOG_DOI, "beta/SE", "this line was added"):
        assert fact in first, f"{fact!r} missing from the scoring file's provenance line"
    rows = _scoring_rows()
    assert len(rows) == 162 and "#variants_number=162" in rest and "#genome_build=GRCh37" in rest
    assert "#weight_type=Inverse-variance weighting" in rest


def test_the_shipped_weights_are_the_scoring_files_unrounded() -> None:
    rows = _scoring_rows()
    payload = json.loads(PAYLOAD.read_text())
    for keys, weights, where in ((payload["keys"], payload["weights"], "payload"),
                                 (json.loads(WEIGHTS.read_text())["rsids"],
                                  json.loads(WEIGHTS.read_text())["weights"], "weights.json")):
        assert len(keys) == len(weights) == 158, where
        for k, w in zip(keys, weights, strict=True):
            assert w == float(rows[k]["effect_weight"]), f"{where}: {k} weight {w} is not the score's"
    assert payload["variants_published"] == len(rows) == 162
    assert len(payload["keys"]) + len(payload["variants_unscored"]) == 162


def test_the_registry_payload_and_overlay_carry_the_credits() -> None:
    from qsh.models.registry import BUNDLED_DEMO_SAMPLES, DS_1000G, DS_PGS_CATALOG, get

    a = DS_PGS_CATALOG.attribution
    for fact in (KACHURI_DOI, CC_BY_3_IGO, BCAC_DOI, "BCAC", CATALOG_DOI, G1K_DOI, "Changes:", "158 of the 162"):
        assert fact in a, f"{fact!r} missing from the PGS Catalog record's attribution"
    assert DS_PGS_CATALOG.licence.startswith("CC-BY-3.0-IGO") and DS_PGS_CATALOG.publishable is True
    assert DS_PGS_CATALOG.url.endswith("/score/PGS000783/")
    assert G1K_DOI in DS_1000G.attribution
    assert get("genomics.prs").hf_repo is None, "the score has no Hub repo; it ships with the page"
    payload = json.loads(PAYLOAD.read_text())
    assert payload["attribution"] == a and payload["genotypes_attribution"] == DS_1000G.attribution
    assert payload["licence_url"] == CC_BY_3_IGO and "CC-BY-3.0-IGO" in payload["licence"]
    assert "158 of PGS000783's 162" in payload["modifications"] and "left out" in payload["modifications"]
    assert payload["provenance"] == BUNDLED_DEMO_SAMPLES["genomics"][0]
    for fact in ("CC BY 3.0 IGO", "Nature 526:68-74", "158", "162", "not filled in"):
        assert fact in payload["provenance"], fact
    overlay = json.loads(OVERLAY.read_text())
    for fact in ("Ensembl's REST variation endpoint", "gnomADg:eas", "gnomADg:nfe",
                 "Ensembl does not name the gnomAD release"):
        assert fact in overlay["provenance"], f"{fact!r} missing: the overlay does not say where its frequencies came from"
    assert "Ensembl release" in overlay["attribution"] and "https://gnomad.broadinstitute.org/" in overlay["attribution"]


def test_the_metric_is_pgs000783s_published_auroc_with_its_caveats() -> None:
    from qsh.models.registry import M_PRS, get

    # PPM002070 (age, array and PCs only) is the baseline, PPM002054 (the risk factors as well) the ceiling;
    # PGS000072's PPM002038 carries PPM002054's figures for the same 162 variants, and the source says so.
    assert M_PRS.baseline == 0.626 and M_PRS.ceiling == 0.637
    for fact in ("PPM002070", "216,273", "Kachuri", "WITH COVARIATES", "age, genotyping array and PCs 1-15",
                 "AUC at 5 years", "PPM002054", "0.637", "PPM002038", "only 162 of the original 187 SNPs"):
        assert fact in M_PRS.baseline_source, fact
    assert "PPM002054" in M_PRS.ceiling_source and "read 2026-11-26" in M_PRS.ceiling_source
    spec = get("genomics.prs")
    assert "158 of PGS000783's 162 variants" in spec.architecture and "P < 5e-8" in spec.architecture
    assert "beta/SE" in spec.architecture
    for fact in ("PGS000783", "Kachuri et al.", "never filled in", "retired"):
        assert fact in spec.notes, fact

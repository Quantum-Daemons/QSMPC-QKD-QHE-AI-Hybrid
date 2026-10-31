"""The genomics score is PGS000072, CC BY 4.0, and its credits travel with every file that ships it.

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
SCORING = REPO / "data" / "genomics" / "pgs000072_breast_cancer.tsv"
PAYLOAD = REPO / "web" / "public" / "data" / "genomics" / "pgs000072_breast_cancer.json"
WEIGHTS = REPO / "models" / "genomics_prs" / "weights.json"
OVERLAY = REPO / "web" / "public" / "data" / "genomics" / "real_2d.json"

#: sha256 of the Catalog's scoring file as downloaded on 2026-10-31 (decompressed), i.e. of every
#: line of the bundled file after the one this project added.
CATALOG_TEXT_SHA256 = "f3626961a7efdb38197ac733e124be1eefb368c73e1447e8da0f12e11db24dde"
GRAFF_DOI = "10.1038/s41467-021-21288-z"
CATALOG_DOI = "10.1038/s41588-024-01937-x"
G1K_DOI = "10.1038/nature15393"
CC_BY_4 = "https://creativecommons.org/licenses/by/4.0/"


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
        assert ids in ([], ["PGS000072"]), f"{rel} is the scoring file of {ids}"
    assert json.loads(PAYLOAD.read_text())["score_id"] == "PGS000072"
    assert json.loads(WEIGHTS.read_text())["score_id"] == "PGS000072"


def test_the_bundled_scoring_file_is_the_catalogs_unchanged_under_one_added_line() -> None:
    first, rest = SCORING.read_text().split("\n", 1)
    assert hashlib.sha256(rest.encode()).hexdigest() == CATALOG_TEXT_SHA256, (
        "the scoring file below its first line is no longer the Catalog's download")
    assert first.startswith("# PGS000072") and CATALOG_TEXT_SHA256 in first
    for fact in ("CC BY 4.0", CC_BY_4, GRAFF_DOI, CATALOG_DOI, "this line was added"):
        assert fact in first, f"{fact!r} missing from the scoring file's provenance line"
    rows = _scoring_rows()
    assert len(rows) == 187 and "#variants_number=187" in rest and "#genome_build=GRCh37" in rest


def test_the_shipped_weights_are_the_scoring_files_unrounded() -> None:
    rows = _scoring_rows()
    payload = json.loads(PAYLOAD.read_text())
    for keys, weights, where in ((payload["keys"], payload["weights"], "payload"),
                                 (json.loads(WEIGHTS.read_text())["rsids"],
                                  json.loads(WEIGHTS.read_text())["weights"], "weights.json")):
        assert len(keys) == len(weights) == 183, where
        for k, w in zip(keys, weights, strict=True):
            assert w == float(rows[k]["effect_weight"]), f"{where}: {k} weight {w} is not the score's"
    assert payload["variants_published"] == len(rows) == 187
    assert len(payload["keys"]) + len(payload["variants_unscored"]) == 187


def test_the_registry_payload_and_overlay_carry_the_credits() -> None:
    from qsh.models.registry import BUNDLED_DEMO_SAMPLES, DS_1000G, DS_PGS_CATALOG, get

    a = DS_PGS_CATALOG.attribution
    for fact in (GRAFF_DOI, CC_BY_4, CATALOG_DOI, G1K_DOI, "Changes:", "183 of the 187"):
        assert fact in a, f"{fact!r} missing from the PGS Catalog record's attribution"
    assert DS_PGS_CATALOG.licence.startswith("CC-BY-4.0") and DS_PGS_CATALOG.publishable is True
    assert DS_PGS_CATALOG.url.endswith("/score/PGS000072/")
    assert G1K_DOI in DS_1000G.attribution
    assert get("genomics.prs").hf_repo is None, "the score has no Hub repo; it ships with the page"
    payload = json.loads(PAYLOAD.read_text())
    assert payload["attribution"] == a and payload["genotypes_attribution"] == DS_1000G.attribution
    assert payload["licence_url"] == CC_BY_4 and "CC-BY-4.0" in payload["licence"]
    assert "183 of PGS000072's 187" in payload["modifications"] and "left out" in payload["modifications"]
    assert payload["provenance"] == BUNDLED_DEMO_SAMPLES["genomics"][0]
    for fact in ("CC BY 4.0", "Nature 526:68-74", "183", "187", "not filled in"):
        assert fact in payload["provenance"], fact
    overlay = json.loads(OVERLAY.read_text())
    for fact in ("Ensembl's REST variation endpoint", "gnomADg:eas", "gnomADg:nfe",
                 "Ensembl does not name the gnomAD release"):
        assert fact in overlay["provenance"], f"{fact!r} missing: the overlay does not say where its frequencies came from"
    assert "Ensembl release" in overlay["attribution"] and "https://gnomad.broadinstitute.org/" in overlay["attribution"]


def test_the_metric_is_pgs000072s_published_auroc_with_its_caveats() -> None:
    from qsh.models.registry import M_PRS, get

    assert M_PRS.baseline == 0.637 and M_PRS.ceiling == 0.637
    for fact in ("PPM002038", "216,273", "Kachuri", "WITH COVARIATES", "162 of the original 187",
                 "AUC at 5 years", "PPM000192", "PPM017165"):
        assert fact in M_PRS.baseline_source, fact
    spec = get("genomics.prs")
    assert "183 of PGS000072's 187 variants" in spec.architecture and "P < 5e-8" in spec.architecture
    assert "PGS000072" in spec.notes and "Graff et al." in spec.notes and "never filled in" in spec.notes

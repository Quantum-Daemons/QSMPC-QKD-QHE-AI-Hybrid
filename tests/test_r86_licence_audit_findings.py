"""The licence audit of 2026-11-26, held: each finding checked at its source, fixed, and kept fixed here.

  - Weights fitted on a CC BY corpus carry the corpus's licence and its credit, as every such file here does (WDBC,
    OPS-SAT-AD, LIDC-IDRI). The 5G-NIDD student, the federation and the federated Hub model's `weights.json` said
    Apache-2.0 against their CC BY 4.0 corpus, and the federated model's own card said cc-by-4.0.
  - PGS000783's article licenses it CC BY 3.0 IGO "as long as you give appropriate credit to the World Health
    Organization" (nature.com, read 2026-11-26), and no credit for the score named WHO. The same paragraph reserves any
    other use of WHO's name or logo, so the credit names WHO and nothing implies its endorsement.
  - D1's record credits the corpus its labels came from (Gilbertson et al., Zenodo 13900918, CC BY 4.0).
  - Jinan-1's credit names the Zenodo record's creator, as this register credits Zenodo creators.
  - UCI Heart Disease is cited with the year its DOI's DataCite record gives, 1989.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WHO = "World Health Organization"


def _json(rel: str) -> dict:
    return json.loads((REPO / rel).read_text())


@pytest.mark.parametrize("rel", ["web/public/data/telecom_nidd/student_weights.json",
                                 "models/telecom_nidd_federated/weights.json"])
def test_weights_fitted_on_5g_nidd_carry_its_licence_and_credit(rel: str) -> None:
    from qsh.models.registry import DS_5G_NIDD

    d = _json(rel)
    assert d["licence"] == DS_5G_NIDD.licence == "CC-BY-4.0", (rel, d["licence"])
    assert d["licence_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert d["attribution"] == DS_5G_NIDD.attribution and d["modifications"], rel


def test_the_federation_file_is_under_the_corpus_licence() -> None:
    from qsh.models.registry import DS_5G_NIDD

    d = _json("web/public/data/telecom_nidd/federated.json")
    assert d["licence"] == "CC-BY-4.0" and d["attribution"]["citation"] == DS_5G_NIDD.attribution


def test_the_federated_card_and_its_weights_agree() -> None:
    card = (REPO / "models" / "telecom_nidd_federated" / "README.md").read_text()
    assert "license: cc-by-4.0" in card.split("---")[1]
    assert _json("models/telecom_nidd_federated/weights.json")["licence"] == "CC-BY-4.0"


def test_every_pgs000783_credit_names_who() -> None:
    from qsh.models.registry import DS_PGS_CATALOG

    assert WHO in DS_PGS_CATALOG.attribution and "implies no endorsement" in DS_PGS_CATALOG.attribution
    for rel in ("web/public/data/genomics/pgs000783_breast_cancer.json", "web/public/data/genomics/pgs000783_loci.json"):
        assert _json(rel)["attribution"] == DS_PGS_CATALOG.attribution, rel
    first = (REPO / "data" / "genomics" / "pgs000783_breast_cancer.tsv").read_text().split("\n", 1)[0]
    assert WHO in first, "the scoring file's first line, which the licence audit reads, does not credit WHO"
    assert WHO in (REPO / "models" / "genomics_prs" / "README.md").read_text()
    assert WHO in _json("models/genomics_prs/weights.json")["licence_audit"]["header"][0]


def test_d1s_record_credits_its_corpus() -> None:
    a = _json("docs/measurements/genomics-3d-divergence-2026-11-26.json")["attribution"]
    assert a["download"] == "https://doi.org/10.5281/zenodo.13900918" and a["licence"] == "CC-BY-4.0"
    assert "Gilbertson" in a["provider"] and "Gilbertson" in a["citation"]


def test_jinan1s_credit_names_the_zenodo_creator() -> None:
    assert "Liao, Sheng-Kai" in _json("web/public/data/satellite/jinan1_pass.json")["attribution"]


def test_uci_heart_disease_is_cited_1989() -> None:
    sites = (REPO / "web" / "public" / "data" / "medical" / "heart_disease_sites.json").read_text()
    sweep = (REPO / "web" / "public" / "data" / "medical" / "fedheart_dp_sweep.json").read_text()
    for name, text in (("heart_disease_sites.json", sites), ("fedheart_dp_sweep.json", sweep)):
        assert "Detrano, R. (1989)" in text and "Detrano, R. (1988)" not in text, name

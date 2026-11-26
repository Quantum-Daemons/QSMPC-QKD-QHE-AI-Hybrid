"""PGS000783's genotype files are PGS000072's columns for the same variants, and PGS000072 is retired, not served.

WHY THIS EXISTS (Round 86, batch 6; the owner's choice of 2026-11-26)

The genomics page scores PGS000783 (CC_Breast_IV; Kachuri et al., Nat Commun 11:6084, 2020, CC BY 3.0 IGO), whose
162 variants are PGS000072's that reached P < 5e-8 in the BCAC meta-analysis, re-weighted by beta/SE. Its dosage
files were not fetched again from Ensembl: `scripts/data/derive_g1k_dosages.py` copies PGS000072's column for each
variant, which is sound only because the fetcher's resolver reads nothing but the variant's scoring-file row and its
Ensembl record, and the two scoring files' rows agree on everything the resolver reads. This file holds that argument
to the files: the rows agree, every copied column is the source's, the declarations are the source's, and the gnomAD
cache is the source's entries. PGS000072 is retired and kept: its files stay in `data/genomics/`, and its payload left
`web/public/`, as PGS000001's did, so the page serves no genotype cell for a score it does not run.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "data" / "genomics"


def _derive():
    spec = importlib.util.spec_from_file_location("derive_g1k_dosages", REPO / "scripts" / "data" / "derive_g1k_dosages.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fetcher():
    spec = importlib.util.spec_from_file_location("fetch_g1k_dosages", REPO / "scripts" / "data" / "fetch_g1k_dosages.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_active_score_is_pgs000783_everywhere_it_is_chosen() -> None:
    from qsh.config import get_config
    from qsh.data.loaders import _PRS_FILES, ACTIVE_PRS
    from qsh.models.registry import _SPECS, DEMO_SAMPLE_FILES, DS_PGS000072, DS_PGS_CATALOG

    assert ACTIVE_PRS == get_config().data.genomics_prs_score == "breast_cancer_pgs000783"
    assert _PRS_FILES[ACTIVE_PRS] == ("pgs000783_breast_cancer.tsv", "g1k_dosages_pgs000783.tsv")
    assert DEMO_SAMPLE_FILES["genomics"] == "web/public/data/genomics/pgs000783_breast_cancer.json"
    assert DS_PGS_CATALOG.name.endswith("PGS000783") and DS_PGS000072.name.endswith("PGS000072")
    assert not [s.key for s in _SPECS if s.dataset is DS_PGS000072], "the retired score's record is attached to a model"


def test_every_variant_agrees_on_what_the_resolver_reads() -> None:
    f, d = _fetcher(), _derive()
    rows = f.read_score(GEN / "pgs000783_breast_cancer.tsv")
    src = {r["rsID"]: r for r in f.read_score(GEN / "pgs000072_breast_cancer.tsv")}
    assert len(rows) == 162 and len(src) == 187
    for r in rows:
        assert all(r[k] == src[r["rsID"]][k] for k in d.RESOLVER_INPUTS), r["rsID"]
    # The same order, a subsequence of PGS000072's: the derived columns follow the scoring file.
    order = [k for k in src if k in {r["rsID"] for r in rows}]
    assert [r["rsID"] for r in rows] == order


@pytest.mark.parametrize("stem", ["g1k_dosages_pgs000783_slice", "g1k_dosages_pgs000783"])
def test_each_copied_column_and_declaration_is_the_sources(stem: str) -> None:
    d = _derive()
    path = GEN / f"{stem}.tsv"
    src = GEN / f"{stem.replace('783', '072')}.tsv"
    if not path.exists() or not src.exists():
        pytest.skip("the 2,504-individual matrices are local only (gitignored)")
    titles, unscored, resolved, header, rows = d._read_dosages(path)
    s_titles, s_unscored, s_resolved, s_header, s_rows = d._read_dosages(src)
    keys = header[3:]
    assert len(keys) == 158 and len(unscored) == 4 and set(unscored) <= set(s_unscored)
    assert all(unscored[k] == s_unscored[k] for k in unscored)
    assert all(resolved[k] == s_resolved[k] for k in resolved) and set(resolved) == set(s_resolved) & set(keys)
    col = {k: j for j, k in enumerate(s_header)}
    assert [r[:3] for r in rows] == [r[:3] for r in s_rows], "not the same individuals in the same order"
    for i, r in enumerate(rows):
        assert r[3:] == [s_rows[i][col[k]] for k in keys], f"row {r[0]}: a column is not the source's"
    assert "Derived by scripts/data/derive_g1k_dosages.py" in " ".join(titles)
    assert titles[0] == s_titles[0].replace("pgs000072", "pgs000783"), "the slice's stated rule is not the source's"


def test_the_gnomad_cache_is_the_sources_entries() -> None:
    cache = json.loads((GEN / "gnomad_af_pgs000783.json").read_text())
    src = json.loads((GEN / "gnomad_af_pgs000072.json").read_text())
    keys = [r["rsID"] for r in _fetcher().read_score(GEN / "pgs000783_breast_cancer.tsv")]
    assert sorted(k for k in cache if k != "_source") == sorted(keys)
    assert all(cache[k] == src[k] for k in keys)
    assert {k for k in keys if cache[k] is None} == {"rs12405132", "rs34207738", "rs7297051", "rs8176636"}
    assert "derive_g1k_dosages.py" in cache["_source"]["derived"]


def test_the_retired_score_is_kept_and_not_served() -> None:
    tracked = set(subprocess.run(["git", "ls-files"], cwd=REPO, check=True, capture_output=True,
                                 text=True).stdout.split())
    for rel in ("data/genomics/pgs000072_breast_cancer.tsv", "data/genomics/g1k_dosages_pgs000072_slice.tsv",
                "data/genomics/pgs000072_prs_retired.json", "data/genomics/pgs000072_loci_retired.json",
                "data/genomics/gnomad_af_pgs000072.json"):
        assert rel in tracked, f"{rel}: the retired score's file is not kept"
    served = [p for p in tracked if p.startswith("web/public/") and "pgs000072" in p]
    assert not served, f"the page still serves the retired score's files: {served}"
    old = json.loads((GEN / "pgs000072_prs_retired.json").read_text())
    assert old["score_id"] == "PGS000072" and len(old["keys"]) == 183


def test_the_full_matrices_stay_out_of_git() -> None:
    for rel in ("data/genomics/g1k_dosages_pgs000072.tsv", "data/genomics/g1k_dosages_pgs000783.tsv"):
        assert subprocess.run(["git", "check-ignore", "-q", rel], cwd=REPO).returncode == 0, f"{rel} is not ignored"

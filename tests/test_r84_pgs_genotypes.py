"""PGS000072's 1000 Genomes genotypes: the same individuals as before, and nothing filled in.

WHY THIS EXISTS (Round 84, the owner's decision of 2026-10-31)

The genotypes for the new score's variants were read through the Ensembl REST API's genotype
endpoint (`scripts/data/fetch_g1k_dosages.py`). The fetcher it replaces wrote `.get(sample, 0)` for
any sample it had no call for, so an unmatched allele became a column of zeros: the retired PGS000001
dosage file has five constant columns that way. The new fetcher scores a variant only if every one of
the 2,504 individuals has a genotype made of Ensembl's alleles, the score's two alleles are found among
them, and exactly one reading puts the effect allele's European frequency within 0.1 of the published
one (the score's own authors' MAF criterion); it declares every other variant `#unscored` with the
reason. 183 of the 187 are scored; the 4 are one variant Ensembl does not map to GRCh38, two with no
1000 Genomes genotypes there, and one with them for 1,582 individuals.

The individuals are the ones the previous slices carried: the browser's 30 (the seed-0 draw over the
sorted 2,504-sample panel) and the CI slice's 97 (the panel's first 48 AFR and first 49 EUR in
sample-ID order).

On the base commit every test here fails: the slice, the payload and `fetch_g1k_dosages.resolve` do
not exist.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "data" / "genomics"
SLICE = GEN / "g1k_dosages_pgs000072_slice.tsv"
FULL = GEN / "g1k_dosages_pgs000072.tsv"
PAYLOAD = REPO / "web" / "public" / "data" / "genomics" / "pgs000072_breast_cancer.json"

#: The browser payload's 30 individuals, the same as the PGS000004 payload's: the seed-0 draw of 30
#: rows from the 2,504-sample panel sorted by sample ID, recovered on 2026-10-31 by replaying that
#: draw against the previous matrix and checking it reproduced the previous payload's rows exactly.
BROWSER_30 = ["HG00103", "HG00142", "HG00244", "HG00264", "HG00404", "HG01344", "HG01871", "HG01921",
              "HG02053", "HG02562", "HG03268", "HG03311", "HG03604", "HG03686", "HG03696", "HG03888",
              "HG04017", "HG04020", "HG04141", "NA11843", "NA18577", "NA18603", "NA18947", "NA19084",
              "NA19108", "NA19331", "NA19437", "NA20289", "NA20543", "NA20864"]
UNSCORED = {"rs12405132", "rs34207738", "rs7297051", "rs8176636"}


def _fetcher():
    spec = importlib.util.spec_from_file_location("fetch_g1k_dosages",
                                                  REPO / "scripts" / "data" / "fetch_g1k_dosages.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read(path: Path) -> tuple[list[str], list[list[str]]]:
    rows = [ln.split("\t") for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return rows[0], rows[1:]


def _panel() -> dict[str, tuple[str, str]]:
    return _fetcher().read_panel(GEN / "g1k_panel.tsv")


def test_the_ci_slice_is_the_same_97_individuals_by_its_stated_rule() -> None:
    header, rows = _read(SLICE)
    panel = _panel()
    ordered = sorted(panel)
    expected = ([s for s in ordered if panel[s][1] == "AFR"][:48]
                + [s for s in ordered if panel[s][1] == "EUR"][:49])
    assert [r[0] for r in rows] == expected
    assert header[:3] == ["sample", "population", "super_population"] and len(header) - 3 == 183
    assert all(v in {"0", "1", "2"} for r in rows for v in r[3:]), "a dosage is not 0, 1 or 2"
    assert len(rows) * (len(header) - 3) == 17_751 <= 30_800, "outside DATA_LICENSES section 4's cell budget"


def test_the_payload_ships_the_same_30_individuals_and_their_real_rows() -> None:
    p = json.loads(PAYLOAD.read_text())
    assert p["sample_ids"] == BROWSER_30 and p["n_individuals_shipped"] == 30
    assert p["n_individuals_available"] == 2504
    # The two committed genotype files agree wherever they hold the same person.
    header, rows = _read(SLICE)
    col = {k: j for j, k in enumerate(header)}
    by_id = {r[0]: r for r in rows}
    shared = [s for s in p["sample_ids"] if s in by_id]
    assert shared, "no individual in both files, so this consistency check checks nothing"
    for s in shared:
        row = p["dosage_samples"][p["sample_ids"].index(s)]
        assert row == [int(by_id[s][col[k]]) for k in p["keys"]], f"{s}: payload and slice disagree"


def test_the_unscored_variants_are_declared_with_reasons_and_left_out() -> None:
    from qsh.data.loaders import load_prs, unscored_variants

    keys, _, _ = load_prs("breast_cancer_pgs000072")
    declared = unscored_variants(SLICE)
    assert set(declared) == UNSCORED and all(declared.values())
    header, _ = _read(SLICE)
    assert [k for k in keys if k not in header] == [k for k in keys if k in UNSCORED]
    p = json.loads(PAYLOAD.read_text())
    assert {u["key"]: u["reason"] for u in p["variants_unscored"]} == declared
    assert not UNSCORED & set(p["keys"])


def test_the_trainer_refuses_an_undeclared_gap_and_accepts_a_declared_one(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("genomics_prs", REPO / "scripts" / "train" / "genomics_prs.py")
    gp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gp)
    body = "sample\tpopulation\tsuper_population\trsA\trsB\nS1\tGBR\tEUR\t0\t1\nS2\tYRI\tAFR\t2\t1\n"
    bare = tmp_path / "bare.tsv"
    bare.write_text(body)
    with pytest.raises(SystemExit, match="does not declare them #unscored"):
        gp.load_dosages(bare, ["rsA", "rsB", "rsC"])
    declared = tmp_path / "declared.tsv"
    declared.write_text("#unscored\trsC\tno genotype\n" + body)
    samples, keys, x = gp.load_dosages(declared, ["rsA", "rsB", "rsC"])
    assert keys == ["rsA", "rsB"] and samples == ["S1", "S2"] and x.tolist() == [[0, 1], [2, 1]]


# --------------------------------------------------------------------------------------------------
# The resolver, on synthetic Ensembl records shaped as the endpoint returns them.

def _record(alleles: str, genotypes: dict[str, str], name: str = "rsX") -> dict:
    gts = [{"sample": f"1000GENOMES:phase_3:{s}", "genotype": g} for s, g in genotypes.items()]
    gts.append({"sample": "GGVP:GGVP0001", "genotype": "Z|Z"})       # another study, ignored
    return {"name": name, "mappings": [{"assembly_name": "GRCh38", "seq_region_name": "1",
                                        "allele_string": alleles}], "genotypes": gts}


PANEL = {"E1": ("GBR", "EUR"), "E2": ("CEU", "EUR"), "A1": ("YRI", "AFR"), "A2": ("YRI", "AFR")}


def _row(ea: str, oa: str, af: float) -> dict[str, str]:
    return {"rsID": "rsX", "effect_allele": ea, "other_allele": oa, "allelefrequency_effect": str(af)}


def test_a_missing_genotype_rejects_the_variant_instead_of_defaulting_to_zero() -> None:
    f = _fetcher()
    rec = _record("G/A", {"E1": "G|A", "E2": "G|G", "A1": "A|A"})        # A2 has no call
    effect, counts, why = f.resolve(_row("G", "A", 0.75), rec, PANEL)
    assert effect is None and counts is None and "3 of the 4" in why


def test_the_resolver_reads_strand_palindromes_and_indels_by_the_rules() -> None:
    f = _fetcher()
    # The other strand: the score writes C/T, Ensembl's GRCh38 alleles are G/A.
    e, c, how = f.resolve(_row("C", "T", 0.75), _record("G/A", {"E1": "G|A", "E2": "G|G", "A1": "A|A",
                                                                 "A2": "G|A"}), PANEL)
    assert e == "G" and c == {"E1": 1, "E2": 2, "A1": 0, "A2": 1} and "other strand" in how
    # A/T: both strands read the same letters, so the European frequency decides (T at 0.75 here).
    e, c, how = f.resolve(_row("A", "T", 0.25), _record("A/T", {"E1": "T|A", "E2": "T|T", "A1": "A|A",
                                                                 "A2": "A|T"}), PANEL)
    assert e == "A" and c["E2"] == 0 and "as written" in how
    e, c, how = f.resolve(_row("A", "T", 0.75), _record("A/T", {"E1": "T|A", "E2": "T|T", "A1": "A|A",
                                                                 "A2": "A|T"}), PANEL)
    assert e == "T" and c["E2"] == 2 and "other strand" in how
    # An anchored insertion, written expanded across its repeat by Ensembl: A/AG is GGGG/GGGGG.
    e, c, how = f.resolve(_row("A", "AG", 0.5), _record("GGGG/GGGGG", {"E1": "GGGG|GGGGG",
                          "E2": "GGGG|GGGGG", "A1": "GGGGG|GGGGG", "A2": "GGGG|GGGG"}), PANEL)
    assert e == "GGGG" and c == {"E1": 1, "E2": 1, "A1": 0, "A2": 2}
    # A complement reading the genotypes never carry is not a reading: rs132390's case.
    e, c, how = f.resolve(_row("C", "T", 0.25), _record("C/A/G/T", {"E1": "C|T", "E2": "T|T",
                          "A1": "T|T", "A2": "C|C"}), PANEL)
    assert e == "C" and "as written" in how


def test_the_resolver_rejects_a_wrong_frequency_and_a_constant_column() -> None:
    f = _fetcher()
    rec = _record("G/A", {"E1": "G|G", "E2": "G|G", "A1": "G|A", "A2": "A|A"})
    effect, _, why = f.resolve(_row("G", "A", 0.2), rec, PANEL)       # EUR frequency 1.0
    assert effect is None and "within 0.1" in why
    rec = _record("G/A", {"E1": "G|A", "E2": "G|A", "A1": "G|A", "A2": "G|A"})
    effect, _, why = f.resolve(_row("G", "A", 0.5), rec, PANEL)
    assert effect is None and "constant column" in why
    assert f.AF_TOLERANCE == 0.1


@pytest.mark.skipif(not FULL.exists(), reason="the 2,504-individual matrix is local only (gitignored)")
def test_the_full_matrix_where_present() -> None:
    header, rows = _read(FULL)
    panel = _panel()
    assert [r[0] for r in rows] == sorted(panel) and len(rows) == 2504
    x = np.asarray([[int(v) for v in r[3:]] for r in rows])
    assert x.shape == (2504, 183) and all(len(set(x[:, j])) > 1 for j in range(x.shape[1])), "a constant column"
    # The cohorts the Python secure sum pools on a maintainer's run, as test_r84_genomics_full_score
    # pins them: the first scored variant, the first 400 people, four contiguous cohorts.
    assert header[3] == "rs616488"
    assert [int(s.sum()) for s in np.array_split(x[:400, 0], 4)] == [141, 131, 138, 118]
    rng = np.random.default_rng(0)
    take = np.sort(rng.choice(2504, size=30, replace=False))
    assert [rows[i][0] for i in take] == BROWSER_30

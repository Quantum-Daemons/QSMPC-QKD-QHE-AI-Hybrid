"""No genomics text denies shipping the genotypes it ships, or claims a licence nothing grants.

WHY THIS EXISTS (Round 84, the owner's decision of 2026-10-31; scratchpad
audit83/pgs-licence-and-alternatives.md findings 3-10 and 15)

The browser payload serves 30 people's 1000 Genomes dosages and three more tracked files hold slices,
while six texts said no per-sample genotype is or may be published, or that genotypes "stay local"
(the registry, the weights' note, the model card, the fetchers' headers, DATA_LICENSES.md and the
overlay generator). `loaders.py` called 1000 Genomes "CC-BY" and "commercial-OK" and offered openSNP,
which its maintainers deleted in 2025; IGSR attaches no licence instrument to the genotypes. The page
said the weights are "not published" while serving them. And `make data` listed a gitignored matrix
as a committed sample, so it failed on every clone.

WHAT IS ASSERTED: each of those is gone and the current fact is stated in its place; the flowchart
nodes name the score that runs; the overlay's consistency tolerance is the dosage resolver's.

On the base commit every test here fails.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: The six texts, and what each now has to say.
GENOTYPE_TEXTS = [
    "src/qsh/models/registry.py", "models/genomics_prs/weights.json", "models/genomics_prs/README.md",
    "scripts/data/fetch_g1k_dosages.py", "scripts/data/build_prs_browser_payload.py",
    "scripts/fetch_gnomad_af.py", "DATA_LICENSES.md", "scripts/prep_real_data_2d.py",
]
DENIALS = [
    r"no per-sample genotype is (ever )?(published|included)",
    r"none may be published",
    r"[Gg]enotypes stay local",
    r"only aggregate statistics and the (published )?(score )?weights are ever (redistributed|published)",
    r"which section 4 does not permit",
    r"so the training corpus\s+cannot ship",
    r"a CC0 reference population stands in",
]


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_text_denies_shipping_the_genotypes_it_ships() -> None:
    found = [f"{rel}: /{pat}/" for rel in GENOTYPE_TEXTS for pat in DENIALS
             if re.search(pat, (REPO / rel).read_text())]
    assert not found, "a text still denies shipping per-sample genotypes:\n  " + "\n  ".join(found)
    from qsh.models.registry import DS_1000G

    assert "Per-sample genotypes ARE redistributed" in DS_1000G.notes
    assert "serves 30 of the 2,504 individuals" in DS_1000G.notes


def test_the_loader_claims_no_licence_nothing_grants() -> None:
    src = (REPO / "src" / "qsh" / "data" / "loaders.py").read_text()
    for wrong in ("1000 Genomes (CC-BY)", "OpenSNP (CC0)", "no-auth, commercial-OK", "commercial-OK",
                  "exact CKKS dot-product"):
        assert wrong not in src, f"loaders.py still says {wrong!r}"
    assert "no licence instrument" in src and "one approximate CKKS dot product" in src
    # Affirmative claims only: "not uniformly commercial-OK: the genomics payload's ..." is accurate.
    for rel in ("web/src/lib/realData.ts", "web/src/lib/clientCompute.ts"):
        text = (REPO / rel).read_text()
        for claim in (r"commercial-OK dataset \([^)]*genomics", r"REAL commercial-OK data",
                      r"genomics[^.\n]{0,60}\bis commercial-OK"):
            assert not re.search(claim, text), f"{rel}: /{claim}/"
    ts = (REPO / "web" / "src" / "lib" / "realData.ts").read_text()
    assert "PGS000072 effect weights are CC BY 4.0" in ts and "carry no licence instrument" in ts


def test_the_page_says_the_weights_are_served_not_withheld() -> None:
    panel = (REPO / "web" / "src" / "components" / "ModelPanel.tsx").read_text()
    assert re.search(r'usecase === "genomics" && !spec\.hf_repo_id\s*\?\s*"[^"]*not on the Hugging Face Hub: '
                     r'the published effect weights are served with this page"', panel)
    cards = (REPO / "web" / "src" / "pages" / "UseCases.tsx").read_text()
    assert re.search(r'u\.id === "genomics"[\s\S]{0,300}the score\'s published effect weights are served with '
                     r'this page, under the licence above', cards)
    assert "PGS000004" not in cards


#: The genomics flowchart's nodes, and what they said before. Module level so the technical report's
#: own tests (outside this repository since Round 86) hold its copy of the flowchart to the same.
GENOMICS_NODES = ("Data: 30 real 1000 Genomes individuals, PGS000072 weights over 183 of its 187 variants",
                  "Private inference: the full 183-variant score under CKKS (node-seal), then the "
                  "90th-percentile cut by a garbled-circuit argmax",
                  "in a field above any total they can reach")
STALE_GENOMICS_NODES = ("PGS000004 weights over 313", "sign test on 8 of the 313", "QSMS over a GHZ state")


def test_the_flowchart_nodes_name_the_score_that_runs() -> None:
    text = (REPO / "README.md").read_text()
    for node in GENOMICS_NODES:
        assert node in text, f"README.md: {node!r}"
    for stale in STALE_GENOMICS_NODES:
        assert stale not in text, f"README.md: {stale!r}"


def test_make_data_lists_only_committed_genomics_files() -> None:
    """`make data` fails when a listed sample is missing, so a gitignored file in the list fails it on
    every clone. `git check-ignore` matches a path whether or not the file exists."""
    from qsh.data import download

    listed = [(label, str(path.relative_to(REPO))) for label, path in download._bundled()]
    ignored = [f"{rel} ({label})" for label, rel in listed
               if subprocess.run(["git", "check-ignore", "-q", rel], cwd=REPO).returncode == 0]
    assert not ignored, "`make data` lists gitignored files as committed samples: " + ", ".join(ignored)
    assert any("pgs000072" in rel for _, rel in listed)


def test_the_overlay_tolerance_is_the_dosage_resolvers() -> None:
    p2d = _module(REPO / "scripts" / "prep_real_data_2d.py", "prep_real_data_2d_tol")
    fetch = _module(REPO / "scripts" / "data" / "fetch_g1k_dosages.py", "fetch_g1k_dosages_tol")
    assert p2d.GNOMAD_AF_TOLERANCE == fetch.AF_TOLERANCE == 0.1
    assert (p2d.GNOMAD_X, p2d.GNOMAD_Y) == ("eas", "nfe")
    # The authors' own criterion is the cited source for 0.1.
    src = (REPO / "scripts" / "data" / "fetch_g1k_dosages.py").read_text()
    assert "differed by >0.10" in src and "Graff et al., Nat Commun 12:970, 2021" in src

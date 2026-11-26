#!/usr/bin/env python3
"""Derive a score's 1000 Genomes dosage files and gnomAD cache from a score whose variants it is a subset of.

Maintainer-only provenance script, like ``scripts/data/fetch_g1k_dosages.py``, whose output it reuses
(Round 86, batch 6: PGS000783, the owner's choice of 2026-11-26). PGS000783 is PGS000072's variants
re-weighted: the Catalog's method is "162 variants from Graff et al (PGS000072) with inverse variant
weights", and its scoring file lists 162 of PGS000072's 187 rows with the same rsID, position, alleles
and published frequency, in the same order. Everything the fetcher's resolver reads for a variant is
that row and that variant's Ensembl record, so on the records PGS000072's files were built from it
resolves each shared variant to the same column, the same ``#unscored`` reason or the same
``#resolved`` reading. This script copies those, after checking that every input is equal, instead of
fetching the 162 records again; a row that differs in any of them is refused, never copied.

It writes, for ``--score``:

  * the full matrix (``data/genomics/<dosage file>``, gitignored: DATA_LICENSES.md section 4
    justifies committing a slice, not the matrix), from the source score's full matrix;
  * the committed CI slice beside it (``<stem>_slice.tsv``), the same individuals as the source's
    slice, which the fetcher chose by ``SLICE_GROUPS``;
  * the gnomAD frequency cache (``data/genomics/gnomad_af_<stem>.json``) that
    ``scripts/prep_real_data_2d.py`` reads, the source cache's entries for the shared variants.

Usage:
    .venv/bin/python scripts/data/derive_g1k_dosages.py --from-score breast_cancer_pgs000072
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GENOMICS = REPO / "data" / "genomics"

#: The scoring-file fields `fetch_g1k_dosages.resolve` reads for a variant. Equal fields and one
#: Ensembl record give one resolution.
RESOLVER_INPUTS = ("rsID", "chr_name", "chr_position", "effect_allele", "other_allele", "allelefrequency_effect")


def _fetcher():
    spec = importlib.util.spec_from_file_location("fetch_g1k_dosages", REPO / "scripts" / "data" / "fetch_g1k_dosages.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_dosages(path: Path) -> tuple[list[str], dict[str, str], dict[str, str], list[str], list[list[str]]]:
    """(title lines, unscored, resolved, header, rows) of a dosage file the fetcher wrote."""
    titles: list[str] = []
    unscored: dict[str, str] = {}
    resolved: dict[str, str] = {}
    header: list[str] | None = None
    rows: list[list[str]] = []
    for line in path.read_text().splitlines():
        if line.startswith("#unscored\t"):
            _, k, why = line.split("\t", 2)
            unscored[k] = why
        elif line.startswith("#resolved\t"):
            _, k, how = line.split("\t", 2)
            resolved[k] = how
        elif line.startswith("#"):
            titles.append(line)
        elif line.strip():
            parts = line.split("\t")
            if header is None:
                header = parts
            else:
                rows.append(parts)
    assert header and header[:3] == ["sample", "population", "super_population"], f"{path.name}: unexpected header"
    return titles, unscored, resolved, header, rows


def main() -> int:
    sys.path.insert(0, str(REPO / "src"))
    from qsh.data.loaders import _PRS_FILES, ACTIVE_PRS  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", choices=sorted(_PRS_FILES), default=ACTIVE_PRS)
    ap.add_argument("--from-score", required=True, choices=sorted(_PRS_FILES),
                    help="the score whose fetched files hold every variant of --score")
    a = ap.parse_args()
    if a.score == a.from_score:
        raise SystemExit("--score and --from-score are the same score")

    f = _fetcher()
    score_file, dosage_file = _PRS_FILES[a.score]
    src_score_file, src_dosage_file = _PRS_FILES[a.from_score]
    rows = f.read_score(GENOMICS / score_file)
    src_rows = {r["rsID"]: r for r in f.read_score(GENOMICS / src_score_file)}
    problems = []
    for r in rows:
        s = src_rows.get(r["rsID"])
        if s is None:
            problems.append(f"{r['rsID']} is not in {src_score_file}")
            continue
        for k in RESOLVER_INPUTS:
            if r.get(k) != s.get(k):
                problems.append(f"{r['rsID']}: {k} {r.get(k)!r} here, {s.get(k)!r} in {src_score_file}")
    if problems:
        raise SystemExit(f"{score_file} is not a subset of {src_score_file} with equal resolver inputs:\n  "
                         + "\n  ".join(problems[:20]))
    keys_in_order = [r["rsID"] for r in rows]

    full_src = GENOMICS / src_dosage_file
    slice_src = GENOMICS / f"{Path(src_dosage_file).stem}_slice.tsv"
    out_full = GENOMICS / dosage_file
    out_slice = GENOMICS / f"{Path(dosage_file).stem}_slice.tsv"
    panel_n = len(f.read_panel(GENOMICS / "g1k_panel.tsv"))

    def derive(src: Path, out: Path, title: str) -> tuple[int, int]:
        titles, unscored, resolved, header, data = _read_dosages(src)
        col = {name: j for j, name in enumerate(header)}
        for k in keys_in_order:
            if (k in col) == (k in unscored):
                raise SystemExit(f"{src.name}: {k} must have exactly one of a column and an #unscored line")
        keys = [k for k in keys_in_order if k in col]
        with out.open("w") as fh:
            fh.write(f"# {title}\n")
            fh.write(f"# Real 1000 Genomes phase 3 effect-allele dosages (0/1/2) for the {score_file} variants, keyed "
                     f"by the scoring file's rsID; {len(data)} of {panel_n} individuals x {len(keys)} of its "
                     f"{len(rows)} variants.\n")
            fh.write("# Source: Ensembl REST /variation/human/{rsid}?genotypes=1;pops=1 (GRCh38), 1000GENOMES:phase_3 "
                     "samples only; populations from data/genomics/g1k_panel.tsv. Derived by "
                     f"scripts/data/derive_g1k_dosages.py from {src.name}, which scripts/data/fetch_g1k_dosages.py "
                     f"wrote: every variant of {score_file} is a row of {src_score_file} with the same rsID, position, "
                     "alleles and published frequency, so its column, or its reason for having none, is copied "
                     "unchanged. No genotype is filled in.\n")
            fh.write("# Cite: The 1000 Genomes Project Consortium, A global reference for human genetic variation, "
                     "Nature 526:68-74 (2015), doi:10.1038/nature15393. IGSR attaches no licence instrument to these "
                     "genotypes; see DATA_LICENSES.md section 4.\n")
            for k in keys_in_order:
                if k in unscored:
                    fh.write(f"#unscored\t{k}\t{unscored[k]}\n")
            for k in keys:
                if k in resolved:
                    fh.write(f"#resolved\t{k}\t{resolved[k]}\n")
            fh.write("\t".join(["sample", "population", "super_population", *keys]) + "\n")
            for r in data:
                fh.write("\t".join([*r[:3], *(r[col[k]] for k in keys)]) + "\n")
        print(f"wrote {out.relative_to(REPO)}: {len(data)} individuals x {len(keys)} of {len(rows)} variants "
              f"({len(rows) - len(keys)} #unscored)")
        return len(data), len(keys)

    n_full, m = derive(full_src, out_full, f"Full panel for {score_file}: local and gitignored (DATA_LICENSES.md "
                                           "section 4).")
    slice_title = _read_dosages(slice_src)[0][0][2:].strip()   # the fetcher's own first line for its slice
    n_slice, m_slice = derive(slice_src, out_slice, slice_title)
    assert m_slice == m and n_full == panel_n, (n_full, m, n_slice, m_slice)

    stem = Path(score_file).stem.replace("_breast_cancer", "")
    src_stem = Path(src_score_file).stem.replace("_breast_cancer", "")
    cache = json.loads((GENOMICS / f"gnomad_af_{src_stem}.json").read_text())
    # Every variant of the score, as the source cache keys it: a frequency record for each scored one and the
    # fetcher's null for each it left unscored.
    _, unscored, _, header, _ = _read_dosages(out_full)
    missing = [k for k in keys_in_order if k not in cache]
    if missing:
        raise SystemExit(f"gnomad_af_{src_stem}.json has no entry for {missing[:5]}")
    wrong = [k for k in keys_in_order if (cache[k] is None) != (k in unscored)]
    if wrong:
        raise SystemExit(f"gnomad_af_{src_stem}.json's null entries disagree with the #unscored lines at {wrong[:5]}")
    source = dict(cache["_source"])
    source["derived"] = (f"the entries of gnomad_af_{src_stem}.json for the {len(keys_in_order)} variants of "
                         f"{score_file}, copied unchanged by scripts/data/derive_g1k_dosages.py (the same records "
                         "and the same resolved alleles; null for the variants the dosages leave unscored)")
    out = {"_source": source, **{k: cache[k] for k in keys_in_order}}
    path = GENOMICS / f"gnomad_af_{stem}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path.relative_to(REPO)}: {len(keys_in_order)} variants, {len(unscored)} of them null")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

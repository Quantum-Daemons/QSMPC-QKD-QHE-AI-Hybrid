#!/usr/bin/env python
"""Fetch 1000 Genomes phase-3 effect-allele dosages for a PGS score keyed by POSITION, not rsID.

WHY A SECOND FETCHER EXISTS. ``fetch_g1k_dosages.py`` keys on rsID and calls Ensembl's
``/variation/human/{rsid}?genotypes=1``. That works for PGS000001, whose bundled scoring file carries
an rsID column. It cannot work for PGS000004 / PRS313, the current clinical breast-cancer instrument:
**the Catalog supplies no rsIDs for it at all.** Every one of its 313 harmonised rows has an empty
``hm_rsID`` with ``hm_source=Author-reported``, meaning the Catalog could not harmonise the
author-reported coordinates to variant identifiers.

Reverse-mapping the coordinates was measured before this file was written, on a 40-variant sample
against the Ensembl GRCh37 REST endpoint: 70% of positions carry exactly one rsID, 28% carry several,
and allele matching resolves only about half of those. Roughly 15% - some 46 of the 313 - would have
had no unique identifier. Choosing one arbitrarily would fetch a DIFFERENT variant's genotypes and
compute the score on the wrong SNPs while every number on the page still looked reasonable, and
dropping them would leave a 267-variant score that is not PRS313 and cannot borrow its published
AUROC.

Keying on position removes the question rather than answering it. The scoring file gives
``chr_name``, ``chr_position``, ``effect_allele`` and ``other_allele``; a tabix slice of the phase-3
VCF at that coordinate gives ``REF`` and ``ALT``; requiring ``{effect, other} <= {ref, *alts}`` is an
exact match against the genotypes actually being read, not a lookup through a third identifier.

Measured, in two stages, because the earlier revision of this paragraph quoted the pilot's numbers as
if they were the result and disagreed with itself inside one sentence ("8 of 8 ... three of those same
six positions"):

* pilot, 8 positions chosen to include an indel (``CAAA``/``C``): **8 of 8** matched, and the rsID
  route was ambiguous at 3 of those 8.
* full run, all 313 positions: **313 of 313** matched, with a 40-position sample audited by hand
  against the scoring file. That is the figure to quote.

The cost is a dependency: ``pysam`` (MIT, actively maintained, htslib) for tabix over HTTP range
requests. It is a maintainer-only provenance dependency - nothing at runtime imports it, exactly as
nothing at runtime imports this script.

Source: 1000 Genomes phase 3 (IGSR). IGSR grants NO blanket licence - "restrictions on different
pieces of data within IGSR and rights claimed on pieces of data vary". Genotypes stay local; only
aggregate statistics and the published score weights are ever redistributed.

Usage:
    python scripts/data/fetch_g1k_dosages_bypos.py \
        --score data/genomics/pgs000004_breast_cancer.tsv \
        --out data/genomics/g1k_dosages_prs313.tsv
"""
from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

PANEL_URL = ("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
             "integrated_call_samples_v3.20130502.ALL.panel")
#: Per-chromosome phase-3 genotype VCFs, GRCh37, with .tbi indexes alongside.
VCF = ("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
       "ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz")
#: chrX has a different filename and hemizygous genotypes, so it is excluded rather than mishandled.
#: PRS313 is autosomal, so this drops nothing - asserted below rather than assumed.
AUTOSOMES = {str(i) for i in range(1, 23)}


def read_panel() -> dict[str, tuple[str, str]]:
    """sample -> (population, super population)."""
    with urllib.request.urlopen(PANEL_URL, timeout=60) as r:
        text = r.read().decode()
    out: dict[str, tuple[str, str]] = {}
    for line in text.splitlines()[1:]:
        f = line.split()
        if len(f) >= 3:
            out[f[0]] = (f[1], f[2])
    return out


def read_score(path: Path) -> list[dict[str, str]]:
    """Rows of a PGS Catalog scoring file, keyed by position.

    Accepts the harmonised or the author-reported layout: both carry `chr_name`, `chr_position`,
    `effect_allele` and `other_allele`, which is all this fetcher needs.
    """
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if header is None:
            header = parts
            continue
        if len(parts) != len(header):
            raise SystemExit(
                f"{path}: row has {len(parts)} fields, header has {len(header)}. Refusing to guess "
                f"which column is missing - a mispaired row scores the wrong allele.\n  {line[:120]}")
        rows.append(dict(zip(header, parts, strict=True)))
    assert rows, f"no variant rows parsed from {path}"
    for r in rows:
        for col in ("chr_name", "chr_position", "effect_allele", "other_allele"):
            assert col in r, f"{path} lacks the '{col}' column this fetcher keys on"
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="stop after N variants (smoke test)")
    a = ap.parse_args()

    try:
        import pysam
    except ImportError:
        print("pysam is required for position-keyed fetching: pip install pysam", file=sys.stderr)
        return 2

    variants = read_score(a.score)
    if a.limit:
        variants = variants[: a.limit]
    non_auto = sorted({v["chr_name"] for v in variants} - AUTOSOMES)
    assert not non_auto, f"non-autosomal contigs present, which this fetcher does not handle: {non_auto}"
    print(f"score: {len(variants)} variants across {len({v['chr_name'] for v in variants})} chromosomes")

    panel = read_panel()
    print(f"panel: {len(panel)} samples")

    by_chrom: dict[str, list[dict[str, str]]] = {}
    for v in variants:
        by_chrom.setdefault(v["chr_name"], []).append(v)

    # key -> {sample: dosage}. The key is the coordinate and the effect allele, so it is unambiguous
    # without an rsID and stays readable in the header row.
    columns: dict[str, dict[str, int]] = {}
    matched = unmatched = 0
    for chrom in sorted(by_chrom, key=lambda c: int(c)):
        vf = pysam.VariantFile(VCF.format(chrom=chrom))
        samples = list(vf.header.samples)
        for v in by_chrom[chrom]:
            pos = int(v["chr_position"])
            ea, oa = v["effect_allele"], v["other_allele"]
            rec = None
            for cand in vf.fetch(chrom, pos - 1, pos):
                if {ea, oa} <= {cand.ref, *(cand.alts or ())}:
                    rec = cand
                    break
            if rec is None:
                unmatched += 1
                print(f"  {chrom}:{pos} {oa}/{ea}: NO ALLELE MATCH in the phase-3 VCF", file=sys.stderr)
                continue
            # Dosage is the count of the EFFECT allele, which may be REF or ALT at this record.
            alleles = [rec.ref, *(rec.alts or ())]
            try:
                eff_idx = alleles.index(ea)
            except ValueError:      # cannot happen given the match above; kept as an assertion
                unmatched += 1
                continue
            col: dict[str, int] = {}
            for s in samples:
                gt = rec.samples[s].get("GT")
                if gt is None:
                    continue
                col[s] = sum(1 for g in gt if g == eff_idx)
            key = f"{chrom}:{pos}:{ea}"
            columns[key] = col
            matched += 1
            if matched % 25 == 0:
                print(f"  [{matched}/{len(variants)}] {key}: {len(col)} genotypes")
        vf.close()

    print(f"matched {matched} variants, {unmatched} unmatched")
    assert matched, "no variants matched - refusing to write an empty dosage table"

    keys = list(columns)
    shared = sorted(set(panel) & set.intersection(*(set(columns[k]) for k in keys)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh:
        fh.write(
            f"# Real 1000 Genomes phase-3 genotype dosages (effect-allele counts 0/1/2) for the "
            f"{a.score.stem} variants, keyed by CHROM:POS:EFFECT_ALLELE.\n"
            "# Source: 1000 Genomes phase 3 (IGSR) genotype VCFs, GRCh37, read by tabix range "
            "request and matched on REF/ALT rather than through an rsID -- the score publishes no "
            "rsIDs, and reverse-mapping its coordinates left about 15% ambiguous.\n"
            "# IGSR grants NO BLANKET LICENCE: rights claimed on pieces of data vary. These genotypes "
            "are local provenance data. Only aggregate statistics and the published weights are ever "
            "redistributed.\n")
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["sample", "population", "super_population", *keys])
        for s in shared:
            pop, sup = panel[s]
            w.writerow([s, pop, sup, *(columns[k].get(s, "") for k in keys)])
    print(f"wrote {a.out}: {len(shared)} samples x {len(keys)} variants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Fetch real 1000 Genomes phase-3 effect-allele dosages for a PGS Catalog score.

Maintainer-only provenance script, in the same spirit as ``scripts/prep_real_kaggle_data.py``: it is
not runtime code and nothing in the pipeline calls it. It exists so the bundled
``data/genomics/g1k_dosages.tsv`` can be regenerated and audited rather than taken on trust.

The bundled file covers 400 samples spanning EUR, EAS and AMR. That is enough to show that a
polygenic score's distribution shifts with genetic ancestry, but it omits AFR — the group where the
published portability gap is largest and therefore the one most worth showing. This script pulls
every sample in the phase-3 panel (2,504 across all five super-populations) so the stratification is
complete.

Source: Ensembl REST over IGSR / 1000 Genomes phase 3. Open access, no key. Note that IGSR is *not*
public domain: "restrictions on different pieces of data within IGSR and rights claimed on pieces of
data vary". Genotypes stay local; only aggregate statistics and the score weights are ever published.

Usage:
    python scripts/data/fetch_g1k_dosages.py --out data/genomics/g1k_dosages_full.tsv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ENSEMBL = "https://rest.ensembl.org"
PANEL_URL = ("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
             "integrated_call_samples_v3.20130502.ALL.panel")

#: Ensembl asks for at most 15 requests/second; one every 0.2 s is well inside that and still
#: finishes 77 variants in under a minute.
REQUEST_INTERVAL = 0.2


def _get(url: str, retries: int = 4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Content-Type": "application/json",
                                                       "User-Agent": "qsh-provenance/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                wait = int(e.headers.get("Retry-After", 2 ** attempt))
                print(f"    rate-limited, sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def load_prs_snps(path: Path) -> list[tuple[str, str]]:
    """``(rsid, effect_allele)`` pairs from a bundled PGS Catalog scoring file."""
    out = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("rsID"):
            continue
        parts = line.split("\t")
        out.append((parts[0], parts[1]))
    return out


def dosages_for(rsid: str, effect_allele: str) -> dict[str, int]:
    """Effect-allele count (0/1/2) per 1000G sample for one variant."""
    data = _get(f"{ENSEMBL}/variation/human/{rsid}?genotypes=1;content-type=application/json")
    out: dict[str, int] = {}
    for g in data.get("genotypes", []):
        sample = g.get("sample", "")
        gt = g.get("genotype", "")
        if not sample or "|" not in gt and "/" not in gt:
            continue
        # Ensembl prefixes 1000G sample ids with the study, e.g. "1000GENOMES:phase_3:HG00096".
        sample = sample.rsplit(":", 1)[-1]
        alleles = gt.replace("/", "|").split("|")
        out[sample] = sum(1 for a in alleles if a == effect_allele)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prs", type=Path, default=Path("data/genomics/pgs000001_breast_cancer.tsv"))
    ap.add_argument("--out", type=Path, default=Path("data/genomics/g1k_dosages_full.tsv"))
    ap.add_argument("--panel-out", type=Path, default=Path("data/genomics/g1k_panel.tsv"))
    args = ap.parse_args()

    snps = load_prs_snps(args.prs)
    print(f"{len(snps)} SNPs from {args.prs}")

    print("fetching the phase-3 sample panel ...")
    panel_txt = urllib.request.urlopen(PANEL_URL, timeout=120).read().decode()
    panel: dict[str, tuple[str, str]] = {}
    for line in panel_txt.splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) >= 3 and cols[0]:
            panel[cols[0]] = (cols[1], cols[2])
    print(f"  {len(panel)} samples across {len({s for _, s in panel.values()})} super-populations")

    columns: dict[str, dict[str, int]] = {}
    for i, (rsid, allele) in enumerate(snps, 1):
        try:
            columns[rsid] = dosages_for(rsid, allele)
            print(f"  [{i:3d}/{len(snps)}] {rsid} ({allele}): {len(columns[rsid])} genotypes")
        except Exception as exc:
            # A variant missing from Ensembl is a real gap in coverage, not something to paper over
            # with an imputed value: record it and let the caller decide.
            print(f"  [{i:3d}/{len(snps)}] {rsid}: FAILED {type(exc).__name__} {exc}", file=sys.stderr)
            columns[rsid] = {}
        time.sleep(REQUEST_INTERVAL)

    usable = [rs for rs, col in columns.items() if col]
    samples = sorted({s for rs in usable for s in columns[rs]} & set(panel))
    print(f"\n{len(usable)}/{len(snps)} SNPs resolved, {len(samples)} samples with panel metadata")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write("# Real 1000 Genomes phase-3 genotype dosages (effect-allele counts 0/1/2).\n")
        f.write(f"# Source: Ensembl REST over IGSR / 1000 Genomes phase 3. Fetched by "
                f"scripts/data/fetch_g1k_dosages.py. SNPs resolved: {len(usable)}/{len(snps)}.\n")
        f.write("# IGSR is open access but NOT public domain; see "
                "https://www.internationalgenome.org/IGSR_disclaimer\n")
        f.write("sample\t" + "\t".join(usable) + "\n")
        for s in samples:
            f.write(s + "\t" + "\t".join(str(columns[rs].get(s, 0)) for rs in usable) + "\n")

    with args.panel_out.open("w") as f:
        f.write("# 1000 Genomes phase-3 sample -> population / super-population.\n")
        f.write(f"# Source: {PANEL_URL}\n")
        f.write("sample\tpop\tsuper_pop\n")
        for s in samples:
            pop, sup = panel[s]
            f.write(f"{s}\t{pop}\t{sup}\n")

    print(f"wrote {args.out} and {args.panel_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

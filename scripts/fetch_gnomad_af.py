#!/usr/bin/env python3
"""Cache gnomAD v4 per-ancestry allele frequencies for the PGS000004 (PRS313) variants.

WHY THIS DATA AND NOT THE TRAINING CORPUS. The genomics page scores real 1000 Genomes genotypes,
but IGSR grants no blanket redistribution licence - it says so itself - so the training corpus
cannot ship in a public web bundle. gnomAD is CC0, which is the strongest possible position: no
attribution obligation, no share-alike, no field-of-use limit. It is a SUBSTITUTE reference
population, not the training data, and everything downstream is labelled that way.

WHY IT IS WORTH PLOTTING AT ALL. The page already shows that one globally-set high-risk cut flags
16.2% of AFR against 2.8% of EUR (PRS313; the retired PRS77 figures were 29.35% and 3.75% of AMR).
That chart states the disparity; it does not explain it. This
one does: for each of the 313 variants the score is built from, it plots the allele frequency in
African-ancestry samples against European-ancestry samples. Points off the diagonal are variants
whose frequency depends on ancestry, and PRS313 was developed in European cohorts - its own 28 Catalog
AUROC records run 0.5877-0.668 for European samples against 0.55 for an African-ancestry one. A score
whose weights were fitted where the
allele frequencies differ is exactly how a distribution shift arises without any accuracy claim
being wrong.

Cached rather than fetched at build time so the generator is reproducible offline and so a
public-demo build never depends on a third-party API being up. gnomAD is CC0, so caching is
unambiguously permitted.

Usage:  .venv/bin/python scripts/fetch_gnomad_af.py            # the active score (PRS313)
        .venv/bin/python scripts/fetch_gnomad_af.py --score prs77   # the retired cache
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "genomics" / "gnomad_af.json"
OUT313 = REPO / "data" / "genomics" / "gnomad_af_prs313.json"
API = "https://gnomad.broadinstitute.org/api"

#: PGS000004's GRCh38-harmonised scoring file. PGS000004 publishes NO rsIDs, so the only way to look
#: its variants up is by coordinate - and gnomAD v4 is GRCh38-only while the score is authored on
#: GRCh37. The Catalog publishes both harmonisations, so each join uses its own build from the same
#: authoritative source: GRCh37 against the phase-3 VCF for dosages, GRCh38 here.
#:
#: Measured 2026-08-11, and this is why the build matters rather than being a technicality:
#: **310 of the 313 positions differ between GRCh37 and GRCh38.** Querying gnomAD v4 with the
#: author-reported GRCh37 coordinates would have missed or mis-hit almost every variant, and an
#: allele-frequency chart drawn from the survivors would still have rendered.
PGS4_GRCH38 = ("https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS000004/"
               "ScoringFiles/Harmonized/PGS000004_hmPOS_GRCh38.txt.gz")

QUERY_RSID = """
{ variant(rsid: "%s", dataset: gnomad_r4) {
    variant_id
    genome { populations { id ac an } }
} }
"""

#: Same field, keyed by coordinate. gnomAD's `variantId` form is `chrom-pos-ref-alt`, and the effect
#: allele is not necessarily ALT - so both orientations are tried and whichever resolves is recorded
#: along with which one it was, because an allele-frequency value means nothing without knowing which
#: allele it counts.
QUERY_VARIANT_ID = """
{ variant(variantId: "%s", dataset: gnomad_r4) {
    variant_id
    genome { populations { id ac an } }
} }
"""


def fetch(rsid: str, timeout: int = 45) -> dict | None:
    return _query(QUERY_RSID % rsid, rsid, timeout)


def fetch_by_position(chrom: str, pos: str, ea: str, oa: str, timeout: int = 45) -> dict | None:
    """gnomAD record for a GRCh38 coordinate, trying both allele orientations.

    Records `effect_is_alt` so a downstream frequency is never read as "frequency of the effect
    allele" when it is the frequency of the other one. That distinction is the whole point of the
    chart this feeds: a variant whose effect-allele frequency differs by ancestry is the finding, and
    getting the orientation backwards would invert it.
    """
    for ref, alt, effect_is_alt in ((oa, ea, True), (ea, oa, False)):
        vid = f"{chrom}-{pos}-{ref}-{alt}"
        rec = _query(QUERY_VARIANT_ID % vid, vid, timeout)
        if rec:
            rec["effect_is_alt"] = effect_is_alt
            rec["queried"] = vid
            return rec
    return None


def _query(query: str, label: str, timeout: int) -> dict | None:
    body = json.dumps({"query": query}).encode()
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "qsh-research/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"  {label}: FETCH FAILED ({type(exc).__name__})")
        return None
    v = (payload.get("data") or {}).get("variant")
    if not v or not v.get("genome"):
        return None
    pops = {p["id"]: p for p in v["genome"]["populations"]}
    out = {"variant_id": v.get("variant_id")}
    for pop in ("afr", "nfe", "eas", "sas", "amr"):
        p = pops.get(pop)
        # an == 0 means the variant was not callable in that group; recording None keeps that
        # distinct from "frequency is genuinely zero", which is a different statement.
        out[pop] = (p["ac"] / p["an"]) if p and p.get("an") else None
    return out


def _prs313_variants() -> list[dict[str, str]]:
    """The 313 PGS000004 variants with their GRCh38 coordinates, from the Catalog's own file."""
    with urllib.request.urlopen(PGS4_GRCH38, timeout=120) as r:  # noqa: S310 - fixed EBI FTP URL
        text = gzip.decompress(r.read()).decode()
    rows = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    header = rows[0].split("\t")
    col = {n: j for j, n in enumerate(header)}
    for need in ("hm_chr", "hm_pos", "effect_allele", "other_allele", "chr_position"):
        if need not in col:
            raise SystemExit(f"{need} missing from the harmonised file: {header}")
    out = []
    for ln in rows[1:]:
        f = ln.split("\t")
        chrom, pos = f[col["hm_chr"]].strip(), f[col["hm_pos"]].strip()
        if not chrom or not pos:
            # A row with no harmonised position cannot be looked up by coordinate, and there is no
            # rsID to fall back to. Recorded as unresolvable rather than skipped silently.
            chrom = pos = ""
        out.append({
            "chrom": chrom, "pos": pos,
            "ea": f[col["effect_allele"]].strip(), "oa": f[col["other_allele"]].strip(),
            # The GRCh37 position is the key the dosage join and the loader use, so it is what ties a
            # cache entry back to the rest of the pipeline. Both builds are carried deliberately.
            "key37": f"{f[col['chr_name']].strip()}:{f[col['chr_position']].strip()}:"
                     f"{f[col['effect_allele']].strip()}",
        })
    return out


def _run_prs77() -> int:
    # Moved out of web/public/ in Round 32. Nothing in the browser read it once realData.ts switched to
    # the PRS313 payload, but Vite still SERVED it - 30 individuals x 77 SNPs of genotype data published
    # for no demo benefit, carrying a provenance string the registry had already replaced. That is the
    # same argument DATA_LICENSES section 4 makes for not shipping the full matrix, applied to a file
    # that had quietly become gratuitous. Kept under data/ because this retired cache is provenance.
    src = json.loads((REPO / "data" / "genomics" / "pgs000001_prs_retired.json").read_text())
    rsids = src["rsids"]
    cache = json.loads(OUT.read_text()) if OUT.exists() else {}
    print(f"PGS000001: {len(rsids)} variants by rsID; {len(cache)} already cached")
    ok = 0
    for i, rs in enumerate(rsids, 1):
        if rs in cache and cache[rs] is not None:
            ok += 1
            continue
        rec = fetch(rs)
        cache[rs] = rec
        if rec:
            ok += 1
            print(f"  [{i:2d}/{len(rsids)}] {rs:14s} AFR={rec['afr'] if rec['afr'] is None else round(rec['afr'],4)}"
                  f"  NFE={rec['nfe'] if rec['nfe'] is None else round(rec['nfe'],4)}")
        else:
            print(f"  [{i:2d}/{len(rsids)}] {rs:14s} no record")
        time.sleep(0.6)                      # be a polite client of a free public API
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}  ({ok}/{len(rsids)} resolved, {OUT.stat().st_size/1024:.1f} KB)")
    return ok


def _run_prs313() -> int:
    variants = _prs313_variants()
    cache = json.loads(OUT313.read_text()) if OUT313.exists() else {}
    print(f"PGS000004: {len(variants)} variants by GRCh38 coordinate; {len(cache)} already cached")
    unresolvable = sum(1 for v in variants if not v["pos"])
    if unresolvable:
        print(f"  {unresolvable} carry no harmonised GRCh38 position and cannot be looked up")
    ok = 0
    for i, v in enumerate(variants, 1):
        k = v["key37"]
        if k in cache and cache[k] is not None:
            ok += 1
            continue
        rec = fetch_by_position(v["chrom"], v["pos"], v["ea"], v["oa"]) if v["pos"] else None
        cache[k] = rec
        if rec:
            ok += 1
            afr = rec["afr"] if rec["afr"] is None else round(rec["afr"], 4)
            nfe = rec["nfe"] if rec["nfe"] is None else round(rec["nfe"], 4)
            flip = "" if rec.get("effect_is_alt") else "  (effect allele is REF)"
            print(f"  [{i:3d}/{len(variants)}] {k:24s} AFR={afr}  NFE={nfe}{flip}")
        else:
            print(f"  [{i:3d}/{len(variants)}] {k:24s} no record")
        time.sleep(0.6)
    OUT313.parent.mkdir(parents=True, exist_ok=True)
    OUT313.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT313.relative_to(REPO)}  ({ok}/{len(variants)} resolved, "
          f"{OUT313.stat().st_size/1024:.1f} KB)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # Defaults to the ACTIVE score. It defaulted to "both", so a bare run made 77 gnomAD API calls
    # to refresh data/genomics/gnomad_af.json - a cache with a writer and, since the 2D overlay moved
    # to the position-keyed cache, no reader at all. A written artefact nobody reads is the
    # "metric with no reader" defect wearing a data-file costume, and it was costing a third party's
    # rate limit to maintain.
    ap.add_argument("--score", choices=("prs77", "prs313", "both"), default="prs313",
                    help="prs313 (default) = PGS000004 by GRCh38 coordinate, the active score; "
                         "prs77 = the retired PGS000001 rsID cache, kept for provenance")
    args = ap.parse_args()
    if args.score in ("prs77", "both"):
        _run_prs77()
    if args.score in ("prs313", "both"):
        _run_prs313()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

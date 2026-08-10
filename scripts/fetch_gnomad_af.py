#!/usr/bin/env python3
"""Cache gnomAD v4 per-ancestry allele frequencies for the PGS000001 variants.

WHY THIS DATA AND NOT THE TRAINING CORPUS. The genomics page scores real 1000 Genomes genotypes,
but IGSR grants no blanket redistribution licence - it says so itself - so the training corpus
cannot ship in a public web bundle. gnomAD is CC0, which is the strongest possible position: no
attribution obligation, no share-alike, no field-of-use limit. It is a SUBSTITUTE reference
population, not the training data, and everything downstream is labelled that way.

WHY IT IS WORTH PLOTTING AT ALL. The page already shows that one globally-set high-risk cut flags
29.35% of AFR against 3.75% of AMR. That chart states the disparity; it does not explain it. This
one does: for each of the 77 variants the score is built from, it plots the allele frequency in
African-ancestry samples against European-ancestry samples. Points off the diagonal are variants
whose frequency depends on ancestry, and PGS000001's development cohort was 100% European
(22,627 individuals, Mavaddat et al. JNCI 2015). A score whose weights were fitted where the
allele frequencies differ is exactly how a distribution shift arises without any accuracy claim
being wrong.

Cached rather than fetched at build time so the generator is reproducible offline and so a
public-demo build never depends on a third-party API being up. gnomAD is CC0, so caching is
unambiguously permitted.

Usage:  .venv/bin/python scripts/fetch_gnomad_af.py
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "web" / "public" / "data" / "genomics" / "pgs000001_prs.json"
OUT = REPO / "data" / "genomics" / "gnomad_af.json"
API = "https://gnomad.broadinstitute.org/api"

QUERY = """
{ variant(rsid: "%s", dataset: gnomad_r4) {
    variant_id
    genome { populations { id ac an } }
} }
"""


def fetch(rsid: str, timeout: int = 45) -> dict | None:
    body = json.dumps({"query": QUERY % rsid}).encode()
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "qsh-research/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"  {rsid}: FETCH FAILED ({type(exc).__name__})")
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


def main() -> int:
    src = json.loads(SRC.read_text())
    rsids = src["rsids"]
    cache = json.loads(OUT.read_text()) if OUT.exists() else {}
    print(f"PGS000001: {len(rsids)} variants; {len(cache)} already cached")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

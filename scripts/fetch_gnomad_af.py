#!/usr/bin/env python3
"""Cache gnomAD genome allele frequencies of the active PGS score's effect alleles, from Ensembl.

WHERE THE FREQUENCIES COME FROM. Ensembl's REST variation endpoint,
``https://rest.ensembl.org/variation/human/{rsid}?genotypes=1;pops=1``, lists for each variant the
allele frequencies of the gnomAD genome groups as populations named ``gnomADg:<group>`` (``afr``,
``amr``, ``eas``, ``nfe``, ``sas`` among them). This script reads those and nothing else: no gnomAD
service is queried. Ensembl's response does not state which gnomAD release it carries, so the cache
records the Ensembl release it was read from (``/info/data``) instead of a gnomAD version. gnomAD's
data are CC0; its terms ask users not to attempt to re-identify participants and to credit gnomAD
with a link back (gnomAD's own terms page).

THE SAME ALLELE AS THE DOSAGES. Each record is resolved with ``resolve`` from
``scripts/data/fetch_g1k_dosages.py``, the function that decides which allele the 1000 Genomes dosages
count, on the same record, so a frequency here is of exactly the allele the score is evaluated on, in
Ensembl's spelling. A variant the dosage fetcher leaves unscored is left out here too. The cache also
keeps that allele's frequency over the panel's 503 European individuals, so the overlay can check two
references of one allele against each other.

WHY IT IS WORTH PLOTTING. The page's ancestry-strata chart shows that one global high-risk cut flags
very different shares of each super-population. This cache feeds the chart that says why: for each
variant of the score, its effect allele's frequency in one gnomAD group against another.

Usage:  .venv/bin/python scripts/fetch_gnomad_af.py --cache /path/to/ensembl/records
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GROUPS = ("afr", "amr", "eas", "nfe", "sas")


def _fetcher():
    """`scripts/data/fetch_g1k_dosages.py` as a module: one resolver for the dosages and this cache."""
    spec = importlib.util.spec_from_file_location("fetch_g1k_dosages",
                                                  REPO / "scripts" / "data" / "fetch_g1k_dosages.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def group_frequency(record: dict, group: str, allele: str) -> float | None:
    """The frequency Ensembl lists for ``allele`` in gnomAD genome group ``group``, else None.

    None when the group lists no frequency for that allele, rather than 0: an allele Ensembl does not
    list is unknown here, not shown to be absent.
    """
    hit = [p["frequency"] for p in record.get("populations", [])
           if p.get("population") == f"gnomADg:{group}" and p.get("allele") == allele]
    return float(hit[0]) if hit else None


def ensembl_release() -> int | None:
    req = urllib.request.Request("https://rest.ensembl.org/info/data",
                                 headers={"Content-Type": "application/json", "User-Agent": "qsh-provenance/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:          # noqa: S310 - fixed Ensembl URL
        releases = json.load(r).get("releases") or [None]
    return releases[0]


def main() -> int:
    sys.path.insert(0, str(REPO / "src"))
    from qsh.data.loaders import _PRS_FILES, ACTIVE_PRS  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", choices=sorted(_PRS_FILES), default=ACTIVE_PRS)
    ap.add_argument("--cache", type=Path, default=None, help="directory of raw Ensembl records to reuse")
    ap.add_argument("--fetched", default="", help="date the records were read, YYYY-MM-DD")
    ap.add_argument("--ensembl-release", type=int, default=None,
                    help="the release the cached records came from (default: ask Ensembl now)")
    a = ap.parse_args()

    f = _fetcher()
    score_file, _ = _PRS_FILES[a.score]
    stem = Path(score_file).stem.replace("_breast_cancer", "")
    out = REPO / "data" / "genomics" / f"gnomad_af_{stem}.json"
    rows = f.read_score(REPO / "data" / "genomics" / score_file)
    panel = f.read_panel(REPO / "data" / "genomics" / "g1k_panel.tsv")
    panel = {s: panel[s] for s in sorted(panel)}
    eur = [s for s in panel if panel[s][1] == "EUR"]

    cache: dict = {"_source": {
        "endpoint": "https://rest.ensembl.org/variation/human/{rsid}?genotypes=1;pops=1",
        "populations": [f"gnomADg:{g}" for g in GROUPS],
        "ensembl_release": a.ensembl_release if a.ensembl_release else ensembl_release(),
        "fetched": a.fetched,
        "note": ("gnomAD genome frequencies as Ensembl's variation endpoint lists them; Ensembl does not "
                 "name the gnomAD release. CC0; gnomAD's terms ask users not to attempt to "
                 "re-identify participants."),
    }}
    kept = 0
    for row in rows:
        record = f.get_record(row["rsID"], a.cache)
        effect, counts, how = f.resolve(row, record, panel)
        if effect is None:
            cache[row["rsID"]] = None
            print(f"  {row['rsID']}: left out, as in the dosages ({how})")
            continue
        rec = {"effect_allele_ensembl": effect,
               "g1k_eur": round(sum(counts[s] for s in eur) / (2 * len(eur)), 6)}
        rec.update({g: group_frequency(record, g, effect) for g in GROUPS})
        cache[row["rsID"]] = rec
        kept += 1
    out.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out.relative_to(REPO)}: {kept} of {len(rows)} variants resolved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

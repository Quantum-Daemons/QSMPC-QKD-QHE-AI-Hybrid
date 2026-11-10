#!/usr/bin/env python
"""Where the variants of the genomics page's polygenic score lie on the genome, for its locus plot (Round 86).

The browser payload names each variant by rsID and carries no position. The scoring file does, so this writes rsID, chromosome, GRCh37 position and effect weight for the 183 variants the page scores, in the payload's order, with the notices the score's CC BY 4.0 publication asks for.

    python scripts/data/build_prs_loci.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "data" / "genomics" / "pgs000072_breast_cancer.tsv"
PAYLOAD = REPO / "web" / "public" / "data" / "genomics" / "pgs000072_breast_cancer.json"
OUT = REPO / "web" / "public" / "data" / "genomics" / "pgs000072_loci.json"


def main() -> int:
    payload = json.loads(PAYLOAD.read_text())
    lines = [ln for ln in SOURCE.read_text().splitlines() if ln and not ln.startswith("#")]
    head = lines[0].split("\t")
    at = {name: head.index(name) for name in ("rsID", "chr_name", "chr_position", "effect_allele", "effect_weight", "variant_description")}
    rows = {r[at["rsID"]]: r for r in (ln.split("\t") for ln in lines[1:])}
    loci = []
    for key, weight in zip(payload["keys"], payload["weights"], strict=True):
        r = rows[key]
        if abs(float(r[at["effect_weight"]]) - float(weight)) > 1e-9:
            raise SystemExit(f"{key}: the scoring file and the browser payload carry different weights")
        position = r[at["chr_position"]]
        if not position:         # the scoring file leaves it blank for a few variants and gives it in the description instead
            found = re.search(r"= \d+:(\d+)", r[at["variant_description"]])
            if not found:
                raise SystemExit(f"{key}: no position in the scoring file")
            position = found.group(1)
        loci.append({"rsid": key, "chr": r[at["chr_name"]], "pos": int(position), "effect_allele": r[at["effect_allele"]], "beta": float(r[at["effect_weight"]])})
    OUT.write_text(json.dumps({
        "name": payload["name"], "score_id": payload["score_id"], "build": "GRCh37",
        "licence": "CC-BY-4.0 (the effect weights)", "licence_url": payload["licence_url"], "attribution": payload["attribution"],
        "modifications": "The 183 scored variants of the published file, its rsID, chromosome, position, effect allele and weight, in the browser payload's order.",
        "loci": loci,
    }, separators=(",", ":")) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}: {len(loci)} loci")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

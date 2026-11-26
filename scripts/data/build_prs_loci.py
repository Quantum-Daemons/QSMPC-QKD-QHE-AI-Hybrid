#!/usr/bin/env python
"""Where the variants of the genomics page's polygenic score lie on the genome, for its locus plot (Round 86).

The browser payload names each variant by rsID and carries no position. The scoring file does, so this writes rsID,
chromosome, GRCh37 position and effect weight for the variants the page scores, in the payload's order, with the
notices the score's publication asks for. The score is the active one (`ACTIVE_PRS`; PGS000783 since Round 86's batch
6), and its weight type is read from the scoring file's own `#weight_type` line: PGS000783's weights are beta/SE, not
log odds ratios, and the plot labels them so.

    python scripts/data/build_prs_loci.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qsh.data.loaders import _PRS_FILES, ACTIVE_PRS  # noqa: E402
from qsh.models.registry import DEMO_SAMPLE_FILES  # noqa: E402

#: The Catalog's `#weight_type` -> what the plot calls a weight, and the field's meaning in this file.
WEIGHT_LABELS = {
    "log(OR)": ("ln(OR)", "the log odds ratio its source GWAS reported"),
    "Inverse-variance weighting": ("beta/SE", "beta/SE, the log odds ratio over its standard error in the source GWAS"),
}


def main() -> int:
    source = REPO / "data" / "genomics" / _PRS_FILES[ACTIVE_PRS][0]
    payload_path = REPO / DEMO_SAMPLE_FILES["genomics"]
    out = payload_path.with_name(payload_path.name.replace("_breast_cancer.json", "_loci.json"))
    payload = json.loads(payload_path.read_text())
    text = source.read_text().splitlines()
    weight_type = next(ln.split("=", 1)[1] for ln in text if ln.startswith("#weight_type="))
    citation = next(ln.split("=", 1)[1] for ln in text if ln.startswith("#citation="))
    label, meaning = WEIGHT_LABELS[weight_type]
    lines = [ln for ln in text if ln and not ln.startswith("#")]
    head = lines[0].split("\t")
    at = {name: head.index(name) for name in ("rsID", "chr_name", "chr_position", "effect_allele", "effect_weight", "variant_description")}
    rows = {r[at["rsID"]]: r for r in (ln.split("\t") for ln in lines[1:])}
    loci = []
    from_description = 0
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
            from_description += 1
        loci.append({"rsid": key, "chr": r[at["chr_name"]], "pos": int(position), "effect_allele": r[at["effect_allele"]],
                     "weight": float(r[at["effect_weight"]])})
    out.write_text(json.dumps({
        "name": payload["name"], "score_id": payload["score_id"], "build": "GRCh37",
        "weight_type": weight_type, "weight_label": label, "weight_meaning": meaning, "citation": citation,
        "positions_from_description": from_description,
        "licence": payload["licence"].split(";")[0].strip(), "licence_url": payload["licence_url"],
        "attribution": payload["attribution"],
        "modifications": (f"The {len(loci)} scored variants of the published file, its rsID, chromosome, position, "
                          "effect allele and weight, in the browser payload's order."),
        "loci": loci,
    }, separators=(",", ":")) + "\n")
    print(f"wrote {out.relative_to(REPO)}: {len(loci)} loci, weights {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

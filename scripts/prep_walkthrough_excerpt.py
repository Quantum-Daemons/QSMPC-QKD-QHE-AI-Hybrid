#!/usr/bin/env python
"""Write the excerpt the 5G-NIDD walkthrough reads: 200 pairs of the flows the published file holds twice.

The file is 472 MB and cannot be served, so the walkthrough checks evenly spaced pairs, on every one of the file's
113 columns, and states the counts over the whole file beside them. Values are the file's own strings, so nothing a
reader compares has been rounded.

Usage: QSH_DATA_ROOT=$PWD/data-full python scripts/prep_walkthrough_excerpt.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.models import datasets as D  # noqa: E402
from qsh.models.registry import DS_5G_NIDD  # noqa: E402

PAIRS = 200
OUT = REPO / "walkthroughs" / "public" / "nidd_copies_excerpt.json"


def main() -> int:
    frame = pd.read_csv(D.nidd_fields_preserved_csv(), dtype=str, keep_default_na=False, low_memory=False)
    flood = (frame["SrcAddr"] == D.NIDD_COPY_SOURCE) & (frame["DstAddr"] == D.NIDD_COPY_DESTINATION)
    benign = np.flatnonzero(flood & (frame["Attack Type"] == "Benign"))
    labelled = np.flatnonzero(flood & (frame["Attack Type"] == "UDPFlood"))
    assert len(benign) == len(labelled) == D.NIDD_CONTRADICTORY_COPIES, (len(benign), len(labelled))
    assert benign.max() < D.NIDD_SECOND_FILE_STARTS_AT <= labelled.min()
    picks = np.linspace(0, len(benign) - 1, PAIRS, dtype=int)
    payload = {
        "provenance": "5G-NIDD, real 5G network traffic: flows from BTS1_BTS2_fields_preserved.csv.",
        "licence": DS_5G_NIDD.licence,
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": DS_5G_NIDD.attribution,
        "modifications": f"{PAIRS} evenly spaced pairs of the {len(benign):,} flows the file holds twice; every column, as the file's own strings.",
        "rows_in_file": len(frame),
        "pairs_in_file": len(benign),
        "columns": list(frame.columns),
        "labelled_benign": frame.iloc[benign[picks]].to_numpy().tolist(),
        "labelled_udpflood": frame.iloc[labelled[picks]].to_numpy().tolist(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}, {OUT.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

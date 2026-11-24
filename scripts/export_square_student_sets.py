#!/usr/bin/env python
"""The square student's three CKKS parameter sets, error against time, for the medical page's plot (Round 86, 2026-11-23).

The owner's choice of 2026-11-23: plot the square student's largest decrypted-logit error against its time per image for
sets A, A2 and B, with the pre-registered 1e-3 limit, from the pre-registered record. The page's config carries A2 and
B's figures, and A, which missed the limit and is not offered, only as a sentence; this copies all three from the record
(`docs/measurements/medical-square-student-2026-11-19.json`) so the plot can draw A as well, marked "not offered".
Aggregate figures over the 156 test images only: no image and no logit. `tests/test_r86_square_student_sets_plot.py`
holds the served file equal to what this writes.

    python scripts/export_square_student_sets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

RECORD = REPO / "docs" / "measurements" / "medical-square-student-2026-11-19.json"
OUTPUT = REPO / "web" / "public" / "data" / "medical" / "square_student_sets.json"
KEY = "medical_fl.square_student"
SETS = ("A", "A2", "B")
FIELDS = ("N", "bits", "scale_bits", "modulus_bits", "pq_bound_bits", "images", "max_abs_logit_error",
          "median_abs_logit_error", "median_s", "max_s", "input_bytes", "output_bytes", "decisions_differ", "criterion_3")
#: The pre-registered limit on the largest absolute decrypted-logit error (criterion 3 of the pre-registration).
LIMIT = 1e-3


def build() -> dict:
    from qsh.models import registry

    record = json.loads(RECORD.read_text(encoding="utf-8"))
    dataset = registry.get(KEY).dataset
    return {
        "_about": "What the medical page's square-student plot draws; written by scripts/export_square_student_sets.py "
                  "from docs/measurements/medical-square-student-2026-11-19.json. Aggregate figures only.",
        "source": "docs/measurements/medical-square-student-2026-11-19.json",
        "timing": "median_s and max_s: the seconds to encrypt, evaluate and decrypt one image, keys excluded",
        "timed_with": f"TenSEAL {record['environment']['tenseal']} in Python {record['environment']['python']} on the "
                      f"development machine, not in a browser",
        "limit": LIMIT,
        "limit_note": "the pre-registered limit on the largest absolute decrypted-logit error (criterion 3); a set "
                      "over it is not offered",
        "sets": {name: {k: record["ckks"][name][k] for k in FIELDS} for name in SETS},
        "attribution": {
            "provider": "MedMNIST",
            "dataset": f"{dataset.name}, the official test split (156 images)",
            "download": dataset.url,
            "citation": dataset.attribution,
            "licence": "CC BY 4.0 asserted by MedMNIST; the BUSI authors publish no licence",
            "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        },
        "modifications": "Aggregate figures only (errors, times and sizes over the 156 test images), computed by this "
                         "project's square student; no image of the dataset",
    }


def main() -> int:
    OUTPUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

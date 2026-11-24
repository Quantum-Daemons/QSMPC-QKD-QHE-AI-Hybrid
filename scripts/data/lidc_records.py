#!/usr/bin/env python3
"""Every LIDC-IDRI nodule cluster's record, from pylidc's own annotation database, with no image read.

pylidc 0.2.3 ships the readers' annotations of all 1,018 CT scans (1,010 patients) in its database, with each scan's
pixel and slice spacing. A cluster's record is what scripts/data/lidc_make_crops.py writes beside its cube: the patient,
the series, the cluster's index and size, every reader's malignancy and subtlety, and the centre the crop is cut at
(the mean of the readers' centroids, in millimetres). Written once before any image is fetched, it is what the
pre-registered split is drawn from (scripts/train/medical_nodule3d_student.py split --records), so the split does not
depend on which series a download happened to finish; the crops cut later must carry the same records.

    venv/bin/python scripts/data/lidc_records.py <out json>      # the pylidc environment of lidc_make_crops.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np


def records() -> list[dict]:
    warnings.simplefilter("ignore")
    import pylidc as pl

    out = []
    for scan in sorted(pl.query(pl.Scan).all(), key=lambda s: (s.patient_id, s.series_instance_uid)):
        ps, ss = float(scan.pixel_spacing), float(scan.slice_spacing)
        for ci, cluster in enumerate(scan.cluster_annotations(verbose=False)):
            c = np.mean([a.centroid for a in cluster], axis=0)                  # voxel (i, j, k)
            out.append({"patient": scan.patient_id, "series": scan.series_instance_uid,
                        "study": scan.study_instance_uid, "cluster": ci, "readers": len(cluster),
                        "malignancy": [int(a.malignancy) for a in cluster],
                        "subtlety": [int(a.subtlety) for a in cluster],
                        "centre_mm": [round(float(c[2] * ss), 2), round(float(c[0] * ps), 2), round(float(c[1] * ps), 2)],
                        "pixel_spacing": ps, "slice_spacing": ss})
    return out


def main() -> int:
    out = Path(sys.argv[1])
    rows = records()
    text = json.dumps({"about": "pylidc 0.2.3's nodule clusters of LIDC-IDRI (TCIA, CC BY 3.0), read from its "
                                "annotation database with no image; one record a cluster, as lidc_make_crops.py "
                                "writes beside each cube",
                       "scans": len({r["series"] for r in rows}), "patients": len({r["patient"] for r in rows}),
                       "records": rows}, indent=1) + "\n"
    out.write_text(text)
    print(f"wrote {len(rows)} records from {len({r['patient'] for r in rows})} patients, "
          f"SHA-256 {hashlib.sha256(text.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

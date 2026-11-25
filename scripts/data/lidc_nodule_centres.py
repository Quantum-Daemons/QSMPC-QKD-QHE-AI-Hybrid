#!/usr/bin/env python3
"""Each LIDC-IDRI nodule cluster's centre in the scan's patient coordinates, for one part of a written split.

Runs under pylidc 0.2.3 (its own environment: Python 3.11, numpy 1.23.5; see scripts/data/lidc_make_crops.py). The
centre is the one the crops use, the mean of the readers' centroids in pylidc's voxel indices (row i, column j, slice
k), taken to LPS millimetres from the DICOM headers: the first slice's ImagePositionPatient and ImageOrientationPatient
for the in-plane axes, and the sorted slice positions for k, interpolated between slices. TotalSegmentator's
size-only teacher (scripts/train/medical_nodule3d_teacher_totalseg.py) maps it into each mask with SimpleITK, which
reads the same series, so neither side has to agree with the other on slice order.

    venv/bin/python scripts/data/lidc_nodule_centres.py <LIDC-IDRI dir> <split json> <part> <out json>
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np


def centres(root: Path, patients: set[str]) -> list[dict]:
    warnings.simplefilter("ignore")
    import pylidc as pl

    out = []
    for patient in sorted(patients):
        for scan in pl.query(pl.Scan).filter(pl.Scan.patient_id == patient).all():
            dirs = list(Path(root, patient).glob(f"*/{scan.series_instance_uid}"))
            if not dirs or not (dirs[0] / ".complete").exists():
                continue
            clusters = scan.cluster_annotations()
            if not clusters:
                continue
            images = scan.load_all_dicom_images(verbose=False)          # sorted by slice position, as to_volume
            first = images[0]
            ipp = np.array(first.ImagePositionPatient, dtype=float)
            iop = np.array(first.ImageOrientationPatient, dtype=float)
            row_dir, col_dir = iop[:3], iop[3:]                           # along a row (j), along a column (i)
            ps = np.array(first.PixelSpacing, dtype=float)                # (between rows, between columns)
            normal = np.cross(row_dir, col_dir)
            positions = np.array([float(np.dot(np.array(im.ImagePositionPatient, dtype=float), normal)) for im in images])
            for ci, cluster in enumerate(clusters):
                i, j, k = np.mean([a.centroid for a in cluster], axis=0)
                along = float(np.interp(k, np.arange(len(positions)), positions))
                point = ipp + j * ps[1] * row_dir + i * ps[0] * col_dir
                point = point + (along - float(np.dot(point, normal))) * normal
                out.append({"patient": patient, "series": scan.series_instance_uid, "cluster": ci,
                            "lps_mm": [round(float(v), 3) for v in point],
                            "series_dir": str(dirs[0].relative_to(root))})
    return out


def main() -> int:
    root, split, part, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    patients = set(json.loads(split.read_text())["parts"][part])
    rows = centres(root, patients)
    out.write_text(json.dumps({"about": f"nodule centres in LPS mm for the {part} part of {split.name}",
                               "records": rows}, indent=1) + "\n")
    print(f"wrote {len(rows)} centres from {len({r['patient'] for r in rows})} patients")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

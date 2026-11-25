#!/usr/bin/env python3
"""28 mm nodule cubes from LIDC-IDRI, one per pylidc nodule cluster (the pre-registration's "The data").

For each scan whose CT series is downloaded and MD5-checked (its `.complete` marker), pylidc's clusters of the
readers' annotations give the nodules. Each nodule's centre is the mean of its readers' centroids; the cube is 28
points a side at 1 mm, centred there, sampled from the scan by trilinear interpolation in millimetres (air, -1000 HU,
outside the scan), clipped to [-1000, 400] HU and scaled to uint8. Kept beside each cube: the patient, the series,
the cluster's index and size, every reader's malignancy and subtlety, and the centre in millimetres. No label is
decided here and no split is made: both are fixed by references/medical-nodule3d-square-student-preregistration.md.

    venv/bin/python scripts/data/lidc_make_crops.py <LIDC-IDRI dir> <out dir> [--workers N] [--limit N]
    venv/bin/python scripts/data/lidc_make_crops.py <LIDC-IDRI dir> <out dir> --hu-side 64 --split SPLIT.json --part train

ITS ENVIRONMENT. pylidc 0.2.3 (MIT) still calls `np.int`, removed in numpy 1.24, so it runs in a Python 3.11 venv of
its own, outside this repository's lock: pylidc 0.2.3, numpy 1.23.5, scipy 1.11.4, scikit-image 0.21.0, matplotlib
3.8.4, SQLAlchemy 1.4.54 and pydicom 2.4.5, with ~/.pylidcrc pointing at the extracted collection. The series are
TCIA's LIDC-IDRI CT series (CC BY 3.0), every file checked against the MD5 TCIA's API lists for it.

Writes <out>/crops_28mm.npz (uint8 [N, 28, 28, 28], axes z, y, x) and <out>/crops_28mm.json (one record a cube).
With --hu-side S, it writes <out>/crops_{S}mm_hu.npz instead: float16 HU cubes of S points a side at 1 mm, unclipped,
for CT-FM's feature extractor, only for the patients of one part of a written split (CT-FM touches training patients
only), with <out>/crops_{S}mm_hu.json listing (series, cluster) for each cube.
"""

from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SIDE, STEP_MM, HU_LO, HU_HI = 28, 1.0, -1000.0, 400.0


def crops_of(job: tuple) -> list[tuple[dict, np.ndarray]]:
    warnings.simplefilter("ignore")
    import pylidc as pl
    from scipy.ndimage import map_coordinates

    patient, root, hu_side = job if len(job) == 3 else (*job, None)
    side = hu_side or SIDE
    out = []
    for scan in pl.query(pl.Scan).filter(pl.Scan.patient_id == patient).all():
        series_dirs = list(Path(root, patient).glob(f"*/{scan.series_instance_uid}"))
        if not series_dirs or not (series_dirs[0] / ".complete").exists():
            continue
        clusters = scan.cluster_annotations()
        if not clusters:
            continue
        vol = scan.to_volume(verbose=False).astype(np.float32)          # (i rows, j cols, k slices), HU
        ps, ss = float(scan.pixel_spacing), float(scan.slice_spacing)
        offs = (np.arange(side) - (side - 1) / 2) * STEP_MM
        for ci, cluster in enumerate(clusters):
            c = np.mean([a.centroid for a in cluster], axis=0)            # voxel (i, j, k)
            z, y, x = np.meshgrid(c[2] * ss + offs, c[0] * ps + offs, c[1] * ps + offs, indexing="ij")
            coords = np.stack([y / ps, x / ps, z / ss])                    # back to (i, j, k) voxel coordinates
            cube = map_coordinates(vol, coords, order=1, mode="constant", cval=HU_LO)
            if hu_side:
                out.append(({"patient": patient, "series": scan.series_instance_uid, "cluster": ci},
                            cube.astype(np.float16)))
                continue
            u8 = np.round((np.clip(cube, HU_LO, HU_HI) - HU_LO) / (HU_HI - HU_LO) * 255).astype(np.uint8)
            meta = {"patient": patient, "series": scan.series_instance_uid, "cluster": ci, "readers": len(cluster),
                    "malignancy": [int(a.malignancy) for a in cluster], "subtlety": [int(a.subtlety) for a in cluster],
                    "centre_mm": [round(float(c[2] * ss), 2), round(float(c[0] * ps), 2), round(float(c[1] * ps), 2)],
                    "pixel_spacing": ps, "slice_spacing": ss}
            out.append((meta, u8))
    return out


def main() -> int:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    hu_side = int(sys.argv[sys.argv.index("--hu-side") + 1]) if "--hu-side" in sys.argv else None
    patients = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("LIDC-IDRI-"))
    if hu_side:
        split = json.loads(Path(sys.argv[sys.argv.index("--split") + 1]).read_text())
        part = set(split["parts"][sys.argv[sys.argv.index("--part") + 1]])
        patients = [p for p in patients if p in part]
    if limit:
        patients = patients[:limit]
    metas, cubes = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, got in enumerate(pool.map(crops_of, [(p, str(root), hu_side) for p in patients]), 1):
            for meta, cube in got:
                metas.append(meta)
                cubes.append(cube)
            if k % 50 == 0:
                print(f"{k}/{len(patients)} patients, {len(cubes)} nodules", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    if hu_side:
        np.savez(out / f"crops_{hu_side}mm_hu.npz", crops=np.stack(cubes))
        (out / f"crops_{hu_side}mm_hu.json").write_text(json.dumps({
            "about": f"{hu_side} mm cubes at 1 mm in HU, unclipped, float16, axes z, y, x (S, P, L), the same centres "
                     "as crops_28mm; for CT-FM's feature extractor, the patients of one part only.",
            "records": metas}, indent=1))
        print(f"wrote {len(cubes)} HU cubes of {hu_side} mm", flush=True)
        return 0
    np.savez_compressed(out / "crops_28mm.npz", crops=np.stack(cubes) if cubes else np.zeros((0, SIDE, SIDE, SIDE), np.uint8))
    (out / "crops_28mm.json").write_text(json.dumps({
        "about": "28 mm cubes at 1 mm, one per pylidc nodule cluster of LIDC-IDRI (TCIA, CC BY 3.0); axes z, y, x; "
                 "HU clipped to [-1000, 400] and scaled to uint8. Data citation: Armato III, S. G., et al. (2015). "
                 "Data From LIDC-IDRI [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX",
        "patients": len({m["patient"] for m in metas}), "nodules": len(metas), "records": metas}, indent=1))
    print(f"wrote {len(cubes)} cubes from {len({m['patient'] for m in metas})} patients", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

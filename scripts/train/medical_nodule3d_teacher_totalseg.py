"""TotalSegmentator's size-only teacher candidate for the 3D nodule student (its pre-registration, "The arms").

TotalSegmentator 2.18.0's `lung_nodules` task (Apache-2.0, weights release v2.5.0-weights, Dataset913; it first crops
to the lungs with the 6 mm `total` model, Dataset298, release v2.0.0-weights) runs on each TRAINING scan only, read
from its DICOM series by SimpleITK, which keeps the scan's orientation and spacing. For each training nodule, the
mask's connected component nearest the nodule's centre (scripts/data/lidc_nodule_centres.py, LPS millimetres, the
same centre as the crop's) gives a volume in cubic millimetres; a scan whose mask is empty gives 0. The feature is
log(1 + volume), since a logarithm of 0 is undefined; a logistic regression on it is scored out of fold over the
five folds of the training patients, and the probabilities go to W/teachers/totalseg_size.json.

Usage statistics: TotalSegmentator posts them to its author's server unless its config says otherwise. TOTALSEG_HOME_DIR
must hold a config.json with "send_usage_stats": false, and the script refuses to run if it does not.

    TOTALSEG_HOME_DIR=... python scripts/train/medical_nodule3d_teacher_totalseg.py --crops DIR --work W \
        --lidc LIDC-IDRI-DIR --centres centres_train.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import medical_nodule3d_student as s  # noqa: E402


def check_home() -> Path:
    home = Path(os.environ.get("TOTALSEG_HOME_DIR", ""))
    cfg = home / "config.json"
    if not home.is_dir() or not cfg.exists() or json.loads(cfg.read_text()).get("send_usage_stats") is not False:
        raise SystemExit("TOTALSEG_HOME_DIR must hold a config.json with \"send_usage_stats\": false")
    for d in ("Dataset913_lung_nodules", "Dataset298_TotalSegmentator_total_6mm_1559subj"):
        if not (home / "nnunet" / "results" / d).is_dir():
            raise SystemExit(f"{d} is not under {home}/nnunet/results: nothing is downloaded here")
    return home


def nearest_volumes(series_dir: Path, points: list[list[float]], device: str) -> tuple[list[float], dict]:
    """The volume (mm^3) of the mask component nearest each LPS point, and what ran."""
    import cc3d
    import nibabel as nib
    import SimpleITK as sitk
    from totalsegmentator.python_api import totalsegmentator

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(series_dir)))
    image = reader.Execute()
    with tempfile.TemporaryDirectory() as tmp:
        ct = Path(tmp) / "ct.nii.gz"
        sitk.WriteImage(image, str(ct))
        t0 = time.perf_counter()
        seg = totalsegmentator(str(ct), None, task="lung_nodules", device=device, quiet=True)
        seconds = time.perf_counter() - t0
        mask_path = Path(tmp) / "mask.nii.gz"
        nib.save(seg, str(mask_path))
        mask = sitk.ReadImage(str(mask_path))               # the mask on the CT's grid, its geometry read back
    arr = sitk.GetArrayFromImage(mask) > 0                   # (z, y, x)
    voxel_mm3 = float(np.prod(mask.GetSpacing()))
    labels = cc3d.connected_components(arr.astype(np.uint8), connectivity=26)
    n = int(labels.max())
    vols = []
    if n:
        idx = np.argwhere(labels > 0)                         # (z, y, x) index of every mask voxel
        lab = labels[labels > 0]
        phys = np.array([mask.TransformIndexToPhysicalPoint((int(x), int(y), int(z))) for z, y, x in idx])
        sizes = np.bincount(lab, minlength=n + 1)
        for p in points:
            d = np.linalg.norm(phys - np.asarray(p), axis=1)
            vols.append(float(sizes[lab[int(np.argmin(d))]] * voxel_mm3))
    else:
        vols = [0.0 for _ in points]
    return vols, {"components": n, "seconds": round(seconds, 1), "voxel_mm3": voxel_mm3}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--lidc", type=Path, required=True)
    ap.add_argument("--centres", type=Path, required=True)
    ap.add_argument("--device", default="gpu")
    args = ap.parse_args()
    s.check_preregistration()
    check_home()

    from sklearn.linear_model import LogisticRegression

    split = s.read_split()
    allowed = set(split["parts"]["train"])
    cubes, records = s.load_crops(args.crops)
    _, y, patients = s.labelled(cubes, records, split["parts"], "train")
    keys = [(r["series"], r["cluster"]) for r in records if r["patient"] in allowed and s.label_of(r["malignancy"]) is not None]
    centres = json.loads(args.centres.read_text())["records"]
    assert all(c["patient"] in allowed for c in centres), "a centre belongs to a patient outside training"
    by_series: dict[str, list[dict]] = {}
    for c in centres:
        by_series.setdefault(c["series"], []).append(c)
    cache = args.work / "teachers" / "totalseg_volumes.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    for n, (series, cs) in enumerate(sorted(by_series.items()), 1):
        if series in done:
            continue
        vols, info = nearest_volumes(args.lidc / cs[0]["series_dir"], [c["lps_mm"] for c in cs], args.device)
        done[series] = {"clusters": {str(c["cluster"]): v for c, v in zip(cs, vols)}, **info}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(done) + "\n")
        print(f"  {n}/{len(by_series)} {series[-12:]}: {info['components']} components, {info['seconds']} s", flush=True)
    vol = np.array([done[sr]["clusters"][str(cl)] for sr, cl in keys])
    x = np.log1p(vol)[:, None]
    fold = s.folds_of(patients, y)
    f = np.array([fold[p] for p in patients])
    oof = np.full(len(y), np.nan)
    for k in range(s.FOLDS):
        lr = LogisticRegression(max_iter=1000).fit(x[f != k], y[f != k])
        oof[f == k] = lr.predict_proba(x[f == k])[:, 1]
    s.write_teacher(args.work, "totalseg_size", patients, y, oof,
                    {"feature": "log(1 + volume in mm^3) of the mask component nearest the centre",
                     "empty_masks": int(sum(1 for v in done.values() if v["components"] == 0)),
                     "zero_volumes": int((vol == 0).sum()), "scans": len(done),
                     "median_seconds": float(np.median([v["seconds"] for v in done.values()])),
                     "environment": s.environment()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

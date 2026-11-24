#!/usr/bin/env python3
"""LIDC-IDRI streamed through the system disk, one patient at a time, verified file by file.

The data drive failed on 2026-11-24 and its NTFS metadata stayed damaged after ntfsfix (the owner's choice: stream,
write nothing to it). For each patient with a nodule record (scripts/data/lidc_records.py), every CT series that
carries one is fetched from TCIA's public REST API (getImageWithMD5Hash), each file's MD5 checked against the list the
archive sends and the file count against TCIA's ImageCount, and the series extracted to a temporary folder in the
layout pylidc reads. Then, from those images:

- every cluster's 28 mm crop, by scripts/data/lidc_make_crops.py's own function;
- for training patients (the written split): the 64 mm HU cubes for CT-FM, the nodules' centres in LPS millimetres
  (scripts/data/lidc_nodule_centres.py), and one TotalSegmentator job a series, which the worker of
  scripts/train/medical_nodule3d_teacher_totalseg.py answers in its own environment.

The patient's folder is then deleted, and a record of what was verified is written. A stopped run resumes at the next
patient. How this differs from the pre-registration's text (883 series, TotalSegmentator during the fetch) is in
references/medical-nodule3d-square-student-preregistration-addendum.md. `merge` joins the parts into the files the training script reads.
`recut` fetched the training patients again after the first run's HU cubes overwrote their crops (part_path), and
checks each re-cut crop against the cluster records and against the centre of that HU cube.

    venv/bin/python scripts/data/lidc_stream.py run --out OUT --tmp TMP --split SPLIT --records RECORDS --queue Q
    venv/bin/python scripts/data/lidc_stream.py recut --out OUT --tmp TMP --records RECORDS
    venv/bin/python scripts/data/lidc_stream.py merge --out OUT

pylidc reads the series from the folder ~/.pylidcrc names, which must be TMP/LIDC-IDRI.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import shutil
import sys
import time
import urllib.parse
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def listing(out: Path) -> list[dict]:
    """TCIA's series list for the collection, fetched once and kept beside the outputs."""
    from lidc_download_ct import get

    f = out / "tcia_getSeries_LIDC-IDRI.json"
    if not f.exists():
        f.write_bytes(get(f"{API}/getSeries?Collection=LIDC-IDRI"))
    return json.loads(f.read_text())


def fetch_patient(patient: str, series: list[dict], root: Path) -> list[dict]:
    """Every series of one patient, verified, into root/<patient>/<study>/<series>/; what was checked, per series."""
    from lidc_download_ct import get, md5_table

    checked = []
    for s in series:
        uid = s["SeriesInstanceUID"]
        dest = root / patient / s["StudyInstanceUID"] / uid
        data = get(f"{API}/getImageWithMD5Hash?SeriesInstanceUID={urllib.parse.quote(uid)}")
        z = zipfile.ZipFile(io.BytesIO(data))
        table = md5_table(z)
        dicoms = [n for n in z.namelist() if n.lower().endswith(".dcm")]
        bad = [n for n in dicoms if hashlib.md5(z.read(n)).hexdigest() != table.get(Path(n).name)]
        expected = int(s.get("ImageCount", -1))
        if bad or len(dicoms) != expected or not dicoms:
            raise RuntimeError(f"{patient} {uid}: {len(bad)} MD5 mismatches, {len(dicoms)} files of {expected}")
        dest.mkdir(parents=True, exist_ok=True)
        for n in dicoms:
            (dest / Path(n).name).write_bytes(z.read(n))
        (dest / ".complete").write_text("{}")
        checked.append({"series": uid, "files": len(dicoms), "md5_checked": len(dicoms), "image_count": expected,
                        "zip_bytes": len(data)})
    return checked


def part_path(stem: Path, ext: str) -> Path:
    """stem + ext, never Path.with_suffix: with_suffix would turn both 'P.crops28' and 'P.hu64' into 'P.npy', and the
    HU cubes overwrote the 28 mm crops of every training patient on 2026-11-24 (the addendum records it)."""
    return stem.parent / f"{stem.name}{ext}"


def save_parts(stem: Path, got: list[tuple[dict, np.ndarray]]) -> int:
    if got:
        np.save(part_path(stem, ".npy"), np.stack([c for _, c in got]))
    part_path(stem, ".json").write_text(json.dumps([m for m, _ in got]))
    return len(got)


def run(args) -> int:
    from lidc_make_crops import crops_of
    from lidc_nodule_centres import centres

    records = json.loads(args.records.read_text())["records"]
    split = json.loads(args.split.read_text())
    train = set(split["parts"]["train"])
    wanted = {r["series"] for r in records}
    by_patient: dict[str, list[dict]] = {}
    for s in listing(args.out):
        if s.get("Modality") == "CT" and s["SeriesInstanceUID"] in wanted:
            by_patient.setdefault(s["PatientID"], []).append(s)
    missing = wanted - {s["SeriesInstanceUID"] for ss in by_patient.values() for s in ss}
    if missing:
        raise SystemExit(f"{len(missing)} recorded series are not in TCIA's listing, e.g. {sorted(missing)[:2]}")
    patients = sorted(by_patient)
    root = args.tmp / "LIDC-IDRI"
    for d in ("parts", "done"):
        (args.out / d).mkdir(parents=True, exist_ok=True)
    args.queue.mkdir(parents=True, exist_ok=True)
    todo = [p for p in patients if not (args.out / "done" / f"{p}.json").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(patients)} patients, {sum(len(v) for v in by_patient.values())} series; {len(todo)} to do", flush=True)
    t0, done_bytes = time.time(), 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        ahead: dict[str, concurrent.futures.Future] = {}
        for k, patient in enumerate(todo):
            for p in todo[k:k + args.prefetch]:
                if p not in ahead:
                    ahead[p] = pool.submit(fetch_patient, p, sorted(by_patient[p], key=lambda s: s["SeriesInstanceUID"]), root)
            try:
                checked = ahead.pop(patient).result()
            except Exception as e:  # noqa: BLE001 - recorded, and the patient is left for a later run
                print(f"[{k + 1}/{len(todo)}] {patient}: NOT verified: {type(e).__name__}: {e}", flush=True)
                shutil.rmtree(root / patient, ignore_errors=True)
                continue
            n = save_parts(args.out / "parts" / f"{patient}.crops28", crops_of((patient, str(root), None)))
            record = {"patient": patient, "series": checked, "crops": n, "train": patient in train}
            if patient in train:
                record["hu64"] = save_parts(args.out / "parts" / f"{patient}.hu64", crops_of((patient, str(root), 64)))
                cs = centres(root, {patient})
                (args.out / "parts" / f"{patient}.centres.json").write_text(json.dumps(cs))
                jobs = []
                for uid in sorted({c["series"] for c in cs}):
                    sc = [c for c in cs if c["series"] == uid]
                    job = args.queue / f"{patient}.{uid}.job.json"
                    job.write_text(json.dumps({"patient": patient, "series": uid,
                                               "series_dir": str(root / sc[0]["series_dir"]), "centres": sc}))
                    jobs.append(job)
                deadline = time.time() + 3600
                while any(j.exists() for j in jobs) and time.time() < deadline:
                    time.sleep(2)
                if any(j.exists() for j in jobs):
                    print(f"[{k + 1}/{len(todo)}] {patient}: TotalSegmentator did not answer within an hour", flush=True)
                    continue
                record["totalseg"] = [json.loads(j.with_suffix(".done").read_text()) for j in jobs]
            shutil.rmtree(root / patient, ignore_errors=True)
            (args.out / "done" / f"{patient}.json").write_text(json.dumps(record))
            done_bytes += sum(c["zip_bytes"] for c in checked)
            rate = done_bytes / max(1.0, time.time() - t0) / 1e6
            print(f"[{k + 1}/{len(todo)}] {patient}: {len(checked)} series verified, {n} crops"
                  + (f", {record['hu64']} HU cubes, {len(record['totalseg'])} segmented" if patient in train else "")
                  + f" ({rate:.1f} MB/s)", flush=True)
    return 0


def rename_old_parts(parts: Path) -> dict:
    """The parts the first run wrote as 'P.npy' and 'P.json' (see part_path), renamed by what each holds: a float16
    cube of 64 points a side is a training patient's HU cubes (which had overwritten its crops), a uint8 cube of 28 a
    crop of a patient outside training."""
    moved = {"hu64": 0, "crops28": 0}
    for npy in sorted(parts.glob("LIDC-IDRI-[0-9][0-9][0-9][0-9].npy")):
        a = np.load(npy, mmap_mode="r")
        kind = ("hu64" if a.dtype == np.float16 and a.shape[1:] == (64, 64, 64) else
                "crops28" if a.dtype == np.uint8 and a.shape[1:] == (28, 28, 28) else None)
        if kind is None:
            raise SystemExit(f"{npy.name}: {a.dtype} {a.shape} is neither kind of part")
        stem = parts / f"{npy.stem}.{kind}"
        npy.rename(part_path(stem, ".npy"))
        parts.joinpath(f"{npy.stem}.json").rename(part_path(stem, ".json"))
        moved[kind] += 1
    return moved


def recut(args) -> int:
    """The training patients' 28 mm crops again (the owner's choice of 2026-11-24, after the HU cubes overwrote them):
    every series fetched and verified as `run` does, the crops cut by the same function, and three checks a patient:
    the crops' records equal the cluster records the split was drawn from, the count equals the first run's, and each
    crop equals the centre of the first run's HU cube (same sample points, float16 there) within one uint8 level."""
    from lidc_make_crops import HU_HI, HU_LO, crops_of

    parts, done_dir, rec_dir = args.out / "parts", args.out / "done", args.out / "recut"
    rec_dir.mkdir(parents=True, exist_ok=True)
    print("renamed old parts:", rename_old_parts(parts), flush=True)
    records = json.loads(args.records.read_text())["records"]
    by_cluster = {(r["patient"], r["series"], r["cluster"]): {k: v for k, v in r.items() if k != "study"} for r in records}
    series_of = {s["SeriesInstanceUID"]: s for s in listing(args.out)}
    first = {p.stem: json.loads(p.read_text()) for p in done_dir.glob("*.json")}
    todo = sorted(p for p, d in first.items() if d["train"] and not (rec_dir / f"{p}.json").exists())
    if args.limit:
        todo = todo[:args.limit]
    root = args.tmp / "LIDC-IDRI"
    print(f"{sum(d['train'] for d in first.values())} training patients; {len(todo)} to re-cut", flush=True)
    t0, done_bytes = time.time(), 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        ahead: dict[str, concurrent.futures.Future] = {}
        for k, patient in enumerate(todo):
            for p in todo[k:k + args.prefetch]:
                if p not in ahead:
                    ss = sorted((series_of[s["series"]] for s in first[p]["series"]), key=lambda s: s["SeriesInstanceUID"])
                    ahead[p] = pool.submit(fetch_patient, p, ss, root)
            try:
                checked = ahead.pop(patient).result()
            except Exception as e:  # noqa: BLE001 - recorded, and the patient is left for a later run
                print(f"[{k + 1}/{len(todo)}] {patient}: NOT verified: {type(e).__name__}: {e}", flush=True)
                shutil.rmtree(root / patient, ignore_errors=True)
                continue
            got = crops_of((patient, str(root), None))
            shutil.rmtree(root / patient, ignore_errors=True)
            metas_equal = all(by_cluster.get((m["patient"], m["series"], m["cluster"])) == m for m, _ in got)
            hu = np.load(part_path(parts / f"{patient}.hu64", ".npy")).astype(np.float32)
            hu_meta = json.loads(part_path(parts / f"{patient}.hu64", ".json").read_text())
            same_order = [(m["series"], m["cluster"]) for m, _ in got] == [(m["series"], m["cluster"]) for m in hu_meta]
            centre = np.round((np.clip(hu[:, 18:46, 18:46, 18:46], HU_LO, HU_HI) - HU_LO) / (HU_HI - HU_LO) * 255)
            diff = np.abs(np.stack([c for _, c in got]).astype(np.int16) - centre.astype(np.int16)) if got else np.zeros(1)
            ok = metas_equal and same_order and len(got) == first[patient]["crops"] and int(diff.max()) <= 1
            if not ok:
                print(f"[{k + 1}/{len(todo)}] {patient}: re-cut does NOT match (metas {metas_equal}, order {same_order}, "
                      f"{len(got)} of {first[patient]['crops']} crops, largest level difference {int(diff.max())})", flush=True)
                continue
            save_parts(parts / f"{patient}.crops28", got)
            (rec_dir / f"{patient}.json").write_text(json.dumps({
                "patient": patient, "series": checked, "crops": len(got), "metas_equal_records": metas_equal,
                "max_level_difference_from_hu64_centre": int(diff.max()),
                "voxels_differing_from_hu64_centre": int((diff > 0).sum()), "voxels": int(diff.size)}))
            done_bytes += sum(c["zip_bytes"] for c in checked)
            rate = done_bytes / max(time.time() - t0, 1e-9) / 1e6
            print(f"[{k + 1}/{len(todo)}] {patient}: {len(checked)} series verified again, {len(got)} crops re-cut, "
                  f"{int((diff > 0).sum())} voxels a level off the HU cube's centre ({rate:.1f} MB/s)", flush=True)
    return 0


def merge(args) -> int:
    """The parts, in patient order, as the files the training and teacher scripts read."""
    def join(kind: str, about: str, name: str) -> int:
        metas, cubes = [], []
        for f in sorted((args.out / "parts").glob(f"*.{kind}.json")):
            m = json.loads(f.read_text())
            if m:
                metas += m
                cubes.append(np.load(part_path(f.parent / f.name[:-len(".json")], ".npy")))
        np.savez_compressed(args.out / f"{name}.npz", crops=np.concatenate(cubes) if cubes else np.zeros((0,)))
        (args.out / f"{name}.json").write_text(json.dumps({"about": about, "patients": len({m["patient"] for m in metas}),
                                                           "nodules": len(metas), "records": metas}, indent=1))
        return len(metas)
    a = join("crops28", "28 mm cubes at 1 mm, one per pylidc nodule cluster of LIDC-IDRI (TCIA, CC BY 3.0); axes z, y, "
             "x; HU clipped to [-1000, 400] and scaled to uint8; streamed and verified file by file (lidc_stream.py)",
             "crops_28mm")
    b = join("hu64", "64 mm cubes at 1 mm in HU, unclipped, float16, axes z, y, x (S, P, L), the same centres as "
             "crops_28mm; training patients only, for CT-FM", "crops_64mm_hu")
    cs = [c for f in sorted((args.out / "parts").glob("*.centres.json")) for c in json.loads(f.read_text())]
    (args.out / "centres_train.json").write_text(json.dumps({"about": "nodule centres in LPS mm, training patients",
                                                             "records": cs}, indent=1))
    print(f"merged {a} crops, {b} HU cubes, {len(cs)} centres")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--tmp", type=Path, required=True)
    r.add_argument("--split", type=Path, required=True)
    r.add_argument("--records", type=Path, required=True)
    r.add_argument("--queue", type=Path, required=True)
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--prefetch", type=int, default=4)
    r.add_argument("--limit", type=int)
    c = sub.add_parser("recut")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--tmp", type=Path, required=True)
    c.add_argument("--records", type=Path, required=True)
    c.add_argument("--workers", type=int, default=3)
    c.add_argument("--prefetch", type=int, default=4)
    c.add_argument("--limit", type=int)
    m = sub.add_parser("merge")
    m.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    return {"run": run, "recut": recut, "merge": merge}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

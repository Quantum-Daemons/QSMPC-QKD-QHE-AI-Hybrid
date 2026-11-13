#!/usr/bin/env python3
"""Prepare NIH ChestX-ray14 for the medical page's chest X-ray federation (Round 86, the owner's item (3)).

FROM THE ORIGINAL PNGs, not from a derived set: the owner's rule for the view-bias work. What this writes, into --out:

  images_<size>.npy   uint8, (N, size, size): every image, grey, resized with Pillow's bilinear filter, in the order
                      of the index below (memory-mapped, so it is never all in memory).
  index.csv           one row per image: its position, file name, patient, view (PA or AP), the official split
                      (train_val or test), the Effusion label, age, sex, follow-up number, the SHA-256 of its decoded
                      grey pixels, and whether a bounding box of Effusion exists for it.
  duplicates.json     groups of images whose decoded pixels are identical, by that hash, and which of them cross the
                      official split: the check of NIH's change log of 2026-10-09, which lists ten pixel-identical pairs.

THE ARCHIVES ARE READ, NOT EXTRACTED. Each member's bytes are read through tarfile's extractfile and decoded in memory,
so no path from an archive is ever written to disk and a hostile member name cannot escape --out. The archives are
untrusted data: run this with `python -I`, from outside the folder that holds them.

Usage: python -I scripts/data/prepare_nih_cxr.py --raw <folder with images_*.tar.gz and the CSVs> --out <folder> [--size 224]

The data: NIH Clinical Center, ChestX-ray14, https://nihcc.app.box.com/v/ChestXray-NIHCC (Wang et al., CVPR 2017). There are
no restrictions on its use; a link to the download site, a citation of the paper and an acknowledgement of the NIH Clinical
Center as the provider are asked for (DATA_LICENSES.md). Nothing derived from it is committed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

LABEL = "Effusion"


def read_entries(raw: Path) -> dict[str, dict]:
    with open(raw / "Data_Entry_2017_v2020.csv", newline="") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    if head[:7] != ["Image Index", "Finding Labels", "Follow-up #", "Patient ID", "Patient Age", "Patient Sex", "View Position"]:
        raise SystemExit(f"unexpected Data_Entry header: {head}")
    return {r[0]: {"labels": r[1].split("|"), "follow_up": int(r[2]), "patient": int(r[3]), "age": int(r[4]),
                   "sex": r[5], "view": r[6]} for r in body}


def one_archive(job: tuple[str, str, int, dict[str, int]]) -> list[tuple[int, str]]:
    """Decode every PNG of one archive into the shared array; return (position, pixel hash) for each."""
    tgz, npy, size, idx = job
    arr = np.load(npy, mmap_mode="r+")
    out = []
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".png"):
                continue
            name = os.path.basename(m.name)
            if name not in idx:
                raise ValueError(f"{name} in {tgz} is not in Data_Entry_2017_v2020.csv")
            f = tf.extractfile(m)
            if f is None:
                raise ValueError(f"{m.name} in {tgz} could not be read")
            im = Image.open(io.BytesIO(f.read()))
            im.load()
            g = im.convert("L")
            i = idx[name]
            out.append((i, hashlib.sha256(g.tobytes()).hexdigest()))
            arr[i] = np.asarray(g.resize((size, size), Image.BILINEAR), dtype=np.uint8)
    arr.flush()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--expected-archives", type=int, default=12, help="NIH ships 12; a test passes its own count")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    entries = read_entries(a.raw)
    test = set((a.raw / "test_list.txt").read_text().split())
    train_val = set((a.raw / "train_val_list.txt").read_text().split())
    if test & train_val:
        raise SystemExit("the official lists overlap")
    order = sorted(entries)
    if set(order) != test | train_val:
        raise SystemExit("the index and the two lists disagree")
    idx = {n: i for i, n in enumerate(order)}
    with open(a.raw / "BBox_List_2017.csv", newline="") as f:
        boxed = {r[0] for r in list(csv.reader(f))[1:] if r[1] == LABEL}
    npy = a.out / f"images_{a.size}.npy"
    np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8, shape=(len(order), a.size, a.size)).flush()
    jobs = [(str(t), str(npy), a.size, idx) for t in sorted(a.raw.glob("images_*.tar.gz"))]
    if len(jobs) != a.expected_archives:
        raise SystemExit(f"expected {a.expected_archives} archives, found {len(jobs)}")
    sha: list[str | None] = [None] * len(order)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(one_archive, jobs):
            for i, h in res:
                sha[i] = h
    missing = [order[i] for i, h in enumerate(sha) if h is None]
    if missing:
        raise SystemExit(f"{len(missing)} indexed images were in no archive, e.g. {missing[:3]}")
    with open(a.out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "image", "patient", "view", "split", "effusion", "age", "sex", "follow_up",
                    "pixel_sha256", "effusion_box"])
        for i, n in enumerate(order):
            e = entries[n]
            w.writerow([i, n, e["patient"], e["view"], "test" if n in test else "train_val",
                        int(LABEL in e["labels"]), e["age"], e["sex"], e["follow_up"], sha[i], int(n in boxed)])
    groups: dict[str, list[str]] = {}
    for i, h in enumerate(sha):
        groups.setdefault(h, []).append(order[i])
    dup = [sorted(g) for g in groups.values() if len(g) > 1]
    cross = [g for g in dup if len({"test" if n in test else "train_val" for n in g}) > 1]
    with open(a.out / "duplicates.json", "w") as f:
        json.dump({"groups": sorted(dup), "cross_split": sorted(cross), "images": len(order)}, f, indent=1)
    print(f"{len(order)} images, {len(dup)} pixel-identical groups, {len(cross)} across the official split")


if __name__ == "__main__":
    main()

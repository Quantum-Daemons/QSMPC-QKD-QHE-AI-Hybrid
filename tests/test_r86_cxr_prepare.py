"""The NIH ChestX-ray14 preparation (Round 86, the chest X-ray federation), on a synthetic folder in NIH's layout.

`scripts/data/prepare_nih_cxr.py` reads the archives in memory and writes a grey 224 px memmap, an index and the groups of
pixel-identical images. This holds, without the 42 GB of real data:

- every indexed image lands at its own position, resized, with its split, view, label and box flag;
- pixel-identical images are grouped by the hash of their decoded pixels, and a group across the official split is named;
- a member whose name the index does not know is refused, so a hostile archive cannot add or overwrite anything.
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "data" / "prepare_nih_cxr.py"
HEADER = ["Image Index", "Finding Labels", "Follow-up #", "Patient ID", "Patient Age", "Patient Sex", "View Position",
          "OriginalImage[Width", "Height]", "OriginalImagePixelSpacing[x", "y]"]


def _png(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(a.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _raw(tmp: Path, extra_member: str | None = None) -> Path:
    raw = tmp / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    same = rng.integers(0, 256, (32, 32))
    images = {
        "00000001_000.png": (same, "Effusion", "1", "PA"),
        "00000002_000.png": (same, "No Finding", "2", "AP"),          # pixel-identical to the first, in the other split
        "00000003_000.png": (rng.integers(0, 256, (32, 32)), "Effusion|Mass", "3", "AP"),
    }
    with open(raw / "Data_Entry_2017_v2020.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for name, (_, labels, pid, view) in images.items():
            w.writerow([name, labels, "0", pid, "50", "F", view, "32", "32", "0.1", "0.1"])
    (raw / "test_list.txt").write_text("00000001_000.png\n")
    (raw / "train_val_list.txt").write_text("00000002_000.png\n00000003_000.png\n")
    with open(raw / "BBox_List_2017.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Image Index", "Finding Label", "Bbox [x", "y", "w", "h]"])
        w.writerow(["00000001_000.png", "Effusion", "1", "2", "3", "4"])
    with tarfile.open(raw / "images_001.tar.gz", "w:gz") as tf:
        members = [(f"images/{n}", _png(a)) for n, (a, *_rest) in images.items()]
        if extra_member:
            members.append((extra_member, _png(images["00000003_000.png"][0])))
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return raw


def _run(raw: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(SCRIPT), "--raw", str(raw), "--out", str(out), "--size", "16",
                           "--workers", "1", "--expected-archives", "1"], capture_output=True, text=True, cwd=raw.parent)


def test_every_image_lands_at_its_position_with_its_metadata(tmp_path: Path) -> None:
    raw = _raw(tmp_path)
    out = tmp_path / "out"
    r = _run(raw, out)
    assert r.returncode == 0, r.stderr
    arr = np.load(out / "images_16.npy")
    assert arr.shape == (3, 16, 16) and arr.dtype == np.uint8
    with open(out / "index.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [x["image"] for x in rows] == ["00000001_000.png", "00000002_000.png", "00000003_000.png"]
    assert [x["split"] for x in rows] == ["test", "train_val", "train_val"]
    assert [x["view"] for x in rows] == ["PA", "AP", "AP"]
    assert [x["effusion"] for x in rows] == ["1", "0", "1"]
    assert [x["effusion_box"] for x in rows] == ["1", "0", "0"]
    assert np.array_equal(arr[0], arr[1]) and not np.array_equal(arr[0], arr[2])


def test_pixel_identical_images_are_grouped_and_a_cross_split_group_is_named(tmp_path: Path) -> None:
    raw = _raw(tmp_path)
    out = tmp_path / "out"
    assert _run(raw, out).returncode == 0
    with open(out / "duplicates.json") as f:
        dup = json.load(f)
    assert dup["groups"] == [["00000001_000.png", "00000002_000.png"]]
    assert dup["cross_split"] == dup["groups"]
    assert dup["images"] == 3


@pytest.mark.parametrize("member", ["../escaped.png", "images/99999999_000.png"])
def test_a_member_the_index_does_not_know_is_refused(tmp_path: Path, member: str) -> None:
    raw = _raw(tmp_path, extra_member=member)
    out = tmp_path / "out"
    r = _run(raw, out)
    assert r.returncode != 0
    assert "is not in Data_Entry_2017_v2020.csv" in r.stderr
    assert not (tmp_path / "escaped.png").exists() and not (raw.parent / "escaped.png").exists()

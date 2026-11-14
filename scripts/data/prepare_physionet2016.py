#!/usr/bin/env python3
"""Prepare the PhysioNet/CinC Challenge 2016 heart sounds for the medical page's heart-sound federation (Round 86).

FROM THE RELEASED ZIP, read in memory: no member is extracted, so no path from the archive is ever written to disk, and a
member is read only when the source's own REFERENCE-SQI.csv names it. What this writes, into --out:

  signals.npy    float16, one array: every recording of training-a to training-f, band-passed 25 to 400 Hz (fourth-order
                 Butterworth, forward and backward, the band Springer et al.'s segmentation uses) and standardised to zero
                 mean and unit variance, end to end in the order of the index.
  index.csv      one row per recording: its position, source (a to f), record, label (1 abnormal, 0 normal), signal
                 quality (1 good, 0 the 'unsure' the Challenge's hand correction marked), offset and length in samples.

All recordings are 2,000 Hz, 16-bit mono. Run with `python -I`, from outside the folder that holds the zip.

The data: Clifford et al., PhysioNet/CinC Challenge 2016, https://physionet.org/content/challenge-2016/1.0.0/, under the
Open Data Commons Attribution License v1.0, which licenses the database and says it does not cover its contents
(DATA_LICENSES.md). Nothing derived from a recording is committed here.

Usage: python -I scripts/data/prepare_physionet2016.py --zip <challenge-2016-1.0.0.zip> --out <folder>
"""

from __future__ import annotations

import argparse
import csv
import io
import wave
import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

RATE = 2000
SOURCES = "abcdef"
BAND = (25.0, 400.0)


def read_wav(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data)) as w:
        if w.getframerate() != RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"expected {RATE} Hz 16-bit mono, got {w.getframerate()} Hz, {w.getnchannels()} channels, {8 * w.getsampwidth()} bits")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)


def condition(x: np.ndarray) -> np.ndarray:
    sos = butter(4, BAND, btype="bandpass", fs=RATE, output="sos")
    y = sosfiltfilt(sos, x)
    sd = y.std()
    if not sd > 0:
        raise ValueError("a silent recording")
    return (y - y.mean()) / sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows, chunks, offset = [], [], 0
    with zipfile.ZipFile(a.zip) as z:
        names = set(z.namelist())
        roots = {n.split("/")[0] for n in names if "/training-a/" in n}
        if len(roots) != 1:
            raise SystemExit("expected one top folder holding training-a")
        root = roots.pop()
        for s in SOURCES:
            ref = f"{root}/training-{s}/REFERENCE-SQI.csv"
            if ref not in names:
                raise SystemExit(f"{ref} is not in the zip")
            for rec, label, sqi in csv.reader(io.TextIOWrapper(z.open(ref), encoding="ascii")):
                if not rec.isalnum() or label not in ("1", "-1") or sqi not in ("0", "1"):
                    raise ValueError(f"an unexpected row in {ref}: {rec}, {label}, {sqi}")
                member = f"{root}/training-{s}/{rec}.wav"
                if member not in names:
                    raise ValueError(f"{member} is named by {ref} and is not in the zip")
                x = condition(read_wav(z.read(member)))
                rows.append([len(rows), s, rec, int(label == "1"), int(sqi), offset, len(x)])
                chunks.append(x.astype(np.float16))
                offset += len(x)
    np.save(a.out / "signals.npy", np.concatenate(chunks))
    with open(a.out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "source", "record", "abnormal", "good_quality", "offset", "length"])
        w.writerows(rows)
    print(f"{len(rows)} recordings, {offset / RATE / 3600:.2f} hours, {sum(r[3] for r in rows)} abnormal")


if __name__ == "__main__":
    main()

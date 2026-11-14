#!/usr/bin/env python3
"""Prepare the CirCor DigiScope phonocardiograms for the medical page's murmur detection under differential privacy
(Round 86).

FROM THE RELEASED ZIP, read in memory: no member is extracted, and a recording is read only when its name has the form
<patient>_<location>[_<n>].wav with a patient that training_data.csv lists and a location among AV, PV, TV, MV and Phc.
What this writes, into --out:

  signals.npy    float16, one array: every recording resampled from 4,000 to 2,000 Hz (polyphase, the same rate as the
                 PhysioNet 2016 preparation), band-passed 25 to 400 Hz and standardised, end to end in the index's order.
  index.csv      one row per recording: its position, patient, person, location, the patient's murmur label (Present,
                 Absent or Unknown), outcome, campaign, and its offset and length in samples.

A person who attended both screening campaigns has a patient ID in each, and the table's Additional ID names the other
one: the person column joins such IDs (the smallest ID of the group), so that a split or a privacy unit can follow the
person rather than the ID. In version 1.0.3 the public table has 140 such IDs, 70 people, every Additional ID naming a
patient the same table lists.

The data: Oliveira et al., The CirCor DigiScope Phonocardiogram Dataset, version 1.0.3, PhysioNet,
https://physionet.org/content/circor-heart-sound/1.0.3/, under the Open Data Commons Attribution License v1.0, which
licenses the database and says it does not cover its contents (DATA_LICENSES.md). Nothing derived from a recording is
committed here. Run with `python -I`, from outside the folder that holds the zip.

Usage: python -I scripts/data/prepare_circor.py --zip <circor-heart-sound-1.0.3.zip> --out <folder>
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import wave
import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

SOURCE_RATE, RATE = 4000, 2000
BAND = (25.0, 400.0)
LOCATIONS = ("AV", "PV", "TV", "MV", "Phc")
NAME = re.compile(r"^(\d+)_(AV|PV|TV|MV|Phc)(?:_(\d+))?\.wav$")


def read_wav(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data)) as w:
        if w.getframerate() != SOURCE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"expected {SOURCE_RATE} Hz 16-bit mono, got {w.getframerate()} Hz, {w.getnchannels()} channels, {8 * w.getsampwidth()} bits")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)


def condition(x: np.ndarray) -> np.ndarray:
    y = resample_poly(x, 1, SOURCE_RATE // RATE)
    y = sosfiltfilt(butter(4, BAND, btype="bandpass", fs=RATE, output="sos"), y)
    sd = y.std()
    if not sd > 0:
        raise ValueError("a silent recording")
    return (y - y.mean()) / sd


def people(patients: dict[str, dict]) -> dict[str, str]:
    """Each patient ID's person: IDs joined through Additional ID, named by the smallest. An Additional ID the table does
    not list joins nothing (the other visit is not in this release)."""
    parent = {p: p for p in patients}

    def root(p: str) -> str:
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    for p, r in patients.items():
        other = (r.get("Additional ID") or "").strip()
        if other in patients:
            a, b = root(p), root(other)
            if a != b:
                parent[max(a, b, key=int)] = min(a, b, key=int)
    return {p: root(p) for p in patients}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.zip) as z:
        names = z.namelist()
        tables = [n for n in names if n.endswith("/training_data.csv") and n.count("/") == 1]
        if len(tables) != 1:
            raise SystemExit("expected one training_data.csv at the top")
        root = tables[0].split("/")[0]
        with z.open(tables[0]) as f:
            patients = {r["Patient ID"]: r for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))}
        members = []
        for n in names:
            if not n.startswith(f"{root}/training_data/") or not n.endswith(".wav"):
                continue
            m = NAME.match(n.rsplit("/", 1)[1])
            if not m or m.group(1) not in patients:
                raise ValueError(f"a recording the table does not account for: {n}")
            members.append((int(m.group(1)), LOCATIONS.index(m.group(2)), int(m.group(3) or 0), n, m.group(1), m.group(2)))
        members.sort()
        person = people(patients)
        rows, chunks, offset = [], [], 0
        for _, _, _, n, pid, loc in members:
            x = condition(read_wav(z.read(n)))
            p = patients[pid]
            rows.append([len(rows), pid, person[pid], loc, p["Murmur"], p["Outcome"], p["Campaign"], offset, len(x)])
            chunks.append(x.astype(np.float16))
            offset += len(x)
    np.save(a.out / "signals.npy", np.concatenate(chunks))
    with open(a.out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "patient", "person", "location", "murmur", "outcome", "campaign", "offset", "length"])
        w.writerows(rows)
    print(f"{len(rows)} recordings of {len({r[1] for r in rows})} patient IDs, {len({r[2] for r in rows})} people, {offset / RATE / 3600:.2f} hours")


if __name__ == "__main__":
    main()

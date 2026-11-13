#!/usr/bin/env python3
"""Build the medical page's four-hospital federated sample from the UCI Heart Disease files, as FLamby prepares them.

WHY THIS EXISTS (Round 86, the owner's choice of 2026-11-13)

The medical page federated a logistic head across four "hospitals" that were round-robin shards of one corpus (WDBC),
so its federation had no hospital in it. UCI Heart Disease holds four real ones - the Cleveland Clinic, the Hungarian
Institute of Cardiology, the University Hospitals of Zurich and Basel, and the VA Long Beach - under CC BY 4.0, and
FLamby (du Terrail et al., NeurIPS 2022 Datasets and Benchmarks) turned them into a federated benchmark,
Fed-Heart-Disease. This script reproduces FLamby's preparation step for step (flamby/datasets/fed_heart_disease/
dataset.py, read 2026-11-13), so the page federates the benchmark's own centres:

  * per centre: "?" read as missing, columns 10, 11 and 12 (slope, ca, thal) dropped, then every row with a missing
    value dropped; the last column is the label;
  * a 66/34 split per centre with scikit-learn's train_test_split, random_state 43, stratified on the binarised label
    where each class has more than two rows;
  * after the four are pooled, columns 2 (chest-pain type) and 6 (resting ECG) dummy-coded with the first level
    dropped: 8 numeric columns and 5 indicators, 13 features;
  * the label binarised, 0 against any disease;
  * each centre standardised with the mean and the sample standard deviation (ddof 1, plus 1e-9) of its OWN training
    rows, in float32, as FLamby's tensors are.

The four files are read from data-full/raw/heart_disease/ and must match the MD5s FLamby's download script checks.

Run:  .venv/bin/python scripts/data/build_heart_disease_sites.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data-full" / "raw" / "heart_disease"
OUT = REPO / "web" / "public" / "data" / "medical" / "heart_disease_sites.json"

#: FLamby's centre numbering and its download script's MD5s (flamby/datasets/fed_heart_disease/dataset_creation_scripts).
CENTRES = [
    ("cleveland", "Cleveland Clinic Foundation", "2d91a8ff69cfd9616aa47b59d6f843db"),
    ("hungarian", "Hungarian Institute of Cardiology, Budapest", "22e96bee155b5973568101c93b3705f6"),
    ("switzerland", "University Hospitals of Zurich and Basel", "9a87f7577310b3917730d06ba9349e20"),
    ("va", "V.A. Medical Center, Long Beach", "4249d03ca7711e84f4444768c9426170"),
]
#: The UCI column order (age ... num); 10, 11 and 12 are the three FLamby drops.
UCI_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak",
               "slope", "ca", "thal", "num"]
TRAIN_FRACTION = 0.66
SPLIT_SEED = 43


def _centre(path: Path) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    df = pd.read_csv(path, header=None)
    df = df.replace("?", np.nan).drop([10, 11, 12], axis=1).dropna(axis=0)
    df = df.apply(pd.to_numeric)
    x, y = df.iloc[:, :-1], df.iloc[:, -1]
    labels = y.where(y == 0, 1)
    levels = np.unique(labels)
    stratify = labels if len(levels) > 1 and all((labels == lev).sum() > 2 for lev in levels) else None
    n = int(x.shape[0])
    _, test_idx = train_test_split(np.arange(n), test_size=1.0 - TRAIN_FRACTION, train_size=TRAIN_FRACTION,
                                   random_state=SPLIT_SEED, shuffle=True, stratify=stratify)
    is_test = np.isin(np.arange(n), test_idx)
    return x, y, is_test


def build() -> dict:
    frames, labels, centre_of, test_of = [], [], [], []
    for k, (key, _, md5) in enumerate(CENTRES):
        path = RAW / f"processed.{key}.data"
        got = hashlib.md5(path.read_bytes()).hexdigest()
        if got != md5:
            raise SystemExit(f"{path.name}: MD5 {got}, FLamby checks {md5}")
        x, y, is_test = _centre(path)
        frames.append(x)
        labels.append(y)
        centre_of += [k] * len(x)
        test_of += is_test.tolist()
    features = pd.concat(frames, ignore_index=True)
    y_all = pd.concat(labels, ignore_index=True)
    features = pd.get_dummies(features, columns=[2, 6], drop_first=True)
    names = []
    for col in features.columns:
        if isinstance(col, str):                    # a dummy: "<column>_<level>"
            base, level = col.split("_")
            names.append(f"{UCI_COLUMNS[int(base)]}={float(level):g}")
        else:
            names.append(UCI_COLUMNS[int(col)])
    x_all = features.to_numpy(dtype=np.float32)
    y_bin = (y_all.to_numpy() != 0).astype(int)
    centre_of, test_of = np.asarray(centre_of), np.asarray(test_of)
    assert x_all.shape[1] == 13, x_all.shape

    centres = []
    for k, (key, name, _) in enumerate(CENTRES):
        tr = (centre_of == k) & ~test_of
        te = (centre_of == k) & test_of
        mean = x_all[tr].mean(axis=0, dtype=np.float32)
        std = x_all[tr].std(axis=0, ddof=1, dtype=np.float32)
        z = ((x_all - mean) / (std + np.float32(1e-9))).astype(np.float32)
        short = lambda m: [[float(np.format_float_positional(v, unique=True, trim="-")) for v in row] for row in m]  # noqa: E731
        centres.append({
            "key": key, "name": name,
            "train": {"X": short(z[tr]), "y": y_bin[tr].tolist()},
            "test": {"X": short(z[te]), "y": y_bin[te].tolist()},
            "mean": [float(v) for v in mean], "std": [float(v) for v in std],
        })
    return {
        "corpus": "UCI Heart Disease: four hospitals, as FLamby's Fed-Heart-Disease prepares them",
        "provenance": ("The processed Cleveland, Hungarian, Switzerland and VA Long Beach files of the UCI Heart Disease "
                       "dataset (Janosi, Steinbrunn, Pfisterer and Detrano, 1988), each checked against the MD5 "
                       "FLamby's download script checks, prepared as FLamby's Fed-Heart-Disease prepares them "
                       "(du Terrail et al., NeurIPS 2022 Datasets and Benchmarks). Each centre is one real hospital."),
        "licence": "CC-BY-4.0",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": ("Janosi, A., Steinbrunn, W., Pfisterer, M. and Detrano, R. (1988). Heart Disease. UCI Machine "
                        "Learning Repository. https://doi.org/10.24432/C52P4X"),
        "modifications": ("Columns slope, ca and thal dropped and rows with a missing value removed; chest-pain type "
                          "and resting ECG dummy-coded with the first level dropped; the diagnosis binarised (0 against "
                          "1 to 4); each hospital split 66/34 (scikit-learn, random_state 43, stratified where each "
                          "class has more than two rows) and standardised with its own training rows' mean and sample "
                          "standard deviation; values written as the shortest decimals that read back as float32."),
        "feature_names": names,
        "label": "heart disease (num 1 to 4) against none (num 0)",
        "split": "per hospital, 66 percent training and 34 percent test, scikit-learn random_state 43",
        "centres": centres,
    }


def main() -> int:
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    rows = [(c["name"], len(c["train"]["y"]), len(c["test"]["y"]), int(sum(c["train"]["y"]) + sum(c["test"]["y"])))
            for c in payload["centres"]]
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size / 1024:.0f} KB), {sum(r[1] + r[2] for r in rows)} rows")
    for name, tr, te, pos in rows:
        print(f"  {name}: {tr} train, {te} test, {pos} with disease")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

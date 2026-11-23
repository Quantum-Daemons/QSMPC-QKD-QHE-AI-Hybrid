#!/usr/bin/env python3
"""The finance page's tabular pipeline on real transaction data: BitcoinHeist, a related task, not a replacement.

The finance models train on IBM's AMLworld, which is synthetic; no real laundering-labelled corpus under a permissive
licence was found (references/aml-real-data-2026-11.md). BitcoinHeist (UCI dataset 526, CC BY 4.0) is real Bitcoin
data whose labels are RANSOMWARE families against "white" addresses, "white" meaning not known to be ransomware. On the
owner's choice of 2026-11-22 it is a sanity check of the same pipeline on real rows: the finance booster's settings, a
threshold chosen on a validation window, the soft logistic student distilled from the booster, and the student's
linear score under CKKS. It says nothing about laundering, and nothing here is a ransomware detector.

    python scripts/measure_bitcoinheist_check.py --zip <path>/bitcoinheistransomwareaddressdataset.zip \
        -o docs/measurements/bitcoinheist-check-<date>.json

The zip is the one UCI serves (116,104,947 bytes); its SHA-256 is checked before anything is read. Its six address-graph
features (length, weight, count, looped, neighbors, income) are the inputs; the address is an identifier and is
dropped, and year and day order the rows into a time split rather than being features: fit on 2011-2014, choose the
threshold on 2015, test on 2016-2017. 2018 holds 3 positives in 330,003 rows and is reported apart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SHA256 = "cdf3bb34199367b08285cf7f9c1db06c4a262d353aa596055e97db77f4514e5b"
FEATURES = ["length", "weight", "count", "looped", "neighbors", "income"]
SPLIT = {"fit": (2011, 2014), "validation": (2015, 2015), "test": (2016, 2017), "reported_apart": (2018, 2018)}
#: scripts/train/finance_aml.py's teacher settings, on the CPU (the owner asks before each GPU job).
BOOSTER = dict(n_estimators=600, max_depth=8, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
               min_child_weight=4, reg_lambda=2.0, eval_metric="aucpr", tree_method="hist", device="cpu")
CKKS_ROWS = 64


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def _best_threshold(y, score) -> float:
    """Minority-class F1's maximum on the given split, as scripts/train/finance_aml.py chooses it."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y, score)
    if thresholds.size == 0:
        return 0.5
    p, r = precision[:-1], recall[:-1]
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return float(thresholds[int(np.argmax(f1))])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    import pandas as pd
    import xgboost as xgb

    from qsh.fhe.backends import get_backend
    from qsh.models import distil, metrics

    digest = _sha256(args.zip)
    if digest != SHA256:
        raise SystemExit(f"{args.zip}: SHA-256 {digest} is not the file this check was written for ({SHA256})")
    with zipfile.ZipFile(args.zip) as z, z.open("BitcoinHeistData.csv") as f:
        df = pd.read_csv(f)
    df = df.drop(columns=["address"])
    df["y"] = (df["label"] != "white").astype(np.int8)

    parts, counts = {}, {}
    for name, (lo, hi) in SPLIT.items():
        part = df[(df["year"] >= lo) & (df["year"] <= hi)].sort_values(["year", "day"], kind="stable")
        parts[name] = part
        counts[name] = {"years": f"{lo}-{hi}" if lo != hi else str(lo), "rows": int(len(part)),
                        "positives": int(part["y"].sum()), "families": int(part.loc[part["y"] == 1, "label"].nunique())}
    X = {k: v[FEATURES].to_numpy(dtype=float) for k, v in parts.items()}
    y = {k: v["y"].to_numpy(dtype=np.int8) for k, v in parts.items()}

    spw = float((y["fit"] == 0).sum() / max((y["fit"] == 1).sum(), 1))
    t0 = time.time()
    teacher = xgb.XGBClassifier(**BOOSTER, scale_pos_weight=spw, n_jobs=args.threads, random_state=args.seed)
    teacher.fit(X["fit"], y["fit"], eval_set=[(X["validation"], y["validation"])], verbose=False)
    fit_seconds = time.time() - t0
    thr = _best_threshold(y["validation"], teacher.predict_proba(X["validation"])[:, 1])
    p_test = teacher.predict_proba(X["test"])
    teacher_metrics = metrics.with_companion(y["test"], (p_test[:, 1] >= thr).astype(int), p_test,
                                             headline="minority_class_f1")
    teacher_metrics["decision_threshold"] = round(thr, 6)

    # The student sees log1p-scaled, standardised features: the trees do not need it, a linear model does
    # (income and weight span many orders of magnitude).
    mu = np.log1p(X["fit"]).mean(0)
    sd = np.log1p(X["fit"]).std(0) + 1e-12
    Z = {k: (np.log1p(v) - mu) / sd for k, v in X.items()}
    student = distil.distil_linear(teacher.predict_proba(X["fit"]), Z["fit"], seed=args.seed)
    s_val = student.predict_proba(Z["validation"])[:, 1]
    s_thr = _best_threshold(y["validation"], s_val)
    s_test = student.predict_proba(Z["test"])
    student_metrics = metrics.with_companion(y["test"], (s_test[:, 1] >= s_thr).astype(int), s_test,
                                             headline="minority_class_f1")
    student_metrics["decision_threshold"] = round(s_thr, 6)
    agreement = float(((s_test[:, 1] >= s_thr) == (p_test[:, 1] >= thr)).mean())

    # The student's linear score under CKKS on test rows, against the same score in the clear.
    be = get_backend("tenseal")
    w, b = student.coef_[0].astype(float), float(student.intercept_[0])
    rows = Z["test"][np.linspace(0, len(Z["test"]) - 1, CKKS_ROWS).astype(int)]
    enc = np.array([be.decrypt(be.add_plain(be.dot_plain(be.encrypt(r), w), b))[0] for r in rows])
    clear = rows @ w + b
    ckks = {"rows": CKKS_ROWS, "backend": be.name, "max_abs_error": float(np.max(np.abs(enc - clear))),
            "decisions_agree": int(((enc >= np.log(s_thr / (1 - s_thr))) == (clear >= np.log(s_thr / (1 - s_thr)))).sum())}

    p18 = teacher.predict_proba(X["reported_apart"])[:, 1]
    record = {
        "what": "the finance page's tabular pipeline on real Bitcoin data with ransomware labels: a related-task sanity check",
        "not": "a laundering result, a replacement for AMLworld, or a ransomware detector",
        "data": {"name": "BitcoinHeist Ransomware Address Dataset", "source": "UCI Machine Learning Repository, dataset 526",
                 "doi": "10.24432/C5BG8V", "licence": "CC BY 4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
                 "citation": "Akcora, Li, Gel and Kantarcioglu, BitcoinHeist: Topological Data Analysis for Ransomware "
                             "Detection on the Bitcoin Blockchain, arXiv:1906.07852",
                 "file_sha256": digest, "rows": int(len(df)),
                 "labels": "28 ransomware families against 'white' (not known to be ransomware); positive = any family",
                 "features": FEATURES, "dropped": ["address (an identifier)"], "order": ["year", "day"]},
        "split": counts,
        "booster": {**{k: v for k, v in BOOSTER.items()}, "scale_pos_weight": round(spw, 3), "threads": args.threads,
                    "seed": args.seed, "fit_seconds": round(fit_seconds, 1)},
        "booster_test": teacher_metrics,
        "student": {"kind": "soft logistic student distilled from the booster (qsh.models.distil.distil_linear)",
                    "inputs": "log1p, standardised on the fit window", "parameters": int(w.size + 1),
                    "test": student_metrics, "agreement_with_booster": round(agreement, 6)},
        "student_under_ckks": ckks,
        "year_2018": {"rows": int(len(p18)), "positives": int(y["reported_apart"].sum()),
                      "flagged_at_the_test_threshold": int((p18 >= thr).sum())},
        "environment": {"git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                              text=True).stdout.strip(),
                        "python": platform.python_version(), "xgboost": xgb.__version__, "numpy": np.__version__,
                        "date": time.strftime("%Y-%m-%d"), "cpu_threads_available": os.cpu_count()},
    }
    Path(args.output).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"split": counts, "booster_test": teacher_metrics, "student_test": student_metrics,
                      "agreement": round(agreement, 4), "ckks": ckks}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

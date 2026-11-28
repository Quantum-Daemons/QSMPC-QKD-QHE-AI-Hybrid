#!/usr/bin/env python3
"""A detector of PFCP attacks tested on an attack day it never saw, as pre-registered (Round 86, batch 12):
references/telecom-pfcp-held-out-day-preregistration.md.

Reads each attack run's UPF flows at the 120-second timeout from the deposit's archives in place (nothing extracted),
holds each day out in turn, and writes docs/measurements/telecom-pfcp-held-out-day-<date>.json, aggregate figures only.

    QSH_DATA_ROOT=... python scripts/check_pfcp_held_out_day.py
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import re
import sys
import zipfile
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.models import datasets as D  # noqa: E402

PREREG = REPO / "references" / "telecom-pfcp-held-out-day-preregistration.md"
PREREG_SHA256 = "bf469fc90783dd3f7ef9f090b1678c9ecfd49eb921521322996dc0b97d62bebe"
IDENTIFIERS = {"", "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"}
CLIP, FPR, BOOT, SEEDS = 20.0, 0.01, 2000, (0, 1, 2, 3, 4)
ARCHIVES = ("PFCP Session Establishment DoS Attack.zip", "PFCP Session Deletion DoS Attack.zip",
            "PFCP Session Modification DoS Attack.zip")


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def runs(root: Path) -> dict[str, dict]:
    """Each attack run's UPF flows at 120 s, by its folder's name; the date is in the folder's name."""
    out = {}
    for a in ARCHIVES:
        with zipfile.ZipFile(root / a) as z:
            for n in z.namelist():
                # The archives spell the folder "CICFlowMeter Flows", "CiCFlowMeter Flows" and "CiCFlowmeter files".
                low = n.lower()
                if low.endswith("/120-sec-csv/upf.csv") and "/cicflowmeter " in low:
                    run = n.split("/")[1]
                    rows = list(csv.reader(io.TextIOWrapper(z.open(n), encoding="utf-8", errors="replace")))
                    out[run] = {"archive": a, "member": n, "head": rows[0], "rows": rows[1:]}
    return out


def matrix(run: dict, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """The run's flows in file order; a value CICFlowMeter writes as empty, infinite or NaN is 0."""
    head = run["head"]
    idx = [head.index(c) for c in names]
    lab = head.index("Label")
    rows = [r for r in run["rows"] if len(r) == len(head)]
    X = np.array([[float(r[i]) if r[i] not in ("", "inf", "-inf", "nan", "NaN", "Infinity", "-Infinity") else 0.0
                   for i in idx] for r in rows], dtype=np.float64)
    X[~np.isfinite(X)] = 0.0
    y = np.array([1 if r[lab].strip().lower() == "malicious" else 0 for r in rows])
    return X, y


def auc(score: np.ndarray, y: np.ndarray) -> float | None:
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return None
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="stable"), kind="stable") + 1.0
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def at_fpr(score_val, y_val, score_te, y_te) -> dict:
    neg = np.sort(score_val[y_val == 0])
    thr = float(neg[int(np.ceil((1 - FPR) * len(neg))) - 1]) if len(neg) else float("inf")
    return {"threshold_from_validation": thr, "detection_rate": float(np.mean(score_te[y_te == 1] > thr)),
            "false_positive_rate": float(np.mean(score_te[y_te == 0] > thr))}


def _ci(vals: list[float]) -> list[float] | None:
    return [round(float(np.percentile(vals, 2.5)), 6), round(float(np.percentile(vals, 97.5)), 6)] if vals else None


def boot(score: np.ndarray, y: np.ndarray, thr: float) -> dict:
    """95 percent intervals over the held-out day's flows (2,000 resamples, default_rng(0)) for the AUC and for the
    detection and false-positive rates at the threshold fixed on validation. A resample without a class adds nothing
    to the figures that need it."""
    rng = np.random.default_rng(0)
    a, d, f = [], [], []
    for _ in range(BOOT):
        i = rng.integers(0, len(y), len(y))
        s, yy = score[i], y[i]
        v = auc(s, yy)
        if v is not None:
            a.append(v)
        if (yy == 1).any():
            d.append(float(np.mean(s[yy == 1] > thr)))
        if (yy == 0).any():
            f.append(float(np.mean(s[yy == 0] > thr)))
    return {"auc_ci95": _ci(a), "detection_rate_ci95": _ci(d), "false_positive_rate_ci95": _ci(f)}


def arm(sv: np.ndarray, yv: np.ndarray, st: np.ndarray, yt: np.ndarray) -> dict:
    rates = at_fpr(sv, yv, st, yt)
    return {"auc": auc(st, yt), **rates, **boot(st, yt, rates["threshold_from_validation"])}


def main() -> int:
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got[:12]}, not the committed {PREREG_SHA256[:12]}: run nothing")
    root = Path(D.data_root()) / "raw" / "5gc-pfcp"
    R = runs(root)
    if len(R) != 4:
        raise SystemExit(f"{len(R)} attack runs found, not the four the pre-registration names: run nothing")
    heads = [r["head"] for r in R.values()]
    shared = set.intersection(*(set(h) for h in heads))
    not_shared = sorted(set.union(*(set(h) for h in heads)) - shared)
    T = _module("telecom_nidd", REPO / "scripts" / "train" / "telecom_nidd.py")
    S = _module("telecom_nidd_student", REPO / "scripts" / "train" / "telecom_nidd_student.py")
    base = [c for c in heads[0] if c in shared and c not in IDENTIFIERS and c != "Label"]
    days = {}
    for held in sorted(R):
        train = [n for n in R if n != held]
        Xs, ys = zip(*(matrix(R[n], base) for n in train), strict=True)
        names = [c for j, c in enumerate(base) if np.ptp(np.concatenate([x[:, j] for x in Xs])) > 0]
        cols = [base.index(c) for c in names]
        fit_X, fit_y, val_X, val_y = [], [], [], []
        for x, yy in zip(Xs, ys, strict=True):
            cut = int(round(0.8 * len(yy)))
            fit_X.append(x[:cut, cols]); fit_y.append(yy[:cut]); val_X.append(x[cut:, cols]); val_y.append(yy[cut:])
        Xf, yf, Xv, yv = np.concatenate(fit_X), np.concatenate(fit_y), np.concatenate(val_X), np.concatenate(val_y)
        Xt, yt = matrix(R[held], base)
        Xt = Xt[:, cols]
        if not (yt == 1).any() or not (yt == 0).any():
            days[held] = {"scorable": False}
            continue
        mu, sd = Xf.mean(0), Xf.std(0)
        sd[sd == 0] = 1.0
        z = lambda a: np.clip((a - mu) / sd, -CLIP, CLIP)  # noqa: E731
        teacher = []
        for seed in SEEDS:
            # The 5G-NIDD teacher's recipe, its hyperparameters unchanged, on the CPU as pre-registered: the recipe
            # names the GPU, which this check does not use.
            m = T._booster(SimpleNamespace(n_estimators=300, max_depth=8, seed=seed))
            m.set_params(device="cpu")
            m.fit(Xf, yf)
            sv, st = m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1]
            teacher.append({"seed": seed, **arm(sv, yv, st, yt)})
        onehot = np.eye(2)
        fits = {l2: S.fit(z(Xf), onehot[yf], l2) for l2 in S.L2_GRID}
        loss = {l2: S.log_loss(S.softmax(z(Xv) @ W.T + b), onehot[yv]) for l2, (W, b) in fits.items()}
        l2 = min(loss, key=lambda k: (loss[k], k))
        W, b = fits[l2]
        lv, lt = z(Xv) @ W.T + b, z(Xt) @ W.T + b
        from scipy.optimize import minimize_scalar

        temperature = float(np.exp(minimize_scalar(lambda u: S.log_loss(S.softmax(lv / np.exp(u)), onehot[yv]),
                                                   bounds=(-4, 4), method="bounded").x))
        lv, lt = lv / temperature, lt / temperature
        sv, st = lv[:, 1] - lv[:, 0], lt[:, 1] - lt[:, 0]
        student = {"l2": l2, "temperature": temperature, **arm(sv, yv, st, yt)}

        def spread(key: str, arms: list[dict] = teacher) -> dict:
            vals = [a[key] for a in arms if a[key] is not None]
            return {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}

        days[held] = {"scorable": True, "train_days": train, "inputs": len(cols), "fit_rows": int(len(yf)),
                      "validation_rows": int(len(yv)), "held_out_rows": int(len(yt)), "held_out_malicious": int(yt.sum()),
                      "teacher": {"per_seed": teacher, "auc": spread("auc"), "detection_rate": spread("detection_rate"),
                                  "false_positive_rate": spread("false_positive_rate")},
                      "student": student}
    record = {"preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": PREREG_SHA256},
              "deposit": {"doi": "10.21227/e0gj-ev03", "zenodo": "7888347", "licence": "CC-BY-4.0",
                          "archives_sha256": {a: hashlib.sha256((root / a).read_bytes()).hexdigest() for a in ARCHIVES}},
              "runs": {n: {"archive": r["archive"], "rows": len(r["rows"]),
                           "rows_dropped_malformed": sum(len(x) != len(r["head"]) for x in r["rows"])} for n, r in R.items()},
              "columns_not_in_every_run": not_shared, "held_out_days": days}
    out = REPO / "docs" / "measurements" / f"telecom-pfcp-held-out-day-{date.today().isoformat()}.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({d: (v.get("student", {}).get("auc"), v.get("teacher", {}).get("auc", {}).get("mean")) for d, v in days.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

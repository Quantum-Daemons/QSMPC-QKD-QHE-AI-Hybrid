#!/usr/bin/env python3
"""The DeFi QCNN's inputs, exactly as pre-registered (Round 86, batch 8).

`references/defi-qcnn-aave-v3-preregistration.md` and its addendum fixed everything below before any label was
computed: FinSurvival's Borrow-to-Liquidated table of Spadea and Seneviratne's Aave v3 archive; rows with `timeDiff` at
or below 0 left out, then rows censored at or before 30 days; the label 1 when `timeDiff` is at most 30 days; windows by
`timestamp`, 70/15/15; the 80 numeric candidate inputs, standardised by the fit window and clipped at 10; the eight with
the largest |AUC - 0.5| on the fit window, ties by name, the first four for the four-wire arm; the fit set (every
positive and four negatives each when positives are under a quarter, else the whole fit window, at most 20,000 rows
drawn stratified at seed 0); the validation and test windows whole.

Writes `$QSH_DATA_ROOT/artifacts/defi_qcnn_inputs.npz` (X: the eight inputs; y; part: 0 fit set, 1 validation, 2 test),
`defi_qcnn_all_inputs.npz` (every candidate input of the whole fit window, validation and test, for the reference
booster) and `defi_qcnn_inputs.json` (the counts, the inputs and their scores). The QCNN trainer's `defi_aave_v3`
corpus reads the first. No account, transaction or other identifier is written anywhere.

Usage: QSH_DATA_ROOT=<data root> python -I scripts/data/defi_qcnn_inputs.py --zip <path to data.zip>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qsh.models import datasets  # noqa: E402

PREREG = REPO / "references" / "defi-qcnn-aave-v3-preregistration.md"
PREREG_SHA256 = "af2cc309b6aa3d69b45c3474a6c8d39ce6eadd332baa87a502a98e2166902110"  # with its two addenda, 9031341
ZIP_BYTES, ZIP_MD5 = 13_863_498_879, "d9f43ce5010d10a1f836719e795c6db7"
TABLE = "data/Borrow/Liquidated/data.csv"
HORIZON_S = 30 * 86_400
#: The addendum's columns left out: the label, the events' names, the identifiers, the raw timestamp, the non-numeric.
LABEL, EVENTS, IDS, TIME, TEXT = (["timeDiff", "status"], ["Index Event", "Outcome Event", "type"],
                                  ["user", "pool", "reserve"], ["timestamp"],
                                  ["coinType", "userCoinTypeMode", "userReserveMode"])
BOOLEANS = ("isWeekend", "userIsNew")
K, NEGATIVES_PER_POSITIVE, CAP, CLIP, QUARTER = 8, 4, 20_000, 10.0, 0.25


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zip", type=Path, required=True)
    args = ap.parse_args()
    started = time.time()
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got}, not the committed {PREREG_SHA256}: run nothing")
    if args.zip.stat().st_size != ZIP_BYTES or md5_of(args.zip) != ZIP_MD5:
        raise SystemExit(f"{args.zip} is not the recorded archive (size or MD5): run nothing")
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    with zipfile.ZipFile(args.zip) as z, z.open(TABLE) as f:
        df = pd.read_csv(f, low_memory=False)
    rows_read = len(df)
    left_out = LABEL + EVENTS + IDS + TIME + TEXT
    missing_cols = [c for c in left_out if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"the table lacks columns the addendum names: {missing_cols}")
    candidates = [c for c in df.columns if c not in left_out]
    for c in BOOLEANS:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].map({"TRUE": 1, "FALSE": 0, "True": 1, "False": 0, True: 1, False: 0})
    unparsed = []
    for c in candidates:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().sum() > df[c].isna().sum():
            unparsed.append(c)
        else:
            df[c] = v.astype(np.float64)
    candidates = [c for c in candidates if c not in unparsed]
    with_missing = {c: int(df[c].isna().sum()) for c in candidates if df[c].isna().any()}
    if with_missing:
        raise SystemExit(f"candidate inputs with missing values, which the pre-registration does not cover: {with_missing}")

    # ---- the label, by FinSurvival's rules as the addendum states them
    t = pd.to_numeric(df["timeDiff"], errors="raise").to_numpy(np.float64)
    s = pd.to_numeric(df["status"], errors="raise").to_numpy(np.int64)
    invalid = t <= 0
    censored_early = (~invalid) & (t <= HORIZON_S) & (s == 0)
    keep = ~(invalid | censored_early)
    y_all = (t <= HORIZON_S).astype(np.int8)
    ts = pd.to_numeric(df["timestamp"], errors="raise").to_numpy(np.float64)
    idx = np.flatnonzero(keep)
    order = idx[np.argsort(ts[idx], kind="stable")]
    n = len(order)
    n_fit, n_val = (70 * n) // 100, (15 * n) // 100
    w_fit, w_val, w_te = order[:n_fit], order[n_fit:n_fit + n_val], order[n_fit + n_val:]
    X_all = df[candidates].to_numpy(np.float64)
    y_fit, y_val, y_te = y_all[w_fit], y_all[w_val], y_all[w_te]

    centre = X_all[w_fit].mean(axis=0)
    scale = X_all[w_fit].std(axis=0)
    scale[scale == 0] = 1.0
    zf, zv, zt = (np.clip((X_all[w] - centre) / scale, -CLIP, CLIP) for w in (w_fit, w_val, w_te))
    print(f"  {rows_read:,} rows; {int(invalid.sum()):,} invalid, {int(censored_early.sum()):,} censored early; "
          f"fit {len(y_fit):,} ({int(y_fit.sum()):,} liquidated), validation {len(y_val):,} ({int(y_val.sum()):,}), "
          f"test {len(y_te):,} ({int(y_te.sum()):,}); {len(candidates)} inputs", flush=True)

    score = {c: abs(float(roc_auc_score(y_fit, zf[:, j])) - 0.5) for j, c in enumerate(candidates)}
    chosen = sorted(score, key=lambda c: (-score[c], c))[:K]
    cols = [candidates.index(c) for c in chosen]
    for c in chosen:
        print(f"    {c:40s} |AUC - 0.5| {score[c]:.6f}")

    rule = "whole fit window"
    take = np.arange(len(y_fit))
    if y_fit.mean() < QUARTER:
        pos, neg = np.flatnonzero(y_fit == 1), np.flatnonzero(y_fit == 0)
        draw = np.random.default_rng(0).choice(neg, size=min(len(neg), NEGATIVES_PER_POSITIVE * len(pos)), replace=False)
        take = np.sort(np.concatenate([pos, draw]))
        rule = "every positive and four negatives each, seed 0"
    if len(take) > CAP:
        take, _ = train_test_split(take, train_size=CAP, stratify=y_fit[take], random_state=0)
        take = np.sort(take)
        rule += f", then {CAP:,} drawn stratified at seed 0"
    X = np.vstack([zf[take][:, cols], zv[:, cols], zt[:, cols]]).astype(np.float32)
    y = np.concatenate([y_fit[take], y_val, y_te]).astype(np.int8)
    part = np.concatenate([np.zeros(len(take), np.int8), np.ones(len(y_val), np.int8), np.full(len(y_te), 2, np.int8)])
    out = datasets.artifacts("defi_qcnn_inputs.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, y=y, part=part)
    np.savez(datasets.artifacts("defi_qcnn_all_inputs.npz"), fit=zf.astype(np.float32), y_fit=y_fit,
             val=zv.astype(np.float32), y_val=y_val, test=zt.astype(np.float32), y_test=y_te)
    meta = {"preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": got},
            "archive": {"bytes": ZIP_BYTES, "md5": ZIP_MD5, "table": TABLE,
                        "doi": "10.6084/m9.figshare.31058368.v1", "licence": "CC-BY-4.0"},
            "horizon_days": 30, "features": chosen,
            "abs_auc_minus_half_on_fit": {c: round(score[c], 6) for c in chosen},
            "all_scores_on_fit": {c: round(v, 6) for c, v in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))},
            "candidates": candidates, "left_out": {"label": LABEL, "events": EVENTS, "identifiers": IDS, "time": TIME,
                                                   "non_numeric": TEXT, "did_not_parse": unparsed},
            "centre": [float(centre[j]) for j in cols], "scale": [float(scale[j]) for j in cols], "clip": CLIP,
            "fit_set_rule": rule,
            "counts": {"rows_read": int(rows_read), "invalid_time": int(invalid.sum()),
                       "censored_at_or_before_horizon": int(censored_early.sum()), "labelled": int(n),
                       "fit_window": int(len(y_fit)), "fit_window_positive": int(y_fit.sum()), "fit_set": int(len(take)),
                       "fit_set_positive": int(y_fit[take].sum()), "validation": int(len(y_val)),
                       "validation_positive": int(y_val.sum()), "test": int(len(y_te)), "test_positive": int(y_te.sum())},
            "windows_by_timestamp": {"fit_last": float(ts[w_fit[-1]]), "validation_first": float(ts[w_val[0]]),
                                     "validation_last": float(ts[w_val[-1]]), "test_first": float(ts[w_te[0]]),
                                     "test_last": float(ts[w_te[-1]])},
            "wall_clock_s": round(time.time() - started, 1)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out} ({X.shape[0]:,} rows x {X.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

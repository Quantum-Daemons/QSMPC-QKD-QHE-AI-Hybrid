#!/usr/bin/env python3
"""The finance QCNN's AMLworld inputs, exactly as pre-registered (Round 86, batch 6).

references/finance-qcnn-amlworld-preregistration.md fixed everything below before it ran: the 114 inputs of
`finance_aml.model_inputs` on the published MLP's temporal windows, standardised by the fit window and clipped at 10
(F7's and F7b's preparation, `finance_aml_mlp.scaling_of` and `_scaled`); the eight inputs with the largest
|AUC - 0.5| on the fit window, ties by name, the first four of them for the four-wire arm; every laundering transaction
of the fit window and a seeded sample (seed 0) of four others for each; the validation and test windows whole.

Writes `$QSH_DATA_ROOT/artifacts/aml_qcnn_inputs.npz` (X: the eight standardised inputs; y; part: 0 fit, 1 validation,
2 test) and its `.json` (the inputs, their scores, the counts). The QCNN trainer's `finance_amlworld` corpus reads it.

Usage: QSH_DATA_ROOT=<data root> python -I scripts/data/aml_qcnn_inputs.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "train"))

from qsh.models import datasets  # noqa: E402

PREREG = REPO / "references" / "finance-qcnn-amlworld-preregistration.md"
PREREG_SHA256 = "5a411b05d7e1984306d08397d95b147ab2a09796a4bfa4a6b624cf515253edd0"
K = 8
NEGATIVES_PER_POSITIVE = 4


def main() -> int:
    started = time.time()
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got}, not the committed {PREREG_SHA256}: run nothing")
    from finance_aml import causal_account_features, load_sorted, model_inputs, temporal_windows
    from finance_aml_mlp import CLIP, scaling_of
    from sklearn.metrics import roc_auc_score

    df = load_sorted()
    fit_w, val_w, te_w, tr_w = temporal_windows(df)
    y_fit, y_val, y_te = (d["label"].to_numpy(dtype=np.int8) for d in (fit_w, val_w, te_w))
    (x_fit, x_val, x_te), names, _ = model_inputs(df, fit_w, val_w, te_w, tr_w, causal_account_features(df))
    centre, scale = scaling_of("standard", x_fit)
    # `finance_aml_mlp._scaled`'s arithmetic in numpy, so nothing reaches a GPU: clip((x - centre) / scale) at CLIP.
    zf, zv, zt = (np.clip((x - centre) / scale, -CLIP, CLIP).astype(np.float32) for x in (x_fit, x_val, x_te))
    print(f"  fit {len(y_fit):,} ({int(y_fit.sum())} laundering), validation {len(y_val):,}, test {len(y_te):,} "
          f"({int(y_te.sum())}); {zf.shape[1]} inputs", flush=True)

    # The inputs, chosen on the fit window only: each alone, |AUC - 0.5| against the label, ties broken by name.
    score = {n: abs(float(roc_auc_score(y_fit, zf[:, j])) - 0.5) for j, n in enumerate(names)}
    chosen = sorted(score, key=lambda n: (-score[n], n))[:K]
    cols = [names.index(n) for n in chosen]
    for n in chosen:
        print(f"    {n:40s} |AUC - 0.5| {score[n]:.6f}")

    pos = np.flatnonzero(y_fit == 1)
    neg = np.flatnonzero(y_fit == 0)
    take = np.sort(np.concatenate([pos, np.random.default_rng(0).choice(neg, size=NEGATIVES_PER_POSITIVE * len(pos),
                                                                        replace=False)]))
    X = np.vstack([zf[take][:, cols], zv[:, cols], zt[:, cols]]).astype(np.float32)
    y = np.concatenate([y_fit[take], y_val, y_te]).astype(np.int8)
    part = np.concatenate([np.zeros(len(take), np.int8), np.ones(len(y_val), np.int8), np.full(len(y_te), 2, np.int8)])
    out = datasets.artifacts("aml_qcnn_inputs.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, y=y, part=part)
    meta = {"preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": got},
            "features": chosen, "abs_auc_minus_half_on_fit": {n: round(score[n], 6) for n in chosen},
            "all_scores_on_fit": {n: round(v, 6) for n, v in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))},
            "centre": [float(centre[j]) for j in cols], "scale": [float(scale[j]) for j in cols], "clip": float(CLIP),
            "counts": {"fit_window": int(len(y_fit)), "fit_window_laundering": int(len(pos)), "fit_sample": int(len(take)),
                       "validation": int(len(y_val)), "validation_laundering": int(y_val.sum()),
                       "test": int(len(y_te)), "test_laundering": int(y_te.sum())},
            "wall_clock_s": round(time.time() - started, 1)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out} ({X.shape[0]:,} rows x {X.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""The published IBM Multi-PNA+EU checkpoints' calibration on the test window: a reliability table and the ECE.

On the owner's choice (2026-11-24: "compute it later", in batch 4). The probabilities are the ones the published record
was scored from: `scripts/train/ibm_multignn/dump_logits.py` wrote each seed's validation and test windows with the
fit-window batch-norm statistics (`pna_project_fitbn_seed<k>/{val,test}.npz`), and this script refuses a dump whose test
AUPRC, row count or positive count differs from `models/metrics.json`'s `finance_aml.gnn` seed, so no new inference runs
and no GPU is needed. Nothing published changes: the record goes to docs/measurements, not to the model's own record,
whose Hub card would otherwise lag it.

For each seed, on the scored test edges:

- the Platt map fitted on validation (`qsh.models.calibration.fit_platt`, on the logit, the finance records' method),
  and the Brier score, log loss and CORP decomposition (Dimitriadis, Gneiting and Jordan, PNAS 2021) raw and calibrated;
- a reliability table in ten equal-width bins of predicted probability: rows, laundering rows, mean prediction, observed
  rate and its 95 % Wilson interval;
- the expected calibration error, sum over bins of (rows in bin / rows) x |observed rate - mean prediction|, in ten and
  fifteen equal-width bins (Guo et al., ICML 2017) and in fifteen equal-mass bins (Nixon et al., CVPR Workshops 2019),
  raw and calibrated. At a base rate of 0.0015 almost every row sits in the first equal-width bin, so the bin-free CORP
  miscalibration term is reported beside it, as in the other finance records.

    python scripts/measure_finance_gnn_calibration.py --dumps <qsh-gnn-work> -o docs/measurements/finance-gnn-calibration-<date>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.models import calibration as cal  # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def reliability(p: np.ndarray, y: np.ndarray, edges: np.ndarray) -> list[dict]:
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        m = idx == b
        n, k = int(m.sum()), int(y[m].sum())
        row = {"from": round(float(edges[b]), 6), "to": round(float(edges[b + 1]), 6), "rows": n, "laundering": k}
        if n:
            lo, hi = cal.wilson_interval(k, n)
            row.update({"mean_prediction": round(float(p[m].mean()), 8), "observed_rate": round(k / n, 8),
                        "observed_ci95": [round(lo, 8), round(hi, 8)]})
        out.append(row)
    return out


def ece(p: np.ndarray, y: np.ndarray, edges: np.ndarray) -> float:
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            total += m.mean() * abs(float(y[m].mean()) - float(p[m].mean()))
    return total


def ece_equal_mass(p: np.ndarray, y: np.ndarray, bins: int) -> float:
    order = np.argsort(p, kind="stable")
    total = 0.0
    for part in np.array_split(order, bins):
        if len(part):
            total += len(part) / len(p) * abs(float(y[part].mean()) - float(p[part].mean()))
    return total


def summary(p: np.ndarray, y: np.ndarray) -> dict:
    ten, fifteen = np.linspace(0, 1, 11), np.linspace(0, 1, 16)
    return {"brier": round(cal.brier(p, y), 8), "log_loss": round(cal.log_loss(p, y), 8),
            "corp": {k: (round(v, 8) if isinstance(v, float) else v) for k, v in cal.corp_decomposition(p, y).items()},
            "ece_10_equal_width": round(ece(p, y, ten), 8), "ece_15_equal_width": round(ece(p, y, fifteen), 8),
            "ece_15_equal_mass": round(ece_equal_mass(p, y, 15), 8),
            "reliability_10_equal_width": reliability(p, y, ten)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dumps", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()
    from sklearn.metrics import average_precision_score

    rec = json.loads((REPO / "models" / "metrics.json").read_text())["finance_aml.gnn"]
    seeds, sources = {}, {}
    for s in rec["seeds"]:
        k = s["seed"]
        vf, tf = (args.dumps / f"pna_project_fitbn_seed{k}" / f"{w}.npz" for w in ("val", "test"))
        v, t = np.load(vf), np.load(tf)
        yv, yt = v["labels"].astype(int), t["labels"].astype(int)
        pv, pt = v["probs"].astype(np.float64), t["probs"].astype(np.float64)
        ap_ = round(float(average_precision_score(yt, pt)), 6)
        if ap_ != s["test_auprc"] or len(yt) != s["n_test_scored"] or int(yt.sum()) != s["test_positives_scored"]:
            raise SystemExit(f"seed {k}: the dump (AUPRC {ap_}, {len(yt)} rows, {int(yt.sum())} positive) is not the "
                             f"published record's ({s['test_auprc']}, {s['n_test_scored']}, {s['test_positives_scored']})")
        sources[f"pna_project_fitbn_seed{k}/val.npz"] = _sha(vf)
        sources[f"pna_project_fitbn_seed{k}/test.npz"] = _sha(tf)
        platt = cal.fit_platt(pv, yv, on_logit=True)
        pc = platt.apply(pt)
        seeds[str(k)] = {"test_rows_scored": int(len(yt)), "test_laundering": int(yt.sum()), "test_auprc": ap_,
                         "decision_threshold": s["decision_threshold"],
                         "platt_fitted_on_validation": {"a": platt.a, "b": platt.b, "validation_rows": int(len(yv)),
                                                        "validation_laundering": int(yv.sum())},
                         "raw": summary(pt, yt), "calibrated": summary(pc, yt)}
        r, c = seeds[str(k)]["raw"], seeds[str(k)]["calibrated"]
        print(f"  seed {k}: ECE(15, equal mass) raw {r['ece_15_equal_mass']:.2e} calibrated {c['ece_15_equal_mass']:.2e}; "
              f"CORP MCB raw {r['corp']['mcb']:.2e} calibrated {c['corp']['mcb']:.2e}", flush=True)
    record = {
        "what": "calibration of the published IBM Multi-PNA+EU checkpoints (finance_aml.gnn) on the test window",
        "model": "finance_aml.gnn", "selected_seed": rec["selected_seed"],
        "probabilities": "the published record's own, as dump_logits.py wrote them with fit-window batch-norm "
                         "statistics; each dump's test AUPRC, rows and positives checked against models/metrics.json",
        "window": "the project's temporal test window, the edges the neighbour sampler scored (1,523,504 transfers, of "
                  "which a seed scores about 1,522,620)",
        "seeds": seeds, "sources": sources,
        "methods": {"platt": "qsh.models.calibration.fit_platt(on_logit=True), fitted on validation, scored on test",
                    "ece": "sum over bins of (rows in bin / rows) x |observed rate - mean prediction|; equal-width "
                           "(Guo et al., ICML 2017) and equal-mass (Nixon et al., CVPR Workshops 2019) bins",
                    "corp": "Dimitriadis, Gneiting and Jordan, PNAS 118 e2016191118 (2021): pool-adjacent-violators, "
                            "no bin count chosen; MCB is its miscalibration term"},
        "date": time.strftime("%Y-%m-%d"),
    }
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Fit an integer risk scorecard for AML, optimised on net benefit, evaluated exactly under BFV.

WHY A SCORECARD, AND WHY HERE
    Every other model in this project is fitted to a surrogate loss and then given an operating
    point afterwards - two approximations to the quantity actually cared about. Chi and Birbil
    (arXiv:2604.04241) make the fit itself the metric: maximise net benefit across a RANGE of
    decision thresholds, over integer coefficients, with an l0 penalty for sparsity.

    Finance is the natural home for it. Integer scorecards are the industry form for credit and AML
    risk - they are what a model-risk function and a regulator will actually read - and at 1 laundering
    transaction in 981 the threshold choice dominates, which is exactly what net benefit prices.

WHY IT MATTERS CRYPTOGRAPHICALLY
    The output is a short vector of integers. BFV evaluates integer arithmetic EXACTLY, so the
    encrypted score equals the plaintext score bit for bit, rather than to a precision bound as a
    CKKS student can only promise. That is a different and stronger claim about the encrypted tier,
    and it needs no scheme switching - see references/he-scheme-selection.md for why a CKKS/BFV
    hybrid is not the route (believed as hard as bootstrapping, and unreachable from a browser).

BINNING
    The scorecard consumes integer BIN INDICES, not raw floats. Bins are quantile edges fitted on
    the training window only and published with the model, so the browser reproduces them exactly
    and there is no "the client must guess the preprocessing" gap - the same discipline the other
    published models follow.

Usage:
    export QSH_DATA_ROOT=$PWD/data-full
    python scripts/train/finance_scorecard.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import finance_aml as F  # noqa: E402

from qsh.models import datasets, metrics, train  # noqa: E402
from qsh.models.scorecard import fit_scorecard  # noqa: E402

#: Bins per feature. Ten is the scorecard convention (deciles) and keeps the score range small,
#: which matters because BFV's plaintext modulus has to hold the score without wrapping.
N_BINS = 10

#: The threshold range the fit integrates over. Chosen from the decision context, not the data: at a
#: 1-in-981 base rate an alerting threshold above a few percent means almost nothing is reviewed, so
#: this brackets "review a 1-in-500 suspicion" to "review a 1-in-20 suspicion".
THRESHOLDS = np.array([0.002, 0.005, 0.01, 0.02, 0.05])


def quantile_bins(Xfit: np.ndarray, n_bins: int = N_BINS):
    """Edges from the TRAINING window only. Returns (edges, transform)."""
    edges = [np.unique(np.quantile(Xfit[:, j], np.linspace(0, 1, n_bins + 1)[1:-1]))
             for j in range(Xfit.shape[1])]

    def to_bins(X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.searchsorted(edges[j], X[:, j], side="right")
                                for j in range(X.shape[1])]).astype(np.int64)

    return edges, to_bins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--coef-range", type=int, default=8, help="coefficients in {-R..R}")
    ap.add_argument("--max-features", type=int, default=12,
                    help="sparsity cap. A scorecard a human cannot read has given up its main "
                         "advantage over the booster it is competing with.")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("models/finance_aml_scorecard"))
    args = ap.parse_args()

    started = time.time()
    train.seed_everything(args.seed)

    print("loading AMLworld HI-Small ...")
    df = datasets.load_amlworld().sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df) * (1.0 - F.TEST_FRACTION))
    val_cut = int(cut * (1.0 - F.VAL_FRACTION))
    fit, val, te, tr = df.iloc[:val_cut], df.iloc[val_cut:cut], df.iloc[cut:], df.iloc[:cut]
    cz = F.causal_account_features(df)
    Xfit = F.graph_features(fit, causal=cz)
    Xval = F.graph_features(val, fit_on=fit, causal=cz)
    Xte = F.graph_features(te, fit_on=tr, causal=cz)
    yfit = fit["label"].to_numpy(dtype=np.int64)
    yval = val["label"].to_numpy(dtype=np.int64)
    yte = te["label"].to_numpy(dtype=np.int64)
    print(f"  fit {Xfit.shape} ({yfit.sum()} pos) | val {len(yval):,} | test {len(yte):,} "
          f"({yte.sum()} pos, 1 in {len(yte)//max(yte.sum(),1)})")

    edges, to_bins = quantile_bins(Xfit)
    Bfit, Bval, Bte = to_bins(Xfit), to_bins(Xval), to_bins(Xte)
    print(f"  binned to {N_BINS} quantiles per feature, edges fitted on the training window only")

    print(f"  fitting scorecard: coefficients in {{-{args.coef_range}..{args.coef_range}}}, "
          f"<= {args.max_features} features, {len(THRESHOLDS)} thresholds ...")
    t0 = time.time()
    sc = fit_scorecard(Bfit, yfit, thresholds=THRESHOLDS, coef_range=args.coef_range,
                       max_features=args.max_features, rounds=args.rounds, seed=args.seed)
    print(f"    {time.time()-t0:.0f}s -> {sc.n_nonzero} non-zero coefficients, "
          f"AUNBC(fit) {sc.aunbc:+.6f}, method {sc.method}")

    # Operating point on VALIDATION, never on test, and AT A THRESHOLD FROM THE DECLARED RANGE.
    # Evaluating net benefit at p=0.5 here would price one false positive at one true positive,
    # which at a 1-in-656 base rate no model can win and which is not the exchange rate this
    # scorecard was fitted for. The fit integrated 0.002-0.05; the report must use the same scale.
    P_REPORT = float(np.median(THRESHOLDS))
    s_val, s_te = sc.score(Bval), sc.score(Bte)
    cand = np.unique(s_val)
    nb = [metrics.net_benefit(yval, (s_val >= c).astype(float), P_REPORT) for c in cand]
    cut_star = int(cand[int(np.argmax(nb))])
    pred = (s_te >= cut_star).astype(int)

    f1 = metrics.minority_class_f1(yte, pred)
    ap_ = metrics.auprc(yte, s_te.astype(float))
    nb_model = metrics.net_benefit(yte, (s_te >= cut_star).astype(float), P_REPORT)
    nb_all = metrics.net_benefit(yte, np.ones_like(yte, dtype=float), P_REPORT)
    print(f"\n  scorecard on test: minority-F1 {f1:.4f}  AUPRC {ap_:.4f}  cut {cut_star}")
    print(f"    net benefit at p={P_REPORT:g} (median of the fitted range): {nb_model:+.6f} "
          f"vs treat-all {nb_all:+.6f} -> margin {nb_model - max(nb_all, 0.0):+.6f}")
    print("    comparators: CKKS student 0.0614, XGBoost teacher 0.1519, published 0.6323")

    nz = np.flatnonzero(sc.w)
    print("\n  the card, which is the point of the form:")
    for j in nz:
        print(f"    {F.FEATURES[j]:28s} bin x {int(sc.w[j]):+d}")
    print(f"    {'intercept':28s}     {int(sc.b):+d}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scorecard.json").write_text(json.dumps({
        "kind": "integer-scorecard",
        "note": ("Integer risk scorecard fitted by maximising net benefit across a range of "
                 "decision thresholds (arXiv:2604.04241), not by fitting a surrogate loss and "
                 "choosing an operating point afterwards. Consumes integer BIN INDICES, so a BFV "
                 "evaluation reproduces the plaintext score exactly rather than to a precision "
                 "bound."),
        "features": F.FEATURES,
        "bin_edges": [e.tolist() for e in edges],
        "n_bins": N_BINS,
        "w": sc.w.tolist(), "b": int(sc.b), "cut": cut_star,
        "thresholds": THRESHOLDS.tolist(),
        "method": sc.method,
        "input": {"preprocessing": ["bin each feature by bin_edges (searchsorted, side=right)"],
                  "dtype": "int64",
                  "layout": "one integer bin index per feature, in `features` order"},
    }, indent=2) + "\n")
    print(f"\n  wrote {args.out}/scorecard.json")

    train.record_metrics("finance_aml.scorecard", {
        "minority_class_f1": round(float(f1), 6),
        "auprc": round(float(ap_), 6),
        "net_benefit": round(float(nb_model), 8),
        "net_benefit_threshold": P_REPORT,
        "net_benefit_treat_all": round(float(nb_all), 8),
        "n_nonzero_coefficients": int(sc.n_nonzero),
        "n_features_available": len(F.FEATURES),
        "coefficient_range": args.coef_range,
        "decision_cut": cut_star,
        "aunbc_fit": round(float(sc.aunbc), 8),
        "fit_method": sc.method,
        "n_test": int(len(yte)),
        "exact_under_bfv": True,
        "metric_note": (
            "Fitted by maximising net benefit across five decision thresholds rather than a "
            "surrogate loss. `fit_method` is 'coordinate-search': the objective it maximises is the "
            "exact AUNBC, but optimality is NOT claimed - the exact integer program over all "
            "(sample x threshold) indicators is not tractable at this row count, and reporting a "
            "heuristic's result as an optimum would be a different claim. `exact_under_bfv` records "
            "that the encrypted score equals the plaintext score bit for bit, which is the property "
            "a CKKS student cannot offer; verified in tests/test_scorecard_bfv_exact.py."
        ),
        "git_sha": train.git_sha(),
        "wall_clock_s": round(time.time() - started, 1),
    })
    print(f"  recorded finance_aml.scorecard in {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

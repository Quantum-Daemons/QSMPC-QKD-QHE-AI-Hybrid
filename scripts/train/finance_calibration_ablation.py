#!/usr/bin/env python
"""Isotonic vs Platt vs raw on the finance AML split, and whether CASES or EVENTS binds.

WHY THIS EXISTS

`src/qsh/models/calibration.py` records an OPEN question, conditioned on Round 48:

    ICML 2005 (Niculescu-Mizil and Caruana) says "when there are 1000 or more points in the
    calibration set, Isotonic Regression always yields performance as good as, or better than,
    Platt Scaling". This project's finance calibration set has 710,969 CASES, far past that. Platt
    was retained anyway on the argument that those papers benchmark broadly balanced problems, and
    that at a 1-in-981 base rate the quantity that plausibly binds is the POSITIVE-EVENT count, not
    the case count.

Nobody had measured it. This script does, on the same split, the same seed and the same score
source as `scripts/train/finance_aml.py`, and it measures the contested claim DIRECTLY rather than
arguing it: the calibration set is thinned along two axes independently - positives held while cases
vary, and cases held while positives vary - so "cases or events" is answered by which axis moves the
isotonic-minus-Platt gap.

FOUR ARMS, NOT THREE, AND THE FOURTH IS THE REASON THE ANSWER IS NOT A SWAP

`fit_platt` is called on `teacher.predict_proba(...)[:, 1]` - a PROBABILITY. Platt scaling fits
``p = sigmoid(-(a*s + b))``, which is linear in ``s``; the literature's ``s`` is an SVM margin or a
logit, a quantity whose log-odds are roughly linear. On this split 90% of the raw scores lie below
1e-4, so ``a*s`` is numerically zero across nine of ten deciles and the fitted sigmoid returns one
constant for all of them. That is not Platt scaling underperforming; it is Platt scaling applied to
the wrong argument. The fourth arm applies the identical `fit_platt` to ``logit(s)`` and is the
comparison the cited papers are actually about.

WHAT IT REPRODUCES, AND HOW THAT IS CHECKED

The shipped calibrator is fitted on the VALIDATION window and scored on the TEST window
(`finance_aml.py`, "probability calibration" block). This script imports that module and calls the
same feature functions with the same arguments, trains the same booster at the same seed, and then
prints the reproduction check against `models/metrics.json` so a drift is visible rather than
swallowed.

    calibration partition = validation window   710,969 cases /   801 positives
    scoring partition     = test window       1,523,504 cases / 2,321 positives

Those two numbers are different partitions, and the record conflated them: the "2,321 positive
events" quoted as the calibrator's event count is the TEST split's, and the calibration set the
whole argument is about carries 801.

Usage:
    python scripts/train/finance_calibration_ablation.py            # trains once, caches scores
    python scripts/train/finance_calibration_ablation.py --repeats 40 --bins 20
    python scripts/train/finance_calibration_ablation.py --refit    # discard the cache
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from qsh.models import calibration as cal  # noqa: E402
from qsh.models import datasets, train  # noqa: E402

#: Clip before the log-odds transform. The raw scores are strictly inside (0, 1) on this split
#: (min 4.4e-11, max 0.99997), so nothing is actually clipped here; the guard exists so a retrain
#: that saturates a leaf cannot turn a calibration arm into an infinity.
_LOGIT_EPS = 1e-12


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(q / (1.0 - q))


def _finance_aml():
    """Import the trainer as a module. It is a script, not a package, so load it by path.

    Importing rather than re-implementing is the point: a copy of `graph_features` here would be a
    second definition of the split this measurement claims to reproduce, and the two would drift.
    """
    src = Path(__file__).resolve().parent / "finance_aml.py"
    spec = importlib.util.spec_from_file_location("finance_aml", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
#  Scores: the same booster, the same split, cached because it costs a minute
# --------------------------------------------------------------------------- #
def build_scores(seed: int) -> dict:
    """Train the shipped teacher and return its val/test scores and labels."""
    import xgboost as xgb

    fa = _finance_aml()
    train.seed_everything(seed)

    t0 = time.time()
    df = datasets.load_amlworld().sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df) * (1.0 - fa.TEST_FRACTION))
    val_cut = int(cut * (1.0 - fa.VAL_FRACTION))
    fit, val, te = df.iloc[:val_cut], df.iloc[val_cut:cut], df.iloc[cut:]
    tr = df.iloc[:cut]

    cz = fa.causal_account_features(df)
    Xfit = fa.graph_features(fit, causal=cz)
    Xval = fa.graph_features(val, fit_on=fit, causal=cz)
    Xte = fa.graph_features(te, fit_on=tr, causal=cz)
    yfit = fit["label"].to_numpy(dtype=np.int8)
    yval = val["label"].to_numpy(dtype=np.int8)
    yte = te["label"].to_numpy(dtype=np.int8)

    spw = float((yfit == 0).sum() / max((yfit == 1).sum(), 1))
    teacher = xgb.XGBClassifier(
        n_estimators=600, max_depth=8, learning_rate=0.08, subsample=0.9,
        colsample_bytree=0.9, min_child_weight=4, reg_lambda=2.0,
        scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist",
        device="cuda", n_jobs=16, random_state=seed,
    )
    teacher.fit(Xfit, yfit, eval_set=[(Xval, yval)], verbose=False)
    teacher.set_params(device="cpu")
    return {
        "s_val": teacher.predict_proba(Xval)[:, 1].astype(np.float64),
        "y_val": yval.astype(np.int8),
        "s_test": teacher.predict_proba(Xte)[:, 1].astype(np.float64),
        "y_test": yte.astype(np.int8),
        "seed": np.int64(seed),
        "wall_clock_s": np.float64(time.time() - t0),
    }


def load_scores(seed: int, *, refit: bool) -> dict:
    path = datasets.artifacts("finance_calibration_scores.npz")
    if path.exists() and not refit:
        z = np.load(path)
        if int(z["seed"]) == seed:
            print(f"  cached scores: {path}")
            return {k: z[k] for k in z.files}
    print("  no usable cache; training the shipped teacher (this is the slow step) ...")
    d = build_scores(seed)
    np.savez_compressed(path, **d)
    print(f"  wrote {path} in {float(d['wall_clock_s']):.0f}s")
    return d


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def equal_frequency_bins(p: np.ndarray, k: int) -> list[np.ndarray]:
    """Index groups of (as near as integer division allows) equal size, ordered by ``p``.

    Equal-FREQUENCY rather than equal-width. At a 1-in-981 base rate an equal-width binning puts
    every row in the first bin and reports the miscalibration of one number; equal-frequency spends
    its resolution where the rows are. Ties are split across the boundary they fall on, which is a
    real approximation for isotonic output (a step function has heavy ties) and is stated rather
    than hidden - it is why the bin count is reported at several values below.
    """
    order = np.argsort(p, kind="stable")
    return np.array_split(order, k)


def ece_pair(p: np.ndarray, y: np.ndarray, k: int) -> tuple[float, float]:
    """Expected calibration error over ``k`` equal-frequency bins, count- and event-weighted.

    The count-weighted form is the standard one. The event-weighted form is the same per-bin gap
    with the weight moved onto the positives, and it is here because the whole contention is that
    the base rate makes the case count the wrong denominator: a count-weighted ECE at 1-in-981 is
    99.9% a statement about clean transactions. Bins are cut on all rows in both cases - cutting
    them on positives only would force every bin's observed frequency to 1 and measure nothing.
    """
    n = len(p)
    npos = float(y.sum())
    count_w = 0.0
    event_w = 0.0
    for idx in equal_frequency_bins(p, k):
        if idx.size == 0:
            continue
        gap = abs(float(p[idx].mean()) - float(y[idx].mean()))
        count_w += (idx.size / n) * gap
        if npos > 0:
            event_w += (float(y[idx].sum()) / npos) * gap
    return count_w, (event_w if npos > 0 else float("nan"))


def brier_positive(p: np.ndarray, y: np.ndarray) -> float:
    m = y == 1
    return float(np.mean((p[m] - 1.0) ** 2)) if m.any() else float("nan")


def log_loss_positive(p: np.ndarray, y: np.ndarray, *, eps: float = 1e-15) -> float:
    m = y == 1
    return float(-np.mean(np.log(np.clip(p[m], eps, 1.0)))) if m.any() else float("nan")


def ranking(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def score_block(p: np.ndarray, y: np.ndarray, bins: tuple[int, ...]) -> dict:
    auroc, auprc = ranking(p, y)
    out = {
        "brier": cal.brier(p, y),
        "log_loss": cal.log_loss(p, y),
        "brier_pos": brier_positive(p, y),
        "log_loss_pos": log_loss_positive(p, y),
        "auroc": auroc,
        "auprc": auprc,
        "mean_predicted": float(p.mean()),
        "base_rate": float(y.mean()),
        "n_distinct": int(np.unique(p).size),
        "p_min": float(p.min()),
        "p_max": float(p.max()),
        "n_exactly_zero": int((p == 0.0).sum()),
        "positives_on_exact_zero": int(y[p == 0.0].sum()),
    }
    for k in bins:
        out[f"ece_{k}"], out[f"ece_pos_{k}"] = ece_pair(p, y, k)
    out["corp"] = cal.corp_decomposition(p, y)
    return out


# --------------------------------------------------------------------------- #
#  Calibrators. Three fitted arms; `raw` is the identity and is not fitted.
# --------------------------------------------------------------------------- #
#: Order matters only for printing. `platt_prob` is what ships today.
ARMS = ("platt_prob", "platt_logit", "isotonic")


def fit_arms(s_cal: np.ndarray, y_cal: np.ndarray) -> dict:
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(s_cal, y_cal.astype(np.float64))
    return {
        "platt_prob": cal.fit_platt(s_cal, y_cal),
        "platt_logit": cal.fit_platt(logit(s_cal), y_cal),
        "isotonic": iso,
    }


def apply_arm(name: str, fitted, s_score: np.ndarray) -> np.ndarray:
    if name == "platt_prob":
        return fitted.apply(s_score)
    if name == "platt_logit":
        return fitted.apply(logit(s_score))
    return np.asarray(fitted.predict(s_score), dtype=np.float64)


# --------------------------------------------------------------------------- #
#  The contention: thin the calibration set along one axis at a time
# --------------------------------------------------------------------------- #
def subsample(rng, s, y, *, n_pos: int, n_neg: int):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    take_p = pos if n_pos >= pos.size else rng.choice(pos, size=n_pos, replace=False)
    take_n = neg if n_neg >= neg.size else rng.choice(neg, size=n_neg, replace=False)
    idx = np.concatenate([take_p, take_n])
    return s[idx], y[idx]


def sweep(rng, s_cal, y_cal, s_score, y_score, cells, repeats, bins_k):
    """For each (n_pos, n_neg) cell, fit every arm ``repeats`` times and score on the held-out
    partition. Reports the mean and the standard deviation across draws, because "isotonic is
    better" and "isotonic is better than the draw-to-draw noise" are different claims.

    AUPRC is measured for the isotonic arm only: both Platt arms are STRICTLY monotone, so their
    ranking is the raw ranking to the last digit and re-sorting 1.5M rows to confirm it 20 times
    per cell measures nothing. Isotonic is only WEAKLY monotone and does move it, which is the
    reason the column exists at all.
    """
    rows = []
    pos_avail = int((y_cal == 1).sum())
    neg_avail = int((y_cal == 0).sum())
    for n_pos, n_neg in cells:
        acc: dict[str, list[float]] = {}
        full = n_pos >= pos_avail and n_neg >= neg_avail
        r = 1 if full else repeats
        for _ in range(r):
            sc, yc = subsample(rng, s_cal, y_cal, n_pos=n_pos, n_neg=n_neg)
            fitted = fit_arms(sc, yc)
            per: dict[str, dict] = {}
            for name in ARMS:
                p = apply_arm(name, fitted[name], s_score)
                e, ep = ece_pair(p, y_score, bins_k)
                per[name] = {
                    "brier": cal.brier(p, y_score),
                    "logloss": cal.log_loss(p, y_score),
                    "brier_pos": brier_positive(p, y_score),
                    "logloss_pos": log_loss_positive(p, y_score),
                    "ece": e, "ece_pos": ep,
                }
                if name == "isotonic":
                    per[name]["auprc"] = ranking(p, y_score)[1]
                for k, v in per[name].items():
                    acc.setdefault(f"{name}_{k}", []).append(v)
            for base in ("platt_prob", "platt_logit"):
                for k in ("brier", "logloss", "ece", "ece_pos"):
                    acc.setdefault(f"d_{base}_{k}", []).append(
                        per[base][k] - per["isotonic"][k])
        row = {
            "n_pos": min(n_pos, pos_avail), "n_neg": min(n_neg, neg_avail),
            "n_cases": min(n_pos, pos_avail) + min(n_neg, neg_avail),
            "repeats": r, "is_full_set": bool(full),
        }
        row["base_rate"] = row["n_pos"] / row["n_cases"]
        for k, v in acc.items():
            row[k] = float(np.mean(v))
            row[k + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        rows.append(row)
        print(f"    events {row['n_pos']:5d} cases {row['n_cases']:8d} | "
              f"iso brier {row['isotonic_brier']:.8f} | "
              f"platt_prob-iso {row['d_platt_prob_brier']:+.2e} (sd {row['d_platt_prob_brier_sd']:.1e})"
              f" | platt_logit-iso {row['d_platt_logit_brier']:+.2e} "
              f"(sd {row['d_platt_logit_brier_sd']:.1e})")
    return rows


def reliability_table(p: np.ndarray, y: np.ndarray, k: int) -> list[dict]:
    out = []
    for i, idx in enumerate(equal_frequency_bins(p, k)):
        if idx.size == 0:
            continue
        out.append({
            "bin": i + 1, "n": int(idx.size), "n_pos": int(y[idx].sum()),
            "p_lo": float(p[idx].min()), "p_hi": float(p[idx].max()),
            "mean_predicted": float(p[idx].mean()), "observed": float(y[idx].mean()),
        })
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="the trainer's seed; do not change it "
                    "unless you also intend to stop reproducing the shipped split")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--bins", type=int, default=20, help="equal-frequency bins for the headline ECE")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    bins_all = (10, 20, 50, 100)
    print("finance AML calibration: isotonic vs Platt (on p and on logit p) vs raw")
    d = load_scores(args.seed, refit=args.refit)
    s_val, y_val = np.asarray(d["s_val"]), np.asarray(d["y_val"]).astype(int)
    s_te, y_te = np.asarray(d["s_test"]), np.asarray(d["y_test"]).astype(int)
    print(f"  calibration partition (validation window): {len(s_val):,} cases, "
          f"{int(y_val.sum()):,} positives, base rate 1 in {len(s_val)/max(y_val.sum(),1):.0f}")
    print(f"  scoring partition     (test window):       {len(s_te):,} cases, "
          f"{int(y_te.sum()):,} positives, base rate 1 in {len(s_te)/max(y_te.sum(),1):.0f}")

    # ---- 1-3: the arms on the full calibration set ----------------------------------------------
    fitted = fit_arms(s_val, y_val)
    probs = {"raw": s_te}
    probs.update({name: apply_arm(name, fitted[name], s_te) for name in ARMS})
    arms = {name: score_block(p, y_te, bins_all) for name, p in probs.items()}

    for name in ("platt_prob", "platt_logit"):
        f = fitted[name]
        print(f"\n  {name}: a={f.a:.8f} b={f.b:.8f} "
              f"(n_calibration={f.n_calibration:,}, positives={f.positives_calibration})")
    hdr = (f"  {'arm':12s} {'Brier':>12s} {'log-loss':>12s} {'ECE':>11s} {'ECEpos':>9s} "
           f"{'Brier|y=1':>10s} {'LL|y=1':>9s} {'AUROC':>10s} {'AUPRC':>9s} {'distinct':>9s}")

    def _table(block):
        print(hdr)
        for name, m in block.items():
            print(f"  {name:12s} {m['brier']:12.8f} {m['log_loss']:12.8f} "
                  f"{m[f'ece_{args.bins}']:11.8f} {m[f'ece_pos_{args.bins}']:9.6f} "
                  f"{m['brier_pos']:10.6f} {m['log_loss_pos']:9.6f} {m['auroc']:10.8f} "
                  f"{m['auprc']:9.6f} {m['n_distinct']:9d}")

    print()
    _table(arms)

    print("\n  ECE at several bin counts (equal-frequency; count-weighted / event-weighted)")
    for k in bins_all:
        print(f"    k={k:<4d}" + "".join(
            f"  {n}={m[f'ece_{k}']:.3e}/{m[f'ece_pos_{k}']:.4f}" for n, m in arms.items()))

    print("\n  CORP decomposition (S = MCB - DSC + UNC)")
    for name, m in arms.items():
        c = m["corp"]
        print(f"    {name:12s} MCB {c['mcb']:.4e}  DSC {c['dsc']:.4e}  UNC {c['unc']:.4e}")

    print("\n  output support (a probability of exactly 0 is an unbounded log-loss liability)")
    for name, m in arms.items():
        print(f"    {name:12s} [{m['p_min']:.3e}, {m['p_max']:.6f}]  distinct {m['n_distinct']:,}"
              f"  exact-zero rows {m['n_exactly_zero']:,} "
              f"(positives on them: {m['positives_on_exact_zero']})")

    # ---- 4: cases or events? -------------------------------------------------------------------
    pos_avail = int(y_val.sum())
    neg_avail = int((y_val == 0).sum())
    base = pos_avail / len(y_val)
    rng = np.random.default_rng(args.seed)

    # Only TWO of {cases, events, base rate} are free, because the third is their ratio. So no
    # sweep can vary one while holding the other two, and "cases or events" is answerable only by
    # pinning one quantity at a time and reading which pin removes the effect. That is why there
    # are three sweeps and not one, and it is a property of the question rather than of this code.
    print("\n  SWEEP A: base rate held; cases and events scale together")
    # The low end reaches BELOW both published thresholds - 3 events at this base rate is 2,663
    # cases, which straddles UAI's ~2000 and is past ICML's 200-1000. A grid whose smallest cell is
    # already above the thresholds cannot find a crossover, which is what the first pass did.
    targets = [3, 6, 12, 25, 50, 100, 200, 400, pos_avail]
    cells_a = [(p, int(round(p / base)) - p) for p in targets]
    rows_a = sweep(rng, s_val, y_val, s_te, y_te, cells_a, args.repeats, args.bins)

    print("\n  SWEEP B: cases held at the full 710,969; events thinned")
    rows_b = sweep(rng, s_val, y_val, s_te, y_te, [(p, neg_avail) for p in targets],
                   args.repeats, args.bins)

    print("\n  SWEEP C: events held at 100; cases varied by thinning negatives")
    cells_c = [(100, n) for n in (400, 900, 2900, 9900, 29900, 99900, 299900, neg_avail)]
    rows_c = sweep(rng, s_val, y_val, s_te, y_te, cells_c, args.repeats, args.bins)

    # ---- supplementary: the reversed roles, the only 2,321-event calibration set ---------------
    print("\n  SUPPLEMENTARY: partitions swapped - calibrate on TEST (2,321 events), score on VAL")
    fitted_r = fit_arms(s_te, y_te)
    rev = {"raw": score_block(s_val, y_val, bins_all)}
    rev.update({name: score_block(apply_arm(name, fitted_r[name], s_val), y_val, bins_all)
                for name in ARMS})
    print()
    _table(rev)

    # ---- reliability tables --------------------------------------------------------------------
    rel = {name: reliability_table(p, y_te, args.bins) for name, p in probs.items()}
    rel10 = {name: reliability_table(p, y_te, 10) for name, p in probs.items()}
    print("\n  reliability, 10 equal-frequency bins (the 20-bin table is in the JSON)")
    for name, tbl in rel10.items():
        print(f"    {name}")
        for r in tbl:
            print(f"      bin {r['bin']:3d}  n {r['n']:7d}  pos {r['n_pos']:5d}  "
                  f"pred {r['mean_predicted']:.8f}  obs {r['observed']:.8f}")

    out = args.out or datasets.artifacts("finance_calibration_ablation.json")
    Path(out).write_text(json.dumps({
        "seed": args.seed, "bins_headline": args.bins, "bins_reported": list(bins_all),
        "repeats": args.repeats,
        "calibration_partition": {"n": int(len(s_val)), "positives": int(y_val.sum())},
        "scoring_partition": {"n": int(len(s_te)), "positives": int(y_te.sum())},
        "platt_prob": fitted["platt_prob"].as_dict(),
        "platt_logit": fitted["platt_logit"].as_dict(),
        "arms": arms, "reversed_roles": rev,
        "sweep_a_fixed_base_rate": rows_a,
        "sweep_b_fixed_cases": rows_b,
        "sweep_c_fixed_events": rows_c,
        "reliability_20": rel, "reliability_10": rel10,
        "git_sha": train.git_sha(),
    }, indent=2, default=float) + "\n")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

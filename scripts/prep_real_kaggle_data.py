#!/usr/bin/env python3
"""Build the bundled finance (PaySim) sample from the Kaggle download.

This is a **one-time provenance script** (run by a maintainer, not at runtime). It turns the large raw
Kaggle datasets into the small real samples the repo ships + the demo loads (PaySim under CC BY-SA 4.0):

  data/finance/paysim_sample.tsv                  PaySim simulator output (synthetic) -> 8 AML-named columns + label

The NASA SMAP/MSL satellite sample this script also wrote, and its browser copy, are not written any
more: no source grants a licence for that data (DATA_LICENSES.md section 2), both were removed from
the tree in Round 84, and both satellite anomaly checks score OPS-SAT-AD segments (CC BY 4.0) built
by `scripts/data/build_heldout_payloads.py`.

`web/public/data/finance/paysim_sample.json` is NOT written here any more. It is built by
`scripts/data/build_federated_samples.py` from the TSV this script produces, alongside the medical
sample, because the browser federates over both and they have to be sized and standardised by one
decision rather than two. Leaving a second writer here would be the drift this repo keeps finding:
whoever re-ran this script for the TSV would silently restore the 16-row unstandardised payload that
made federated accuracy worse than the constant predictor, and every gate would stay green.

Sources (licences re-read 2026-08-03 -- see DATA_LICENSES.md for what each actually grants):
  * PaySim  : Kaggle `ealaxi/paysim1`, license **CC-BY-SA-4.0** (Lopez-Rojas et al. 2016 mobile-money simulator).

Obtain the raw data first (needs Kaggle API creds in ~/.kaggle/kaggle.json):
  kaggle datasets download -d ealaxi/paysim1 -p /tmp/kaggle_dl/paysim --unzip

Then:  python scripts/prep_real_kaggle_data.py --src /tmp/kaggle_dl
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np

#: Provenance text is NOT written here. It comes from the registry, which is also what the browser
#: prints and what tests/test_demo_provenance.py asserts these JSON files against. Re-declaring the
#: strings locally is how this script previously re-introduced a licence claim that had already been
#: corrected everywhere else.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qsh.models.registry import (  # noqa: E402
    BUNDLED_DEMO_SAMPLES,
    PAYSIM_MODEL_FEATURES,
    PAYSIM_SAMPLE_COLUMNS,
)

# ----- finance column layout: the registry's, in TSV order -----------------------------------------
# The TSV keeps all eight columns, the layout its readers parse; no shipped model reads all eight.
# The demo's linear model, here and in usecases/finance_aml, and the finance QCNN artifact since its
# Round 84 retrain read four of them (PAYSIM_MODEL_FEATURES); the reasons the other four are not
# inputs are PAYSIM_EXCLUDED_FEATURES.
FIN_NAMES = list(PAYSIM_SAMPLE_COLUMNS)
#: The comment block that heads the TSV, written here and checked against the shipped file by
#: tests/test_r84_finance_paysim_model.py, so a regeneration cannot drop the balance-column note.
FIN_TSV_HEADER = (
    "# PaySim simulator output (Kaggle ealaxi/paysim1, CC-BY-SA-4.0; Lopez-Rojas et al. 2016). PaySim is a",
    "# SIMULATOR - these are synthetic transactions from aggregated real mobile-money logs, not real",
    "# transactions. Columns = 8 AML features (PaySim-derived) + isFraud.",
    "# device_risk and geo_mismatch are computed from PaySim's balance columns (oldbalanceOrg,",
    "# newbalanceOrig, oldbalanceDest, newbalanceDest), which the dataset's description says must not be",
    "# used for fraud detection because fraudulent transactions are cancelled. The demo's linear model",
    "# reads amount, cross_border, night_tx and new_payee only (src/qsh/models/registry.py).",
)
assert all(name in FIN_TSV_HEADER[-1] for name in PAYSIM_MODEL_FEATURES), "TSV header names the wrong columns"

FIN_PROVENANCE = BUNDLED_DEMO_SAMPLES["finance_aml"][0]


def _paysim_features(df):
    """Map PaySim (simulator) columns -> the 8 AML-named columns (in FIN_NAMES order).

    The NAMES are this project's labels, not PaySim fields, and three of them describe something
    else: PaySim has no geo or device fields, so `cross_border` is "type is TRANSFER or CASH_OUT",
    `device_risk` is the origin-balance inconsistency flag and `geo_mismatch` measures how far the
    destination's balance change falls short of the amount. The other five are computed from PaySim
    fields as their names say; `amount` is log1p(amount) and `amount_zscore` is a z-score of that
    same value, so the two carry one signal.

    `device_risk` and `geo_mismatch` read oldbalanceOrg, newbalanceOrig, oldbalanceDest and
    newbalanceDest, which the dataset's description says must not be used for fraud detection
    because fraudulent transactions are cancelled (registry.PAYSIM_BALANCE_WARNING). They are still
    written, so the TSV keeps the layout its readers parse; no model in this repository reads them
    since the finance QCNN artifact's Round 84 retrain on four columns
    (registry.PAYSIM_EXCLUDED_FEATURES)."""
    import numpy as np
    amount = df["amount"].to_numpy(float)
    log_amt = np.log1p(amount)
    step = df["step"].to_numpy(float)
    # velocity_24h: prior same-origin transactions within 24 steps (PaySim step = 1 hour)
    df = df.assign(_log_amt=log_amt)
    df_sorted = df.sort_values(["nameOrig", "step"])

    def _vel(steps):
        steps = steps.to_numpy(float)
        firsts = np.searchsorted(steps, steps - 24, side="right")
        return np.arange(len(steps)) - firsts

    vel = df_sorted.groupby("nameOrig")["step"].transform(_vel).reindex(df.index).to_numpy(float)
    typ = df["type"].to_numpy(str)
    cross_border = np.isin(typ, ["TRANSFER", "CASH_OUT"]).astype(float)        # PaySim fraud-bearing types
    night_tx = ((step % 24) < 6).astype(float)
    new_payee = (~df["nameDest"].duplicated()).to_numpy().astype(float)        # first global sight of the payee
    amount_zscore = (log_amt - log_amt.mean()) / (log_amt.std() + 1e-9)
    err_orig = np.abs(df["oldbalanceOrg"].to_numpy(float) - amount - df["newbalanceOrig"].to_numpy(float))
    device_risk = (err_orig > 0.01).astype(float)                              # origin balance inconsistency
    dest_delta = df["newbalanceDest"].to_numpy(float) - df["oldbalanceDest"].to_numpy(float)
    geo_mismatch = np.clip(np.abs(amount - dest_delta) / (amount + 1.0), 0, 1)  # dest did not receive funds
    X = np.stack([log_amt, vel, cross_border, night_tx, new_payee, amount_zscore, device_risk, geo_mismatch], axis=1)
    return X.astype(np.float32)


def prep_finance(src: Path, repo: Path, n: int = 4000, fraud_rate: float = 0.05, seed: int = 0):
    # The XGBoost teacher and the logistic student moved with the demo JSON to
    # `scripts/data/build_federated_samples.py`; this function's job is now only the TSV.
    import pandas as pd

    csv = next(src.glob("paysim/*.csv"))
    print(f"[finance] reading {csv} ...")
    df = pd.read_csv(csv)
    print(f"[finance] {len(df):,} rows; computing real features ...")
    X = _paysim_features(df)
    y = df["isFraud"].to_numpy(np.int64)

    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    pick = np.concatenate([rng.choice(fraud_idx, min(n_fraud, len(fraud_idx)), replace=False),
                           rng.choice(legit_idx, n - n_fraud, replace=False)])
    rng.shuffle(pick)
    Xs, ys = X[pick], y[pick]
    print(f"[finance] bundled sample: {Xs.shape}, fraud={int(ys.sum())}/{len(ys)}")

    # bundled TSV (backend loader reads this)
    out_tsv = repo / "data" / "finance" / "paysim_sample.tsv"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w") as f:
        f.write("".join(f"{line}\n" for line in FIN_TSV_HEADER))
        f.write("\t".join(FIN_NAMES) + "\tisFraud\n")
        for row, label in zip(Xs, ys, strict=True):
            f.write("\t".join(f"{v:.5f}" for v in row) + f"\t{int(label)}\n")
    print(f"[finance] wrote {out_tsv} ({out_tsv.stat().st_size} B)")

    # The demo JSON is NOT written here. It used to be, with `demo_idx` picking the first 8 fraud and
    # first 8 legit rows - the 16 rows that made `federated_accuracy` a 4-row score, and a 4-row score
    # that came out BELOW the constant predictor on 4 of 5 seeds. It is now built by
    # `scripts/data/build_federated_samples.py`, which sizes and standardises it together with the
    # medical sample because the browser federates over both.
    #
    # Two writers for one artefact is the drift this repo keeps finding. Re-running this script for
    # the TSV must not quietly undo that, so it says what to run instead rather than doing it.
    print("[finance] demo JSON NOT written here - run scripts/data/build_federated_samples.py "
          "after this, which builds it from the TSV above together with the medical sample")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/kaggle_dl", type=Path, help="dir with the paysim/ Kaggle download")
    ap.add_argument("--repo", default=Path(__file__).resolve().parents[1], type=Path)
    ap.add_argument("--only", choices=["finance"], help="prep only one vertical")
    args = ap.parse_args()
    if args.only in (None, "finance"):
        prep_finance(args.src, args.repo)
    print("done.")


if __name__ == "__main__":
    main()

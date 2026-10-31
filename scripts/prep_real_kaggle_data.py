#!/usr/bin/env python3
"""Build the bundled REAL finance (PaySim) + satellite (NASA SMAP/MSL) samples from Kaggle downloads.

This is a **one-time provenance script** (run by a maintainer, not at runtime). It turns the large raw
Kaggle datasets into the small real samples the repo ships + the demo loads (PaySim under CC BY-SA 4.0;
SMAP/MSL with no licence granted for the data, see below):

  data/finance/paysim_sample.tsv                  PaySim simulator output (synthetic) -> 8 AML-named columns + label
  data/satellite/smap_msl_telemetry.tsv           real NASA SMAP/MSL telemetry (6 channels) + anomaly label

`web/public/data/satellite/telemetry.json` is NOT written here any more (Round 84). It put SMAP/MSL
windows on the public site, and no source grants a licence for that data (below). The browser's
satellite anomaly demo now scores OPS-SAT-AD segments, CC BY 4.0, built by
`scripts/data/build_heldout_payloads.py` from a file the repository already tracks. Writing the old
file from here would quietly restore an unlicensed public copy whenever someone refreshed the TSV.

`web/public/data/finance/paysim_sample.json` is NOT written here any more. It is built by
`scripts/data/build_federated_samples.py` from the TSV this script produces, alongside the medical
sample, because the browser federates over both and they have to be sized and standardised by one
decision rather than two. Leaving a second writer here would be the drift this repo keeps finding:
whoever re-ran this script for the TSV would silently restore the 16-row unstandardised payload that
made federated accuracy worse than the constant predictor, and every gate would stay green.

Sources (licences re-read 2026-08-03 -- see DATA_LICENSES.md for what each actually grants):
  * PaySim  : Kaggle `ealaxi/paysim1`, license **CC-BY-SA-4.0** (Lopez-Rojas et al. 2016 mobile-money simulator).
  * NASA SMAP/MSL : Kaggle `patrickfleith/nasa-anomaly-detection-dataset-smap-msl`; the underlying data is
    NASA JPL spacecraft telemetry (Hundman et al. 2018, KDD; the Telemanom project). NO SOURCE GRANTS A
    LICENCE FOR THE DATA. The Telemanom repository's LICENSE.txt is "Copyright (c) 2018, California
    Institute of Technology" with BSD-style terms for its software and says nothing about the data, and
    the Kaggle mirror says "Data files (c) Original Authors". Nor is the data shown to be a US Government
    work: 17 U.S.C. 105 covers works prepared by federal officers or employees, and JPL is a NASA centre
    operated by Caltech -- see DATA_LICENSES.md.

Obtain the raw data first (needs Kaggle API creds in ~/.kaggle/kaggle.json):
  kaggle datasets download -d ealaxi/paysim1 -p /tmp/kaggle_dl/paysim --unzip
  kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl -p /tmp/kaggle_dl/nasa --unzip

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
    DS_SMAP_MSL,
    PAYSIM_MODEL_FEATURES,
    PAYSIM_SAMPLE_COLUMNS,
)

# ----- finance column layout: the registry's, in TSV order -----------------------------------------
# The TSV keeps all eight columns because the finance QCNN artifact was trained on them and reads
# them by position. The demo's linear model, here and in usecases/finance_aml, reads four of them
# (PAYSIM_MODEL_FEATURES); the reasons the other four are not inputs are PAYSIM_EXCLUDED_FEATURES.
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
#: The TSV's header names SMAP/MSL by its own registry record, not by the satellite DEMO SAMPLE's
#: text: the demo sample is OPS-SAT-AD since Round 84, and borrowing its provenance here would label
#: NASA telemetry as OPS-SAT data.
SAT_PROVENANCE = f"{DS_SMAP_MSL.name}: {DS_SMAP_MSL.licence} - see DATA_LICENSES.md section 2."


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
    written, because the finance QCNN artifact was trained on all eight columns; no linear model in
    this repository reads them (registry.PAYSIM_EXCLUDED_FEATURES)."""
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


def prep_satellite(src: Path, repo: Path, n_channels: int = 6, length: int = 900,
                   anchor: int = 350, max_anom: int = 150):
    import csv as _csv

    nasa = src / "nasa"
    labels_csv = nasa / "labeled_anomalies.csv"
    test_dir = nasa / "data" / "data" / "test"

    # Pick channels whose FIRST labeled anomaly is short (a point/contextual event, <= max_anom long) so the
    # window stays mostly nominal, and anchor every channel's anomaly to the SAME local index so the OR across
    # channels yields a realistic (~15-20%) anomaly rate with BOTH classes (not a saturated all-anomalous slice).
    chans = []
    with labels_csv.open() as f:
        for row in _csv.DictReader(f):
            seqs = ast.literal_eval(row["anomaly_sequences"])
            if not seqs or not (test_dir / f"{row['chan_id']}.npy").exists():
                continue
            a0, a1 = seqs[0]
            if (a1 - a0) <= max_anom and a0 >= anchor and int(row["num_values"]) >= (a0 - anchor) + length:
                chans.append((row["chan_id"], seqs))
            if len(chans) >= n_channels:
                break
    assert len(chans) >= n_channels, f"only found {len(chans)} usable channels"
    names = [c for c, _ in chans]
    print(f"[satellite] channels: {names}")

    cols = []   # per channel: (telemetry[length], anomaly[length])
    for ch, seqs in chans:
        series = np.load(test_dir / f"{ch}.npy")[:, 0].astype(float)            # col 0 = real telemetry value
        mask = np.zeros(len(series), dtype=int)
        for s, e in seqs:
            mask[s:e + 1] = 1
        start = int(np.clip(seqs[0][0] - anchor, 0, len(series) - length))      # anchor the 1st anomaly at `anchor`
        cols.append((series[start:start + length], mask[start:start + length]))

    telem = np.stack([c[0] for c in cols], axis=1)                              # (length, C)
    anom = np.clip(np.sum([c[1] for c in cols], axis=0), 0, 1)                  # OR across channels per timestep
    print(f"[satellite] telemetry {telem.shape}, anomalous timesteps={int(anom.sum())}/{length} "
          f"({100*anom.mean():.1f}%)")
    assert 0 < anom.sum() < length, "need both nominal + anomalous timesteps"

    out_tsv = repo / "data" / "satellite" / "smap_msl_telemetry.tsv"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w") as f:
        f.write(f"# {SAT_PROVENANCE}\n"
                "# Columns = 6 real telemetry channels + labeled anomaly (1).\n")
        f.write("\t".join(names) + "\tanomaly\n")
        for i in range(length):
            f.write("\t".join(f"{telem[i, c]:.5f}" for c in range(len(names))) + f"\t{int(anom[i])}\n")
    print(f"[satellite] wrote {out_tsv} ({out_tsv.stat().st_size} B)")

    # No browser payload from here: see the module docstring. The TSV above is what the Python
    # deployment's loader reads; whether it may stay in the repository at all is the same licence
    # question, recorded in DATA_LICENSES.md section 2.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/kaggle_dl", type=Path, help="dir with paysim/ + nasa/ Kaggle downloads")
    ap.add_argument("--repo", default=Path(__file__).resolve().parents[1], type=Path)
    ap.add_argument("--only", choices=["finance", "satellite"], help="prep only one vertical")
    args = ap.parse_args()
    if args.only in (None, "finance"):
        prep_finance(args.src, args.repo)
    if args.only in (None, "satellite"):
        prep_satellite(args.src, args.repo)
    print("done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the bundled REAL finance (PaySim) + satellite (NASA SMAP/MSL) samples from Kaggle downloads.

This is a **one-time provenance script** (run by a maintainer, not at runtime). It turns the large raw
Kaggle datasets into the small, commercial-OK real samples the repo ships + the demo loads:

  data/finance/paysim_sample.tsv                  PaySim simulator output (synthetic) -> the 8 AML features + label
  data/satellite/smap_msl_telemetry.tsv           real NASA SMAP/MSL telemetry (6 channels) + anomaly label
  web/public/data/satellite/telemetry.json        demo: real telemetry windows + a real anomaly model

`web/public/data/finance/paysim_sample.json` is NOT written here any more. It is built by
`scripts/data/build_federated_samples.py` from the TSV this script produces, alongside the medical
sample, because the browser federates over both and they have to be sized and standardised by one
decision rather than two. Leaving a second writer here would be the drift this repo keeps finding:
whoever re-ran this script for the TSV would silently restore the 16-row unstandardised payload that
made federated accuracy worse than the constant predictor, and every gate would stay green.

Sources (licences re-read 2026-08-03 -- see DATA_LICENSES.md for what each actually grants):
  * PaySim  : Kaggle `ealaxi/paysim1`, license **CC-BY-SA-4.0** (Lopez-Rojas et al. 2016 mobile-money simulator).
  * NASA SMAP/MSL : Kaggle `patrickfleith/nasa-anomaly-detection-dataset-smap-msl`; the underlying data is
    NASA JPL spacecraft telemetry (Hundman et al. 2018, KDD; the Telemanom project). Per NASA's official data
    policy (NASA Science Data Portal + STI), US-Government works are not copyrightable (17 U.S.C. s.105) and
    NASA-mission data carries NO licence statement from any provider. 17 U.S.C. s.105 removes US
    copyright domestically but is not an affirmative grant -- see DATA_LICENSES.md.

Obtain the raw data first (needs Kaggle API creds in ~/.kaggle/kaggle.json):
  kaggle datasets download -d ealaxi/paysim1 -p /tmp/kaggle_dl/paysim --unzip
  kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl -p /tmp/kaggle_dl/nasa --unzip

Then:  python scripts/prep_real_kaggle_data.py --src /tmp/kaggle_dl
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np

# ----- finance feature layout (must match usecases/finance_aml feature_names order) ----------------
FIN_NAMES = ["amount", "velocity_24h", "cross_border", "night_tx",
             "new_payee", "amount_zscore", "device_risk", "geo_mismatch"]
#: Provenance text is NOT written here. It comes from the registry, which is also what the browser
#: prints and what tests/test_demo_provenance.py asserts these JSON files against. Re-declaring the
#: strings locally is how this script previously re-introduced a licence claim that had already been
#: corrected everywhere else.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qsh.models.registry import BUNDLED_DEMO_SAMPLES  # noqa: E402

FIN_PROVENANCE = BUNDLED_DEMO_SAMPLES["finance_aml"][0]
SAT_PROVENANCE = BUNDLED_DEMO_SAMPLES["satellite_comms"][0]


def _paysim_features(df):
    """Map REAL PaySim columns -> the 8 AML features (in FIN_NAMES order). 5 are direct real fields
    (amount, velocity_24h, night_tx, new_payee, amount_zscore); 3 (cross_border, device_risk,
    geo_mismatch) repurpose PaySim fraud signatures since PaySim has no geo/device fields."""
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
        f.write("# PaySim simulator output, i.e. SYNTHETIC transactions (Kaggle ealaxi/paysim1, CC-BY-SA-4.0; "
                "Lopez-Rojas et al. 2016). PaySim is a SIMULATOR (synthetic transactions). "
                "Columns = 8 AML features + isFraud.\n")
        f.write("\t".join(FIN_NAMES) + "\tisFraud\n")
        for row, label in zip(Xs, ys):
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


def prep_satellite(src: Path, repo: Path, n_channels: int = 6, length: int = 900, window: int = 8,
                   anchor: int = 350, max_anom: int = 150, seed: int = 0):
    import csv as _csv
    from sklearn.linear_model import LogisticRegression

    nasa = src / "nasa"
    labels_csv = nasa / "labeled_anomalies.csv"
    test_dir = nasa / "data" / "data" / "test"
    rng = np.random.default_rng(seed)

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

    # build windows (channel-major, matching the backend loader) -> train a real linear anomaly model for the demo
    C, W = len(names), window
    N = length - W + 1
    Xw = np.zeros((N, C * W), dtype=float)
    yw = np.zeros(N, dtype=int)
    for i in range(N):
        Xw[i] = np.concatenate([telem[i:i + W, c] for c in range(C)])
        yw[i] = int(anom[i:i + W].max())
    clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(Xw, yw)
    W2 = [list(map(float, np.zeros(C * W))), list(map(float, clf.coef_[0]))]
    b2 = [0.0, float(clf.intercept_[0])]
    feat_names = [f"{names[c]}[t{t}]" for c in range(C) for t in range(W)]
    demo_idx = np.concatenate([np.where(yw == 1)[0][:8], np.where(yw == 0)[0][:8]])
    rng.shuffle(demo_idx)
    demo = {
        "provenance": SAT_PROVENANCE, "channel_names": names, "names": feat_names,
        "windows": [[round(float(v), 5) for v in Xw[i]] for i in demo_idx],
        "labels": [int(yw[i]) for i in demo_idx],
        "model": {"W": W2, "b": b2},
        "real_model": "linear telemetry-anomaly classifier (logistic) over real NASA channels, run under CKKS",
    }
    out_json = repo / "web" / "public" / "data" / "satellite" / "telemetry.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(demo))
    print(f"[satellite] wrote {out_json} ({out_json.stat().st_size} B); anomaly_train_acc="
          f"{(clf.predict(Xw)==yw).mean():.3f}")


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

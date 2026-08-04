#!/usr/bin/env python3
"""Build **model-native** demo inputs so the browser can actually run the published HF models.

Why this exists
---------------
Until Round 18 the browser tier could never have worked, and a 401 on the private HF repos was
masking it. Each use-case page bundles a small sample for the *crypto* pipeline -- 8 features, so a
CKKS linear layer and a garbled-circuit argmax stay tractable in a tab -- but the model published
for that page was trained on a different corpus with a different input entirely:

    page        bundled pipeline sample              published model expects
    finance     PaySim, 8 engineered features        AMLworld, 32 graph/temporal features
    medical     Wisconsin WDBC, 8 tabular features   BreastMNIST, [1,1,28,28] images
    satellite   NASA SMAP/MSL, 6ch x 8 steps         ESA-ADB ch 41-46, [1,250,6] windows

So the fix is not a tensor reshape. Each published model needs a real sample drawn from the
distribution it was actually trained on. This script writes those, next to -- not instead of -- the
pipeline samples, which keep doing their own job.

Preprocessing travels WITH the data, deliberately
-------------------------------------------------
Writing this script, the first attempt scored the medical ResNet-18 at 0.4615 on the real test
split against a recorded 0.8782 -- below the 0.7308 majority-class baseline. The model was fine.
The mistake was applying `/255` and omitting the `(x-0.5)/0.5` step on line 65 of
`scripts/train/medical_breastmnist.py`. A browser reimplementing normalisation by hand would make
exactly that mistake and show a confidently wrong number with no error anywhere.

So every file this writes carries an explicit ``input.preprocessing`` list that the browser applies
verbatim, plus ``expected.accuracy`` measured here on these very rows. The browser re-checks its own
result against that figure and reports a mismatch instead of rendering it. That converts "MEASURED"
from an assertion into something the page proves each time it runs.

Usage (maintainer, one-time; not part of any runtime path)::

    export QSH_DATA_ROOT=$PWD/data-full
    python scripts/prep_model_inputs.py --usecase medical_fl
    python scripts/prep_model_inputs.py --usecase finance_aml
    python scripts/prep_model_inputs.py --usecase satellite_comms

Corpora are fetched by the commands in ``src/qsh/models/datasets.py`` (AMLWORLD_FETCH, ESA_FETCH);
BreastMNIST is fetched by ``medmnist`` itself. None of them are redistributed by this repository --
only the small derived samples below, whose licences are recorded in DATA_LICENSES.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

OUT_ROOT = REPO / "web" / "public" / "data"
#: Keep the payload small enough to ship. Large enough that the accuracy self-check is meaningful
#: rather than a coin flip on a handful of rows.
N_SAMPLES = 48


def _session(model_dir: str, prefer_int8: bool = True):
    """The published graph, preferring the int8 the browser tier actually downloads."""
    import onnxruntime as ort

    base = REPO / "models" / model_dir / "onnx"
    for name in (["model_int8.onnx", "model.onnx"] if prefer_int8 else ["model.onnx"]):
        p = base / name
        if p.exists():
            return ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]), name
    raise FileNotFoundError(f"no ONNX graph under {base}")


def _write(usecase: str, payload: dict) -> Path:
    out = OUT_ROOT / _DIR[usecase] / "model_input.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    size = out.stat().st_size
    print(f"  wrote {out.relative_to(REPO)}  ({size/1024:.0f} KB)")
    print(f"    expected accuracy on these rows: {payload['expected']['accuracy']:.4f} "
          f"(majority-class baseline {payload['expected']['majority_baseline']:.4f})")
    return out


_DIR = {"medical_fl": "medical", "finance_aml": "finance", "satellite_comms": "satellite"}


# --------------------------------------------------------------------------------------------- #
#  medical_fl -- BreastMNIST 28x28, the corpus medical_fl.resnet18 was actually trained on
# --------------------------------------------------------------------------------------------- #
def build_medical(rng: np.random.Generator) -> dict:
    import os

    from medmnist import INFO, BreastMNIST

    root = os.path.join(os.environ["QSH_DATA_ROOT"], "raw", "medmnist")
    ds = BreastMNIST(split="test", download=True, root=root)
    imgs, labels = ds.imgs, ds.labels.ravel()

    # Stratified so the sample cannot be all-majority; the accuracy self-check would be meaningless.
    idx = np.concatenate([
        rng.choice(np.flatnonzero(labels == c), size=N_SAMPLES // 2, replace=False)
        for c in (0, 1)
    ])
    rng.shuffle(idx)
    x_u8, y = imgs[idx], labels[idx]

    x = x_u8.astype(np.float32)[:, None, :, :] / 255.0
    x = (x - 0.5) / 0.5                               # train/medical_breastmnist.py:64-65
    sess, graph = _session("medical_breastmnist_resnet18")
    pred = sess.run(None, {sess.get_inputs()[0].name: x})[0].argmax(1)

    return {
        "provenance": (
            "real MedMNIST v2 BreastMNIST test-split images (Yang et al., Scientific Data 10:41, "
            "2023; CC BY 4.0), derived from the Al-Dhabyani et al. BUSI ultrasound dataset. These "
            "are the images medical_fl.resnet18 was trained and evaluated on -- unlike the Wisconsin "
            "WDBC sample this page uses for the FHE/MPC pipeline, which is a different corpus."
        ),
        "input": {
            "shape": [1, 1, 28, 28], "dtype": "float32",
            # Applied by the browser verbatim. Omitting the second step scores 0.4615 instead of
            # 0.8782 -- silently, with no error. See the module docstring.
            "preprocessing": ["divide by 255", "subtract 0.5", "divide by 0.5"],
            "layout": "samples[i] is a flat row-major 28*28 uint8 image, values 0-255",
        },
        "classes": INFO["breastmnist"]["label"],
        "samples": x_u8.reshape(len(idx), -1).astype(int).tolist(),
        "labels": y.astype(int).tolist(),
        "expected": {
            "graph": graph,
            "accuracy": float((pred == y).mean()),
            "majority_baseline": float(max(np.bincount(y)) / len(y)),
            "note": ("Measured here with onnxruntime on exactly these rows. The browser must "
                     "reproduce it to within tolerance or report a mismatch rather than render it."),
        },
    }


# --------------------------------------------------------------------------------------------- #
#  finance_aml -- AMLworld HI-Small, the 32 engineered features finance_aml.gbm expects
# --------------------------------------------------------------------------------------------- #
def build_finance(rng: np.random.Generator) -> dict:
    from qsh.models import datasets as D

    # Reuse the training script's own feature engineering rather than reimplementing 32 columns --
    # a second implementation would drift from the model on the first edit.
    from train.finance_aml import FEATURES, graph_features   # type: ignore

    # load_amlworld normalises "Is Laundering" -> "label" (datasets.py:84-91); the training script
    # reads that name too, so use it rather than the raw CSV header.
    df = D.load_amlworld(nrows=400_000)
    if "label" not in df:
        raise SystemExit(f"AMLworld frame has no 'label' column; got {list(df.columns)[:12]}")
    y_all = df["label"].to_numpy().astype(int)

    feats = graph_features(df)
    X_all = np.asarray(feats[list(FEATURES)].to_numpy(dtype=np.float32)
                       if hasattr(feats, "columns") else feats, dtype=np.float32)

    pos = np.flatnonzero(y_all == 1)
    neg = np.flatnonzero(y_all == 0)
    # 1-in-981 prevalence: a random draw would almost surely contain zero positives and the
    # self-check would say nothing. Force a visible minority so the number means something.
    take_pos = min(len(pos), N_SAMPLES // 4)
    idx = np.concatenate([rng.choice(pos, take_pos, replace=False),
                          rng.choice(neg, N_SAMPLES - take_pos, replace=False)])
    rng.shuffle(idx)
    x, y = X_all[idx], y_all[idx]

    sess, graph = _session("finance_aml_gbm")
    out = sess.run(None, {sess.get_inputs()[0].name: x})
    scores = next((o for o in out if getattr(o, "ndim", 0) == 2 and o.shape[1] > 1), out[0])
    pred = np.asarray(scores).argmax(1)

    return {
        "provenance": (
            "real IBM AMLworld HI-Small transactions (Altman et al., Kaggle "
            "ealtman2019/ibm-transactions-for-anti-money-laundering-aml, CDLA-Sharing-1.0). IBM "
            "states the corpus is entirely SYNTHETIC -- generated by a multi-agent virtual world, "
            "not real customer data. These are the engineered features finance_aml.gbm was trained "
            "on, unlike the 8-feature PaySim sample this page uses for the FHE/MPC pipeline."
        ),
        "input": {"shape": [1, len(FEATURES)], "dtype": "float32",
                  "preprocessing": [], "layout": "samples[i] is one row of the 32 features below"},
        "feature_names": list(FEATURES),
        "samples": np.asarray(x, dtype=float).round(6).tolist(),
        "labels": y.astype(int).tolist(),
        "expected": {
            "graph": graph,
            "accuracy": float((pred == y).mean()),
            "majority_baseline": float(max(np.bincount(y)) / len(y)),
            "note": ("Stratified to contain positives -- AMLworld's real prevalence is ~1 in 981, "
                     "so an unstratified draw would be all-negative and accuracy would be vacuous."),
        },
    }


# --------------------------------------------------------------------------------------------- #
#  satellite_comms -- ESA-ADB Mission1 channels 41-46, [1,250,6] windows
# --------------------------------------------------------------------------------------------- #
def build_satellite(rng: np.random.Generator) -> dict:
    from qsh.models import datasets as D

    seq, chans = 250, list(D.ESA_MISSION1_LIGHTWEIGHT)
    frame = D.load_esa_mission1(channels=chans)
    values = frame[chans].to_numpy(dtype=np.float32)
    truth = frame["anomaly"].to_numpy().astype(int) if "anomaly" in frame else np.zeros(len(frame), int)

    starts = np.arange(0, len(values) - seq, seq)
    win_lab = np.array([int(truth[s:s + seq].any()) for s in starts])
    pos, neg = np.flatnonzero(win_lab == 1), np.flatnonzero(win_lab == 0)
    take_pos = min(len(pos), N_SAMPLES // 3)
    pick = np.concatenate([rng.choice(pos, take_pos, replace=False),
                           rng.choice(neg, min(len(neg), N_SAMPLES - take_pos), replace=False)])
    rng.shuffle(pick)

    x = np.stack([values[starts[i]:starts[i] + seq] for i in pick]).astype(np.float32)
    y = win_lab[pick]
    sess, graph = _session("satellite_telemanom")
    out = np.asarray(sess.run(None, {sess.get_inputs()[0].name: x})[0])
    pred = (out.reshape(len(x), -1).max(1) > 0.5).astype(int)

    return {
        "provenance": (
            "real ESA Anomaly Dataset (ESA-ADB) Mission1 lightweight channels 41-46 "
            "(Zenodo 12528696, CC BY 3.0 IGO). Missions are anonymised Mission1/2/3 -- they are not "
            "named Sentinel anything. These are the windows satellite_comms.telemanom was trained "
            "on, unlike the NASA SMAP/MSL sample this page uses for the FHE/MPC pipeline."
        ),
        "input": {"shape": [1, seq, len(chans)], "dtype": "float32", "preprocessing": [],
                  "layout": f"samples[i] is {seq} timesteps x {len(chans)} channels, row-major"},
        "channels": chans,
        "samples": x.round(6).tolist(),
        "labels": y.astype(int).tolist(),
        "expected": {
            "graph": graph,
            "accuracy": float((pred == y).mean()),
            "majority_baseline": float(max(np.bincount(y)) / len(y)),
            "note": "Window is anomalous if any timestep in it is labelled an anomaly.",
        },
    }


BUILDERS = {"medical_fl": build_medical, "finance_aml": build_finance,
            "satellite_comms": build_satellite}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usecase", choices=sorted(BUILDERS), required=True)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    import os
    if not os.environ.get("QSH_DATA_ROOT"):
        raise SystemExit("set QSH_DATA_ROOT to the raw-corpus root (see src/qsh/models/datasets.py)")

    print(f"building model-native demo input for {args.usecase} ...")
    payload = BUILDERS[args.usecase](np.random.default_rng(args.seed))
    _write(args.usecase, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CT-FM's probe, a teacher candidate of the 3D nodule student (its pre-registration, "The arms").

CT-FM's feature extractor (Apache-2.0, `project-lighter/ct_fm_feature_extractor` at revision 527b0d1c, Pai et al.,
arXiv:2501.09001) is MONAI's SegResEncoder with the configuration its repository holds: 3 spatial dimensions, one input
channel, 32 initial filters, blocks down (1, 2, 2, 4, 4). It is built here from MONAI and loaded from the pinned
safetensors with every key matched, so no other package's code runs. The input is prepared as the model card
prescribes: orientation SPL (the crops' axes z, y, x are S, P, L), HU scaled from [-1024, 2048] to [0, 1] and clipped;
its last feature map is averaged to 512 features.

Only training patients' cubes exist for it (make_crops.py --hu-side 64 --split ... --part train), so it touches no
validation or test crop. An L2-regularised logistic regression on the standardised features is scored out of fold
over the five folds of the training patients, its strength chosen inside each training fold by grouped
cross-validation over patients; the probabilities go to W/teachers/ctfm_probe.json.

    python scripts/train/medical_nodule3d_teacher_ctfm.py --crops DIR --work W --weights PATH/model.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import medical_nodule3d_student as s  # noqa: E402

CTFM_SHA256 = "b521ff13764ad0fce67f8ad2e5aa9ccc0823f69e4544b446581d8a2dee686215"
CTFM_REVISION = "527b0d1c262b433396c85ff2de84c427b26945a1"
CONFIG = {"spatial_dims": 3, "in_channels": 1, "init_filters": 32, "blocks_down": (1, 2, 2, 4, 4)}


def encoder(weights: Path):
    import torch
    from monai.networks.nets.segresnet_ds import SegResEncoder
    from safetensors.torch import load_file

    if hashlib.sha256(weights.read_bytes()).hexdigest() != CTFM_SHA256:
        raise SystemExit(f"{weights} is not CT-FM's pinned model.safetensors")
    net = SegResEncoder(**CONFIG)
    state = load_file(weights)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"CT-FM's weights do not fit MONAI's SegResEncoder: missing {missing[:5]}, "
                         f"unexpected {unexpected[:5]}")
    return net.eval().to("cuda" if torch.cuda.is_available() else "cpu")


def features(net, cubes_hu: np.ndarray) -> np.ndarray:
    import torch

    dev = next(net.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(cubes_hu), 16):
            x = np.clip((cubes_hu[i:i + 16].astype(np.float32) + 1024.0) / 3072.0, 0.0, 1.0)[:, None]
            fmap = net(torch.tensor(x).to(dev))[-1]
            out.append(torch.nn.functional.adaptive_avg_pool3d(fmap, 1).flatten(1).double().cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--side", type=int, default=64)
    args = ap.parse_args()
    s.check_preregistration()

    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    split = s.read_split()
    cubes, records = s.load_crops(args.crops)
    _, y, patients = s.labelled(cubes, records, split["parts"], "train")
    keys = [(r["series"], r["cluster"]) for r in records
            if r["patient"] in set(split["parts"]["train"]) and s.label_of(r["malignancy"]) is not None]
    hu_meta = json.loads((args.crops / f"crops_{args.side}mm_hu.json").read_text())["records"]
    hu = np.load(args.crops / f"crops_{args.side}mm_hu.npz")["crops"]
    where = {(m["series"], m["cluster"]): i for i, m in enumerate(hu_meta)}
    assert all(k in where for k in keys), "a labelled training crop has no HU cube"
    allowed = set(split["parts"]["train"])
    assert all(m["patient"] in allowed for m in hu_meta), "an HU cube belongs to a patient outside training"
    feats = features(encoder(args.weights), hu[[where[k] for k in keys]])

    fold = s.folds_of(patients, y)
    f = np.array([fold[p] for p in patients])
    groups = np.array(patients)
    oof = np.full(len(y), np.nan)
    chosen = {}
    for k in range(s.FOLDS):
        tr, te = f != k, f == k
        scale = StandardScaler().fit(feats[tr])
        # The inner folds group by patient, so no patient's nodules are on both sides of a choice of strength.
        inner = list(GroupKFold(n_splits=4).split(feats[tr], y[tr], groups[tr]))
        lr = LogisticRegressionCV(Cs=np.logspace(-4, 2, 13), cv=inner, scoring="roc_auc", max_iter=5000)
        lr.fit(scale.transform(feats[tr]), y[tr])
        oof[te] = lr.predict_proba(scale.transform(feats[te]))[:, 1]
        chosen[k] = float(lr.C_[0])
    s.write_teacher(args.work, "ctfm_probe", patients, y, oof,
                    {"revision": CTFM_REVISION, "weights_sha256": CTFM_SHA256, "cube_mm": args.side,
                     "features": int(feats.shape[1]), "C_by_fold": chosen, "environment": s.environment()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""A classical model with the QCNN's own hypothesis class, measured on the QCNN's split (Round 86, the owner's choice).

The circuit's readout is a linear form in the 3^n products of sines and cosines of the encoded angles (`src/qsh/qhe/qcnn_pauli.py`). So a logistic regression on those products can represent
what the circuit computes, and more. That is the comparison Schreiber, Eisert and Meyer (PRL 131, 100803, 2023) call a classical surrogate, and the benchmarking study of Bowles, Ahmed and Schuld
(arXiv:2403.07059) is why a QML figure reported without a classical comparator says little. The QCNN has no teacher, since it is trained on labels, so this and the logistic regression on the raw inputs,
which the trainer already records, are its comparators.

This adds `classical_surrogate` to each shipped artifact without retraining the circuit, on the identical split the trainer used (checked: the logistic regression it refits must reproduce the recorded
figure to the last digit, or the script stops). The trainer writes the same key for a future retrain.

    python scripts/train/qcnn_classical_surrogate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "train"))

from qcnn_variational import classical_surrogate, load, shuffled_split  # noqa: E402

ARTIFACTS = (
    ("medical", 8, "models/qcnn_medical/weights.json", "web/public/data/medical/qcnn_weights.json"),
    ("finance", 4, "models/qcnn_finance/weights.json", "web/public/data/finance/qcnn_weights.json"),
    ("telecom", 8, "models/qcnn_telecom/weights.json", "web/public/data/telecom/qcnn_weights.json"),
    ("telecom", 7, "models/qcnn_telecom_q7/weights.json", "web/public/data/telecom/qcnn_weights_q7.json"),
)


def main() -> int:
    for usecase, _wires, model_path, web_path in ARTIFACTS:
        w = json.loads((REPO / model_path).read_text())
        X, y, _cols, _prov = load(usecase)
        perm, cut, vcut = shuffled_split(len(y), int(w["training"]["seed"]))
        X, y = X[perm], y[perm]
        n = len(y)
        mu, sd = np.asarray(w["feature_mean"]), np.asarray(w["feature_std"])
        Xs = np.tanh((X[:, :len(mu)] - mu) / np.where(sd == 0, 1.0, sd))
        tr, te = slice(0, cut), slice(vcut, n)
        # The check that this is the trainer's split: the same estimator, refitted, must give the recorded figure.
        lr = LogisticRegression(max_iter=2000).fit(Xs[tr], y[tr])
        refit = round(float(lr.score(Xs[te], y[te])), 6)
        if refit != w["classical_lr_accuracy"]:
            raise SystemExit(f"{model_path}: logistic regression refitted to {refit}, the artifact records {w['classical_lr_accuracy']}: not the trainer's split")
        readout_q = int((w.get("readout") or {}).get("readout_qubits", 0) or 0)
        s = classical_surrogate(Xs, y, tr, te, float(w.get("encoding_scale", 1.0)), readout_q)
        block = {"classical_surrogate": s}
        rendered = json.dumps(block, indent=1)[2:-2]  # the "classical_surrogate": {...} entry at the file's indent
        for path in (model_path, web_path):
            target = REPO / path
            text = target.read_text()
            data = json.loads(text)
            at = text.index('\n "note":')
            if "classical_surrogate" in data:      # a rerun replaces the entry
                text = text[:text.index('\n "classical_surrogate":')] + "\n" + rendered + "," + text[at:]
            else:
                text = text[:at] + "\n" + rendered + "," + text[at:]
            json.loads(text)
            target.write_text(text)
        print(f"  {model_path}: {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

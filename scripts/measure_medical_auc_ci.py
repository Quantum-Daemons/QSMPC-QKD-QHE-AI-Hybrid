"""How far one AUROC on 156 test images can be trusted (Round 86).

`medical_fl.resnet18` records AUC 0.9147 against a published 0.901. The test split is the official BreastMNIST one, 42
malignant and 114 benign images, and a single training run. This script scores each published medical model through its
own ONNX graph on its own `eval_sample.json`, requires the record's AUC back to six decimals, and writes the DeLong
interval and a stratified percentile bootstrap beside it. It writes `docs/measurements/medical-auc-ci-<date>.json` and copies
the interval into `models/metrics.json`. Run with the CPU's onnxruntime; nothing here is timed.

    .venv/bin/python scripts/measure_medical_auc_ci.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

KEYS = {"medical_fl.resnet18": "models/medical_breastmnist_resnet18", "medical_fl.resnet50": "models/medical_breastmnist_resnet50",
        "medical_fl.student": "models/medical_breastmnist_student"}


def _score(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort

    sample = json.loads((model_dir / "eval_sample.json").read_text())
    shape = sample["input"]["shape"]
    x = np.asarray(sample["samples"], dtype=np.float32).reshape(-1, *shape[1:])
    if sample["input"]["preprocessing"]:      # "divide by 255", "subtract 0.5", "divide by 0.5"
        x = (x / 255.0 - 0.5) / 0.5
    sess = ort.InferenceSession(str(model_dir / "onnx" / "model.onnx"), providers=["CPUExecutionProvider"])
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    return np.asarray(sample["labels"]), logits[:, 1] - logits[:, 0]


def main() -> int:
    from qsh.models import metrics

    path = REPO / "models" / "metrics.json"
    record = json.loads(path.read_text())
    out: dict[str, dict] = {}
    for key, rel in KEYS.items():
        y, s = _score(REPO / rel)
        ci = metrics.delong_auc_ci(y, s)
        # The record's AUC is the trainer's, in PyTorch; this is the served graph's. One pair of images changing order moves
        # an AUC by 1 / (m n) = 2.1e-4 here, and float32 export moves a few pairs (ResNet-50: 2.5), so five pairs is the
        # tolerance; anything larger means the graph is not the model the record describes.
        five_pairs = 5 / (ci["n_pos"] * ci["n_neg"])
        if abs(ci["auc"] - record[key]["auc"]) > five_pairs:
            raise SystemExit(f"{key}: the published graph gives AUC {ci['auc']}; the record says {record[key]['auc']}")
        boot = metrics.bootstrap_auc_ci(y, s)
        out[key] = {"auc_onnx_graph": round(ci["auc"], 6), "auc_record": round(record[key]["auc"], 6), "delong_95": [round(ci["lo"], 4), round(ci["hi"], 4)],
                    "delong_se": round(ci["se"], 4), "bootstrap_95": [round(boot["lo"], 4), round(boot["hi"], 4)],
                    "n_boot": boot["n_boot"], "bootstrap_seed": boot["seed"], "n_pos": ci["n_pos"], "n_neg": ci["n_neg"]}
        print(key, out[key])
        record[key]["auc_ci_95"] = {"lo": out[key]["delong_95"][0], "hi": out[key]["delong_95"][1], "method": "DeLong",
                                    "n_pos": ci["n_pos"], "n_neg": ci["n_neg"],
                                    "source": "scripts/measure_medical_auc_ci.py"}
    today = dt.date.today().isoformat()
    (REPO / "docs" / "measurements" / f"medical-auc-ci-{today}.json").write_text(
        json.dumps({"measured": today, "script": "scripts/measure_medical_auc_ci.py",
                    "positive_class": "class 1, normal or benign, as the record's AUC is computed", "results": out}, indent=1) + "\n")
    path.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

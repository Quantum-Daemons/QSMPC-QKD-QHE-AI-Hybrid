#!/usr/bin/env python3
"""The chest X-ray model's last layer, as numbers a browser can evaluate under CKKS (Round 86, split inference).

The owner's fourth choice of 2026-11-14: on the medical page, the published backbone (`onnx/features_int8.onnx`) runs in
the visitor's tab on one of the model's 16 published films, and only the 512 pooled features are encrypted; the network's
own last layer, a 512-weight linear map to the effusion logit, is evaluated on them under CKKS (`encryptedLinear` in
web/src/lib/clientFHE.ts, unchanged). This writes that layer for the page.

The layer is read from the published `model.safetensors` (`fc.weight`, 1 x 512, and `fc.bias`) and checked against the
published graphs before anything is written: on each of the 16 films of `eval_sample.json`, the head applied to the float32
features graph must give the float32 classifier graph's logit z to 1e-4, and the head applied to the int8 features graph
must give the same decision at the operating point as the int8 classifier graph, which the page's own re-scoring uses.
The graphs must hash to their pins in `models/hub_file_hashes.json`.

Writes web/public/data/medical/cxr_head.json and docs/measurements/cxr-head-2026-11-15.json.

Usage: python -I scripts/export_cxr_head.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
KEY = "medical_fl.cxr_fedavg"
ART = REPO / "models" / "medical_cxr14_fedavg_resnet18"
CARD = REPO / "web" / "public" / "data" / "medical" / "cxr_head.json"
RECORD = REPO / "docs" / "measurements" / "cxr-head-2026-11-15.json"
TOL = 1e-4


def main() -> int:
    import onnxruntime as ort
    from safetensors.numpy import load_file

    from qsh.models import registry

    spec = registry.get(KEY)
    pins = json.loads((REPO / "models" / "hub_file_hashes.json").read_text())[KEY]
    if pins["revision"] != spec.hf_revision:
        raise SystemExit(f"the pinned revision {pins['revision']} is not the registry's {spec.hf_revision}")
    for rel in ("onnx/model.onnx", "onnx/model_int8.onnx", "onnx/features.onnx", "onnx/features_int8.onnx", "eval_sample.json"):
        if hashlib.sha256((ART / rel).read_bytes()).hexdigest() != pins["files"][rel]:
            raise SystemExit(f"{rel} is not the pinned published file")
    w = load_file(str(ART / "model.safetensors"))
    W, b = w["fc.weight"].astype(np.float64), w["fc.bias"].astype(np.float64)
    assert W.shape == (1, 512) and b.shape == (1,), (W.shape, b.shape)
    sample = json.loads((ART / "eval_sample.json").read_text())
    thr = float(sample["expected"]["decision_threshold"])
    z_cut = math.log(thr / (1 - thr))
    x = np.array(sample["samples"], dtype=np.float32).reshape(-1, 1, 224, 224) / 255.0

    def run(name: str) -> np.ndarray:
        s = ort.InferenceSession(str(ART / "onnx" / name), providers=["CPUExecutionProvider"])
        return s.run(None, {s.get_inputs()[0].name: x})[0].astype(np.float64)

    f32, f8 = run("features.onnx"), run("features_int8.onnx")
    z32, z8 = run("model.onnx")[:, 1], run("model_int8.onnx")[:, 1]
    head32, head8 = f32 @ W[0] + b[0], f8 @ W[0] + b[0]
    err32 = float(np.abs(head32 - z32).max())
    same_int8 = int(((head8 >= z_cut) == (z8 >= z_cut)).sum())
    if err32 > TOL:
        raise SystemExit(f"the head on the float32 features misses the classifier's logit by {err32:.3g} (> {TOL})")
    if same_int8 != len(z8):
        raise SystemExit(f"the head on the int8 features decides {same_int8} of {len(z8)} films as the int8 classifier does")
    check = {
        "films": int(len(z32)),
        "max_abs_logit_difference_float32": err32,
        "max_abs_logit_difference_int8_features_vs_int8_classifier": float(np.abs(head8 - z8).max()),
        "decisions_agreeing_int8": same_int8,
        "operating_point_logit": z_cut, "decision_threshold": thr,
    }
    card = {
        "_about": ("The chest X-ray model's last layer for the medical page's split inference; written by "
                   "scripts/export_cxr_head.py from the published model.safetensors and checked against the published graphs."),
        "model": KEY, "hf_repo": spec.hf_repo_id, "hf_revision": spec.hf_revision,
        "features_graph": "onnx/features_int8.onnx", "features": 512,
        "weights": [float(np.float32(v)) for v in W[0]], "bias": float(np.float32(b[0])),
        "decision_threshold": thr, "operating_point_logit": z_cut,
        "licence": sample["licence"], "licence_url": sample["licence_url"], "attribution": sample["attribution"],
        "modifications": "the network's own last layer (fc.weight and fc.bias) read from the published model.safetensors, unchanged",
        "check": check,
    }
    CARD.write_text(json.dumps(card, indent=1) + "\n")
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps({"_about": card["_about"], "model": KEY, "hf_revision": spec.hf_revision,
                                  "check": check}, indent=1) + "\n")
    print(json.dumps(check, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

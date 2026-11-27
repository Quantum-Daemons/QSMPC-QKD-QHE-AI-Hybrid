#!/usr/bin/env python3
"""The DeFi QCNN's export criterion on the seed it would ship (Round 86, batch 8).

`references/defi-qcnn-aave-v3-preregistration.md`, criterion 2: the page's clear contraction of the arm-1 seed with the
best validation average precision at eight wires (`clearReadout` in `web/src/lib/clientQcnnTreeCkks.ts`) equals the
state-vector readout within 1e-5 on every row of the sample the page ships, and no decision differs. This checks it on
every one of the 182,351 test-window rows, of which any sample the page ships is a subset. The state-vector readout is
the arm's saved test logits (the trainer's circuit) through the seed's scale and bias; the contraction is the page's
TypeScript, bundled and run in node as the finance check runs it, and Python's `qsh.qhe.qcnn_tree` beside it. Writes
`export_check.json` in the comparison's work folder, which the record reads.

Usage: QSH_DATA_ROOT=<data root> python -I scripts/check_defi_qcnn_export.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
TOL = 1e-5


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    from sklearn.metrics import average_precision_score as ap

    from qsh.qhe import qcnn_tree

    started = time.time()
    C = _module("defi_qcnn_comparison", REPO / "scripts" / "train" / "defi_qcnn_comparison.py")
    page_readout = _module("check_finance_qcnn_export", REPO / "scripts" / "check_finance_qcnn_export.py").page_readout
    C.check_prereg()
    d = C.load_inputs()
    yv = d["y"][d["val"]]
    per = [(float(ap(yv, np.load(C.WORK / "qcnn8" / f"seed{s}" / "val_logits.npy").astype(np.float64))), s) for s in range(20)]
    seed = max(per, key=lambda r: (r[0], -r[1]))[1]                 # the record's rule: best validation AP, ties to the lower
    o = C.WORK / "qcnn8" / f"seed{seed}"
    w = json.loads((o / "weights.json").read_text())
    X = d["X"][d["te"]].astype(np.float64)
    scale, bias = float(w["scale"]), float(w["bias"])
    sv = (np.load(o / "test_logits.npy").astype(np.float64) - bias) / scale
    net = qcnn_tree.network(w)
    page_net = {"maps": [m.tolist() for m in net["maps"]], "readout": net["readout"].tolist(), "scale": scale, "bias": bias,
                "feature_mean": [float(v) for v in w["feature_mean"]], "feature_std": [float(v) for v in w["feature_std"]],
                "encoding_scale": float(w.get("encoding_scale", 1.0)), "n_qubits": int(w["n_qubits"])}
    py = qcnn_tree.readout(w, X)
    with tempfile.TemporaryDirectory() as tmp:
        page = page_readout(page_net, X, Path(tmp))
    dec = lambda r: (r * scale + bias) > 0  # noqa: E731
    out = {
        "seed": seed, "test_rows": int(len(X)), "tolerance": TOL,
        "page_vs_state_vector": {"max_abs_readout_difference": float(np.max(np.abs(page - sv))),
                                 "decisions_differing": int(np.sum(dec(page) != dec(sv)))},
        "python_vs_state_vector": {"max_abs_readout_difference": float(np.max(np.abs(py - sv))),
                                   "decisions_differing": int(np.sum(dec(py) != dec(sv)))},
        "page_vs_python": {"max_abs_readout_difference": float(np.max(np.abs(page - py)))},
        "state_vector": "the arm's test_logits.npy: the trainer's circuit in float32 on the GPU, through the seed's scale and bias",
        "wall_clock_s": round(time.time() - started, 1),
    }
    pv = out["page_vs_state_vector"]
    out["criterion_met"] = bool(pv["max_abs_readout_difference"] <= TOL and pv["decisions_differing"] == 0)
    (C.WORK / "export_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

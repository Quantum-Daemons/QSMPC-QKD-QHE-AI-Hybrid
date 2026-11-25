#!/usr/bin/env python3
"""The satellite QCNNs' tree network under CKKS on every test row, beside the Pauli-form path (pre-registration D).

references/satellite-qcnn-tn-preregistration.md, section D and addendum 3. For each corpus's seed-0 artifact: the stage
maps (`qsh.qhe.qcnn_tree.network`), the test rows in raw units and their clear readouts (the contraction, equal to the
state vector) go to `web/scripts/check_qcnn_tree_ckks.mjs`, which runs `clientQcnnTreeCkks.ts` on the real node-seal
over every test row, one row a slot, and the existing Pauli-form path (`clientQcnnCkks.ts`, one row at a time) on the
first rows. The criteria: the decrypted readout within 1e-3 of the clear one on every test row, and no decision changed.

    python scripts/measure_qcnn_tree_ckks.py --artifacts DIR -o docs/measurements/satellite-qcnn-tree-ckks-<date>.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "train"))

CRITERION = 1e-3
PAULI_ROWS = 40


def main() -> int:
    from qcnn_variational import load, split_for

    from qsh.qhe import qcnn_tree
    from qsh.qhe.qcnn_trained import _run_circuit

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    t0 = time.time()
    cases = []
    for short, usecase in (("opssat", "satellite_opssat"), ("esa", "satellite_esa")):
        wpath = args.artifacts / f"{short}-s0" / "weights.json"
        w = json.loads(wpath.read_text())
        X, y, _, _ = load(usecase)
        order, _, vcut = split_for(usecase, y, 0)
        Xte = X[order[vcut:]]
        net = qcnn_tree.network(w)
        expected = qcnn_tree.readout(w, Xte)
        spot = np.array([_run_circuit(x, w)[0] for x in Xte[:200]])
        if np.max(np.abs(spot - expected[:200])) > 1e-12:
            raise SystemExit(f"{short}: the contraction is not the state vector")
        cases.append({"name": short, "weights_path": str(wpath), "pauli_rows": PAULI_ROWS,
                      "network": {"maps": [m.tolist() for m in net["maps"]], "readout": net["readout"].tolist(),
                                  "scale": float(w["scale"]), "bias": float(w["bias"]),
                                  "feature_mean": list(map(float, w["feature_mean"])), "feature_std": list(map(float, w["feature_std"])),
                                  "encoding_scale": float(w.get("encoding_scale", 1.0)), "n_qubits": int(w["n_qubits"])},
                      "rows": Xte.tolist(), "expected": expected.tolist()})
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.json", Path(tmp) / "out.json"
        src.write_text(json.dumps({"cases": cases}))
        r = subprocess.run(["node", str(REPO / "web" / "scripts" / "check_qcnn_tree_ckks.mjs"), str(src), str(dst)],
                           capture_output=True, text=True, cwd=REPO)
        print(r.stdout + r.stderr)
        if r.returncode:
            raise SystemExit(f"node exited {r.returncode}")
        result = json.loads(dst.read_text())
    for case in result["cases"]:
        tree = case["tree"]
        case["criterion_met"] = tree["max_abs_readout_error"] <= CRITERION and tree["decisions_differing"] == 0
        case["per_row"] = {"tree_ms": round(tree["total_ms"] / case["rows"], 4),
                           "tree_bytes_up": round(tree["up_bytes"] / case["rows"], 1),
                           "pauli_ms": round(case["pauli"]["totalMs"] / max(1, case["pauli"]["rows"]), 2),
                           "pauli_bytes_up": round(case["pauli"]["upBytes"] / max(1, case["pauli"]["rows"]), 1)}
    record = {"what": "the satellite QCNNs' tree network under CKKS on every test row, and the Pauli-form path on the first "
                      f"{PAULI_ROWS} of them",
              "preregistration": "references/satellite-qcnn-tn-preregistration.md, D and addendum 3",
              "criterion": {"max_abs_readout_error": CRITERION, "decisions_differing": 0},
              **result, "machine": platform.processor() or platform.machine(),
              "seconds": round(time.time() - t0, 1),
              "environment": {"git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                                    capture_output=True, text=True).stdout.strip(),
                              "python": platform.python_version(), "date": time.strftime("%Y-%m-%d")}}
    Path(args.output).write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

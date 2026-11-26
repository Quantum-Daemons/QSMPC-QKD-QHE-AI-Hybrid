#!/usr/bin/env python3
"""The finance comparison's export criterion on the seed it would ship (Round 86, batch 6).

`references/finance-qcnn-amlworld-comparison-preregistration.md`, criterion 1, takes the first pre-registration's export
check: the page's clear contraction of the shipped seed (`clearReadout` in `web/src/lib/clientQcnnTreeCkks.ts`) equals
the state-vector readout within 1e-5 on every test row, and no decision differs. The state-vector readout is the arm's
own saved test logits (`test_logits.npy`, the trainer's circuit on all 1,523,504 test-window rows) through the seed's
scale and bias; the contraction is the page's TypeScript, bundled and run in node on the same rows, and Python's
`qsh.qhe.qcnn_tree` beside it. Writes `export_check.json` in the comparison's work folder, which the record reads.

Usage: QSH_DATA_ROOT=<data root> python -I scripts/check_finance_qcnn_export.py
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
WEB = REPO / "web"
TOL = 1e-5
CHUNK = 100_000


def _comparison():
    spec = importlib.util.spec_from_file_location("finance_qcnn_comparison", REPO / "scripts" / "train" / "finance_qcnn_comparison.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def page_readout(net: dict, X: np.ndarray, tmp: Path) -> np.ndarray:
    """`clearReadout` from the page's module, in node, over every row, a chunk at a time."""
    (tmp / "x.bin").write_bytes(np.ascontiguousarray(X, dtype="<f8").tobytes())
    (tmp / "net.json").write_text(json.dumps(net))
    entry, bundle = tmp / "entry.ts", tmp / "bundle.mjs"
    entry.write_text(f'export {{ clearReadout }} from {json.dumps(str(WEB / "src" / "lib" / "clientQcnnTreeCkks.ts"))};\n')
    subprocess.run([str(WEB / "node_modules" / ".bin" / "esbuild"), str(entry), "--bundle", "--format=esm", "--platform=node",
                    f"--outfile={bundle}", "--log-level=error"], check=True, cwd=WEB)
    runner = tmp / "run.mjs"
    runner.write_text(f"""
import {{ readFileSync, writeFileSync }} from "node:fs";
import {{ clearReadout }} from {json.dumps(bundle.as_uri())};
const net = JSON.parse(readFileSync({json.dumps(str(tmp / "net.json"))}, "utf8"));
const buf = readFileSync({json.dumps(str(tmp / "x.bin"))});
const x = new Float64Array(buf.buffer, buf.byteOffset, buf.byteLength / 8);
const w = {X.shape[1]}, n = x.length / w, out = new Float64Array(n);
for (let s = 0; s < n; s += {CHUNK}) {{
  const e = Math.min(n, s + {CHUNK}), rows = [];
  for (let r = s; r < e; r++) rows.push(x.subarray(r * w, r * w + w));
  out.set(clearReadout(net, rows), s);
}}
writeFileSync({json.dumps(str(tmp / "out.bin"))}, Buffer.from(out.buffer));
""")
    subprocess.run(["node", "--max-old-space-size=8192", str(runner)], check=True, cwd=WEB)
    return np.frombuffer((tmp / "out.bin").read_bytes(), dtype="<f8").copy()


def main() -> int:
    from sklearn.metrics import average_precision_score as ap

    from qsh.qhe import qcnn_tree

    started = time.time()
    F = _comparison()
    F.check_prereg()
    d = F.load_inputs()
    yv = d["y"][d["val"]]
    per = []
    for s in range(20):
        o = F.WORK / "qcnn1" / f"seed{s}"
        per.append((float(ap(yv, np.load(o / "val_logits.npy").astype(np.float64))), s))
    seed = max(per, key=lambda r: (r[0], -r[1]))[1]                # the record's rule: best validation AP, ties to the lower
    o = F.WORK / "qcnn1" / f"seed{seed}"
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
    (F.WORK / "export_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

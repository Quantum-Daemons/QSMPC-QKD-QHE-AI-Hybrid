#!/usr/bin/env python3
"""The finance QCNN's fair comparison on AMLworld, as pre-registered (Round 86, batch 6).

references/finance-qcnn-amlworld-comparison-preregistration.md fixed the arms, their grids, the seeds and the metrics on
2026-11-26 before anything ran (the owner's choices of that day). This runs them and writes the record:

  --arm qcnn1 | qcnn2 | noent   the QCNN arms 1, 2 and 5, one trainer run a seed (scripts/train/qcnn_variational.py), each
                                into WORK/<arm>/seed<k>/ with its validation and test logits
  --comparators                 arms 3 and 4 on the CPU: the quantum kernels, logistic regression, the booster, the RBF
                                support vector machine and the matrix-product-state classifier
  --schmidt                     arm 6: Schmidt ranks of every arm-1 seed's state, by layer, on 1,000 test rows
  --record                      every arm's figures, intervals and paired differences into
                                docs/measurements/finance-qcnn-amlworld-comparison-2026-11-26.json

Arms 1 and 2 run on the GPU (the owner's approval of 2026-11-26 for their 40 runs); arm 5 on the CPU.

Usage: QSH_DATA_ROOT=<data root> python -I scripts/train/finance_qcnn_comparison.py --arm qcnn1 --seeds 0-19
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

PREREG = REPO / "references" / "finance-qcnn-amlworld-comparison-preregistration.md"
PREREG_SHA256 = "dc56325772e004894575ba59e71faf7e50d40f8250ac46c81f0d27a6b8322d2c"
WORK = REPO.parent / "tools_external" / "finance_qcnn_comparison_work"
RECORD = REPO / "docs" / "measurements" / "finance-qcnn-amlworld-comparison-2026-11-26.json"
TRAINER = REPO / "scripts" / "train" / "qcnn_variational.py"
COMMON = ["--usecase", "finance_amlworld", "--arch", "tree", "--layers", "3", "--epochs", "120", "--qubits", "8",
          "--threads", "4", "--select-on", "ap", "--save-scores"]
ARMS = {
    "qcnn1": {"flags": ["--encoding-scale", "0.5"], "device": "gpu"},
    "qcnn2": {"flags": ["--encoding-scale", "1", "--encoding", "quantile", "--reupload", "2"], "device": "gpu"},
    "noent": {"flags": ["--encoding-scale", "0.5", "--no-entangle"], "device": "cpu"},
}


def check_prereg() -> str:
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got}, not the committed {PREREG_SHA256}: run nothing")
    return got


def seeds_of(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def run_arm(arm: str, seeds: list[int]) -> int:
    spec = ARMS[arm]
    env = dict(os.environ)
    if spec["device"] == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""                 # arm 5 is pre-registered on the CPU
    for s in seeds:
        out = WORK / arm / f"seed{s}"
        out.mkdir(parents=True, exist_ok=True)
        if (out / "weights.json").exists() and (out / "test_logits.npy").exists():
            print(f"{arm} seed {s}: done already, kept")
            continue
        t0 = time.time()
        cmd = [sys.executable, "-I", str(TRAINER), *COMMON, *spec["flags"], "--seed", str(s), "--out", str(out)]
        with open(out / "train.log", "w") as log:
            code = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{arm} seed {s}: exit {code} in {time.time() - t0:.0f} s", flush=True)
        if code:
            return code
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--seeds", default="0-19")
    args = ap.parse_args()
    check_prereg()
    if args.arm:
        return run_arm(args.arm, seeds_of(args.seeds))
    ap.error("choose --arm (the comparators, the Schmidt check and the record follow in this file)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

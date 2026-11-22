#!/usr/bin/env python3
"""Repeat the GPU backend's CKKS decode and record whether any run differs from another or from plaintext.

HEonGPU v2.1.1, the submodule's pin, builds GPU-FFT at `b743607`. GPU-FFT's `5f2cd2c` ("Fix warp level race in last forward
FFT stages", 2026-10-30) adds a `__syncwarp()` that commit lacks, and HEonGPU's CKKS decoding runs that forward transform
(`decode_ckks` in `src/lib/host/ckks/encoder.cu`). A race of that kind shows as a decode that differs between runs of the same
ciphertext, so this decodes one ciphertext many times, and decodes fresh encryptions against their plaintext. The ring and
moduli are the configured ones (`FHEConfig`: N = 8192, moduli (60, 40, 40, 60), scale 2^40), as the backend uses them.

    python scripts/measure_heongpu_decode_repeat.py -o docs/measurements/heongpu-decode-repeat-<date>.json

The module is found in `native/heongpu_py/build` or in `QSH_HEONGPU_BUILD` (`scripts/build_heongpu.sh` builds it).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SAME, FRESH = 20000, 2000


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--heongpu-source", default=str(REPO / "submodules" / "HEonGPU"),
                    help="the HEonGPU checkout the module was built from, to record its version and its GPU-FFT pin")
    args = ap.parse_args()
    sys.path.insert(0, str(REPO / "src"))
    from qsh.fhe.backends import get_backend, heongpu_available

    if not heongpu_available():
        print("the HEonGPU module is not built here, or there is no GPU", file=sys.stderr)
        return 1
    b = get_backend("heongpu")
    rng = np.random.default_rng(2026)
    x = rng.normal(size=b.slot_count)
    ct = b.encrypt(x)
    first = np.asarray(b.decrypt_slots(ct))
    differing, run_dev, plain_dev = 0, 0.0, 0.0
    t0 = time.time()
    for _ in range(SAME):
        out = np.asarray(b.decrypt_slots(ct))
        d = float(np.max(np.abs(out - first)))
        differing += d != 0.0
        run_dev = max(run_dev, d)
        plain_dev = max(plain_dev, float(np.max(np.abs(out.real - x))))
    t1 = time.time()
    fresh_over, fresh_dev = 0, 0.0
    for _ in range(FRESH):
        y = rng.normal(size=b.slot_count)
        e = float(np.max(np.abs(np.asarray(b.decrypt_slots(b.encrypt(y))).real - y)))
        fresh_dev = max(fresh_dev, e)
        fresh_over += e > 1e-6
    t2 = time.time()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False).stdout.strip()
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=False).stdout.strip().splitlines()
    src = args.heongpu_source
    record = {
        "what": "the GPU backend's CKKS decode, repeated on one ciphertext and on fresh encryptions",
        "why": "HEonGPU v2.1.1 builds GPU-FFT at b743607, before GPU-FFT's 5f2cd2c (a warp-level race in the last forward "
               "FFT stages); CKKS decoding runs that transform",
        "machine": {"gpu": gpu, "nvcc": nvcc[-1] if nvcc else "", "kernel": platform.release(), "date": time.strftime("%Y-%m-%d")},
        "versions": {"HEonGPU": _git("-C", src, "describe", "--tags"),
                     "HEonGPU commit": _git("-C", src, "rev-parse", "HEAD"),
                     "GPU-FFT pinned by that commit": _git("-C", src, "rev-parse", "HEAD:thirdparty/GPU-FFT"),
                     "GPU-NTT pinned by that commit": _git("-C", src, "rev-parse", "HEAD:thirdparty/GPU-NTT")},
        "parameters": {"slots": b.slot_count, "seed": 2026, "same_ciphertext_decodes": SAME, "fresh_encryptions": FRESH},
        "results": {"decodes_differing_from_the_first": int(differing), "max_run_to_run_difference": run_dev,
                    "max_difference_from_plaintext_same_ciphertext": plain_dev,
                    "fresh_encryptions_beyond_1e-6": int(fresh_over), "max_difference_from_plaintext_fresh": fresh_dev,
                    "seconds": {"same_ciphertext": round(t1 - t0, 2), "fresh": round(t2 - t1, 2)}},
    }
    Path(args.output).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["results"]))
    return 0 if differing == 0 and fresh_over == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

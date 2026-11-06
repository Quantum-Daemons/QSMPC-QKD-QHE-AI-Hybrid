#!/usr/bin/env python3
"""The CKKS operations the hybrid uses, timed on each backend that loads here, with the same inputs.

A local measurement for `docs/models-and-benchmarks.md`. TenSEAL and HEonGPU get the configured CKKS parameters (`FHEConfig`: N = 8192,
moduli (60, 40, 40, 60), scale 2^40). OpenFHE gets its own (`openfhe_*`: depth 4, scale 2^50, 128 slots, the ring OpenFHE picks for 128-bit
security), which the output records, so its row is not the same computation. Every backend gets the same inputs and a check that its answer
matches plaintext before its time is kept; an input wider than its slots is skipped and says so.
GPU work is asynchronous, so a GPU backend is synchronized before the clock stops; a backend without `synchronize` is CPU-bound.

OpenFHE parallelizes with OpenMP, whose default here is one thread per hardware thread (20 on this 6 P-core, 8 E-core CPU). At
that default its 33-value dot product ranged from 81 to 577 ms as other processes came and went, because a parallel loop waits for
its slowest thread; at 6 threads it held at 44 to 54 ms (Round 86). So the thread count is set, and the output records it:

    OMP_NUM_THREADS=6 python scripts/bench_fhe_backends.py -o docs/measurements/fhe-backends-<date>.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPEATS = 20


def _timed(backend, fn) -> float:
    sync = getattr(backend, "synchronize", lambda: None)
    fn()
    sync()
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(samples), 4)


def measure(name: str) -> dict:
    from qsh.fhe.backends import get_backend

    backend = get_backend(name)
    rng = np.random.default_rng(0)
    rows = {}
    for n in (33, 256):
        if n > backend.slot_count:
            rows[f"dot_{n}_ms"] = f"not run: {n} values do not fit {backend.slot_count} slots"
            continue
        x, w = rng.normal(size=n), rng.normal(size=n)
        ct = backend.encrypt(x)
        got = backend.decrypt(backend.dot_plain(ct, w))[0]
        assert abs(got - float(w @ x)) < 1e-4, f"{name}: dot product of {n} is off by {abs(got - float(w @ x))}"
        rows[f"dot_{n}_ms"] = _timed(backend, lambda ct=ct, w=w: backend.dot_plain(ct, w))
    x, w = rng.normal(size=33), rng.normal(size=33)
    ct = backend.encrypt(x)
    rows["encrypt_ms"] = _timed(backend, lambda: backend.encrypt(x))
    rows["mul_plain_ms"] = _timed(backend, lambda: backend.mul_plain(ct, w))
    rows["add_ms"] = _timed(backend, lambda: backend.add(ct, ct))
    dot = backend.dot_plain(ct, w)
    rows["decrypt_ms"] = _timed(backend, lambda: backend.decrypt(dot))
    mask = rng.uniform(-2**30, 2**30, size=backend.slot_count)
    rows["mask_every_slot_ms"] = _timed(backend, lambda: backend.add_plain_slots(dot, mask))
    ring = backend._cc.GetRingDimension() if hasattr(backend, "_cc") else backend.slot_count * 2
    return {"backend": backend.name, "slots": backend.slot_count, "ring_dimension": int(ring), **rows}


def _module_gpu_code() -> str:
    """The GPU architectures the HEonGPU module carries code for. On any other the driver recompiles its PTX at load, and
    the first build here carried only Turing's (sm_75) for a Blackwell GPU (Round 86)."""
    import shutil

    build = Path(os.environ.get("QSH_HEONGPU_BUILD", Path(__file__).resolve().parents[1] / "native" / "heongpu_py" / "build"))
    modules = sorted(build.glob("qsh_heongpu*.so"))
    tool = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    if not modules or not Path(tool).exists():
        return "not read: no module or no cuobjdump"
    listing = subprocess.run([tool, "-all", "-lelf", str(modules[0])], capture_output=True, text=True, check=False).stdout
    return " ".join(sorted(set(re.findall(r"sm_[0-9]+[a-z]?", listing)))) or "none"


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    load_before = [round(v, 2) for v in os.getloadavg()]
    results = []
    for name in ("tenseal", "openfhe", "heongpu"):
        try:
            results.append(measure(name))
        except Exception as e:  # a backend that does not load here is reported, not hidden
            results.append({"backend": name, "not_run": f"{type(e).__name__}: {str(e)[:160]}"})
        print(results[-1], flush=True)
    # The load average before and after, because a CPU backend's time here depends on what else is running.
    load_after = [round(v, 2) for v in os.getloadavg()]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False).stdout.strip()
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=False).stdout.strip().splitlines()
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for package in ("tenseal", "openfhe"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    heongpu = subprocess.run(["git", "-C", "submodules/HEonGPU", "describe", "--tags"], capture_output=True, text=True,
                             check=False).stdout.strip()
    Path(args.output).write_text(json.dumps({
        "machine": {"gpu": gpu, "nvcc": nvcc[-1] if nvcc else "", "kernel": platform.release(), "cpu": _cpu_model(),
                    "hardware_threads": os.cpu_count(), "date": time.strftime("%Y-%m-%d"),
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset: OpenMP's default, one per hardware thread"),
                    "loadavg_1_5_15_before": load_before, "loadavg_1_5_15_after": load_after},
        "versions": {**versions, "HEonGPU": heongpu, "HEonGPU module GPU code": _module_gpu_code()}, "results": results},
        indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

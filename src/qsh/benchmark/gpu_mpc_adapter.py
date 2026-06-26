"""Glue to the optional high-performance GPU MPC-FHE path (our own integration).

This adapter drives the *upstream* OSS submodules -- HEonGPU (GPU BFV/CKKS for linear
layers, Apache-2.0) and PIGEON/``hpmpc`` (GPU secret-sharing MPC for non-linear
layers, MIT) -- through a thin integration we wrote. It is intentionally decoupled:
if the submodules are not present/built it returns ``[]`` and the benchmark falls
back to TenSEAL/OpenFHE (CPU).

We do NOT clone or vendor the mixed-authorship ``RhombusEnd2End_HEonGPU`` repo; this
is our own arrangement over the upstream projects. Build the submodules with
``scripts/build_gpu_mpc.sh`` (sets ``CMAKE_CUDA_ARCHITECTURES=120`` for Blackwell).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .metrics import BenchResult

_ROOT = Path(__file__).resolve().parents[3]
_HEONGPU = _ROOT / "submodules" / "HEonGPU"
_HPMPC = _ROOT / "submodules" / "hpmpc"


def available() -> dict:
    """Report which GPU-MPC submodules are present and built."""
    heongpu_built = (_HEONGPU / "build" / "src").exists() and any(
        (_HEONGPU / "build").glob("**/libheongpu*")) if _HEONGPU.exists() else False
    hpmpc_built = any(_HPMPC.glob("executables/run-P*")) if _HPMPC.exists() else False
    return {
        "heongpu_present": _HEONGPU.exists(),
        "heongpu_built": bool(heongpu_built),
        "hpmpc_present": _HPMPC.exists(),
        "hpmpc_built": bool(hpmpc_built),
    }


def run_gpu_mpc(workload: str = "cnn") -> list[BenchResult]:
    """Run the GPU MPC-FHE path if the submodules are built; else return [].

    Looks for a result JSON emitted by our integration harness; parses it into
    :class:`BenchResult`. Kept defensive so the benchmark never hard-fails on a box
    without the C++/CUDA build.
    """
    state = available()
    if not (state["heongpu_built"] or state["hpmpc_built"]):
        return []

    result_json = _ROOT / "results" / "gpu_mpc_result.json"
    runner = _ROOT / "scripts" / "run_gpu_mpc.sh"
    if runner.exists():
        try:
            subprocess.run(["bash", str(runner), workload], cwd=_ROOT,
                           timeout=600, check=False)
        except Exception:
            pass
    if not result_json.exists():
        return []

    from .security import compute_security_score

    sec = compute_security_score(
        security_model="computational (RLWE) + secret-sharing MPC",
        pqc_hybrid=True, kem="ml-kem-768+x25519", qber=None)
    data = json.loads(result_json.read_text())
    out = []
    for entry in data if isinstance(data, list) else [data]:
        out.append(BenchResult(
            path="MPC-FHE",
            backend=entry.get("backend", "HEonGPU+hpmpc (GPU)"),
            workload=entry.get("workload", workload),
            latency_ms=float(entry.get("latency_ms", 0.0)),
            throughput_ips=float(entry.get("throughput_ips", 0.0)),
            accuracy=entry.get("accuracy"),
            security_model="computational (RLWE) + secret-sharing MPC",
            security_score=sec["total"],
            device="gpu",
            extra={"source": "upstream HEonGPU + PIGEON/hpmpc via qsh glue",
                   "security_components": sec["components"]},
        ))
    return out

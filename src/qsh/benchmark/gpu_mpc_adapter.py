"""Which of the optional GPU submodules are present, and which this project can run.

HEonGPU (GPU CKKS, Apache-2.0) runs as :class:`qsh.fhe.backends.HeongpuBackend` since Round 86, through this project's
binding (``native/heongpu_py``, built by ``scripts/build_heongpu.sh``), and the benchmark runner adds its row itself.
Until then this module ran ``scripts/run_gpu_mpc.sh``, which timed one multiply in HEonGPU's own example benchmark and
reported it as the GPU row: none of this project's arithmetic. Both are gone.

hpmpc (PIGEON's MPC framework, MIT) is vendored as a submodule. No stage or page runs it: since Round 86
``scripts/bench_hpmpc_gpu.sh`` measures its secret-shared matrix multiplication on the CPU and the GPU, in a clone
outside the submodule (``docs/measurements/hpmpc-gpu-*.json``). ``scripts/build_gpu_mpc.sh``, which built hpmpc's
CPU default inside the submodule, discarded its errors and reported a GPU build, is gone.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_HEONGPU = _ROOT / "submodules" / "HEonGPU"
_HPMPC = _ROOT / "submodules" / "hpmpc"


def available() -> dict:
    """Whether each submodule is checked out, and whether it can run here.

    ``heongpu_built`` is the test the backend registry uses (the module imports and a GPU is present), so this cannot
    report HEonGPU as built while the selector withholds it, or the reverse.
    """
    from ..fhe.backends import heongpu_available

    hpmpc_built = any(_HPMPC.glob("executables/run-P*")) if _HPMPC.exists() else False
    return {
        "heongpu_present": _HEONGPU.exists(),
        "heongpu_built": bool(heongpu_available()),
        "hpmpc_present": _HPMPC.exists(),
        "hpmpc_built": bool(hpmpc_built),
    }

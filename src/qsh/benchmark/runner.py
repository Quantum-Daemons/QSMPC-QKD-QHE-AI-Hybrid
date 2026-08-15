"""Benchmark runner: classical MPC-FHE vs quantum QSMPC-FHE.

Produces the comparison table the SPA Benchmark page consumes (CSV + JSON). Runs the
classical CKKS path on every available backend (TenSEAL CPU, OpenFHE CPU, and the optional
HEonGPU/hpmpc GPU path via our glue) and the quantum QHE path, on a small linear /
CNN / Transformer workload.

CLI:  ``python -m qsh.benchmark.runner --workload linear --out results/``
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..monitoring.exporter import record_benchmark
from .classical_mpc_fhe import run_hybrid
from .metrics import BenchResult
from .quantum_qsmpc_fhe import run_quantum

MODEL_SIZES = ("lightweight", "medium", "large")


def _linear_model_and_data(seed: int = 0):
    from sklearn.linear_model import LogisticRegression

    from ..data.loaders import load_breast_cancer
    from ..fhe.ckks_inference import LinearModel

    X, y = load_breast_cancer()
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    w = np.vstack([np.zeros_like(clf.coef_[0]), clf.coef_[0]])
    b = np.array([0.0, float(clf.intercept_[0])])
    return LinearModel(w.astype(float), b), X.astype(float), y.astype(int)


# Model-size tiers + per-use-case architectures + the workload-class cost multipliers all live in the
# central config (config/defaults.py: BenchmarkConfig.tier_architectures / .generic_architecture /
# .workload_multipliers) — single source of truth, served to the frontend via /api/config so the demo's
# cost model never drifts from the backend. lightweight === the trained linear model (faithful SHAP);
# medium/large are representative shapes (the encrypted hybrid preserves the plaintext decision for any
# weights, so the metric is FHE-linear/MPC-non-linear COST + fidelity, not task SOTA). 'large' is projected.


def _params(layers) -> int:
    """Parameter count, spec-aware (handles legacy ``(W, b)`` lists and Conv/Dense specs)."""
    from ..mpc.layers import ConvSpec, DenseSpec, normalize_layers

    total = 0
    for s in normalize_layers(layers):
        if isinstance(s, (ConvSpec, DenseSpec)):
            total += int(np.asarray(s.weight).size) + int(np.asarray(s.bias).size)
    return total


def usecase_arch_layers(usecase, model_size: str, *, workload: str = "linear", seed: int = 0,
                        base=None, n_in: int = 30):
    """Per-use-case model layer stack + an architecture label. lightweight === the trained linear model
    (so SHAP stays faithful); medium/large use the per-industry hidden shapes from the central config,
    scaled by the workload class. Returns ``(layers, label, n_params)``."""
    from ..config import get_config

    bcfg = get_config().benchmark
    arch = bcfg.tier_architectures.get(usecase)
    if model_size == "lightweight" and base is not None:
        layers = [(np.atleast_2d(base.weight), np.asarray(base.bias))]
        return layers, (arch["label"] if arch else "linear model") + " · logistic (1 layer)", _params(layers)
    mw = bcfg.workload_multipliers.get(workload) or bcfg.workload_multipliers["linear"]
    mult, wtag = float(mw[0]), str(mw[1])
    if arch:
        hidden = arch["hidden"].get(model_size, [16])
        label = arch["label"] + f" · {model_size}{wtag}"
    else:                                   # generic (standalone benchmark, no use case)
        hidden = bcfg.generic_architecture.get(model_size, [16])
        label = f"MLP · {model_size}{wtag}"
    hidden = [max(2, int(round(h * mult))) for h in hidden]
    dims = [n_in, *hidden, 2]
    rng = np.random.default_rng(seed + 17)
    layers = [(rng.normal(size=(o, i)) * 0.1, rng.normal(size=o) * 0.05)
              for i, o in zip(dims[:-1], dims[1:])]
    return layers, label, _params(layers)


def _tier_mlp(model_size: str, seed: int = 0, base=None, n_in: int = 30):
    """Generic tiered MLP (standalone benchmark). Per-use-case shapes go through ``usecase_arch_layers``."""
    layers, _, _ = usecase_arch_layers(None, model_size, seed=seed, base=base, n_in=n_in)
    return layers


def usecase_arch_specs(usecase, model_size: str, *, workload: str = "linear", seed: int = 0,
                       base=None, n_in: int = 30, img=None):
    """Conv-aware sibling of :func:`usecase_arch_layers`.

    For the **medical CNN** case (``workload='cnn'`` + an image ``(C,H,W)``) it returns genuine
    convolution specs — ``lightweight`` = pixel logistic (a faithful 1-layer SHAP surrogate),
    ``medium`` = a real 1-block CNN, ``large`` = a 2-block CNN (projected from medium). Every other
    case delegates to :func:`usecase_arch_layers` (dense), so behaviour elsewhere is unchanged.
    Returns ``(specs_or_layers, label, n_params)``.
    """
    if usecase == "medical_fl" and workload == "cnn" and img is not None:
        from ..ai.cnn_layers import random_cnn_specs
        from ..mpc.layers import DenseSpec

        in_shape = tuple(int(v) for v in img)
        npix = int(np.prod(in_shape))
        if model_size == "lightweight":                 # pixel logistic (faithful SHAP surrogate)
            if base is not None and int(getattr(base, "n_in", -1)) == npix:
                specs = [DenseSpec(np.atleast_2d(base.weight), np.asarray(base.bias), activation="none")]
            else:
                rng = np.random.default_rng(seed + 5)
                specs = [DenseSpec(rng.normal(size=(2, npix)) * 0.05, np.zeros(2), activation="none")]
            return specs, f"pixel logistic (MedMNIST {in_shape[1]}x{in_shape[2]}) · 1 layer", _params(specs)
        channels = (4,) if model_size == "medium" else (4, 8)   # large = 2-block (projected)
        specs = random_cnn_specs(in_shape, channels=channels, hidden=16, seed=seed)
        label = f"diagnostic CNN (MedMNIST {in_shape[1]}x{in_shape[2]}, {len(channels)}-block) · {model_size}"
        return specs, label, _params(specs)
    return usecase_arch_layers(usecase, model_size, workload=workload, seed=seed, base=base, n_in=n_in)


def _macs(layers):
    """Multiply-accumulate count (linear/FHE work), spec-aware. Conv = oc·Hout·Wout·(ic·kh·kw)."""
    from ..fhe.conv import conv_cost
    from ..mpc.layers import ConvSpec, DenseSpec, normalize_layers

    total = 0
    for s in normalize_layers(layers):
        if isinstance(s, ConvSpec):
            total += conv_cost(s)["macs"]
        elif isinstance(s, DenseSpec):
            total += int(s.weight.shape[0]) * int(s.weight.shape[1])
    return total


def _cmps(layers):
    """Non-linear comparison count (MPC work), spec-aware: ReLU outputs + maxpool windows + argmax."""
    from ..fhe.conv import out_hw
    from ..mpc.layers import ConvSpec, DenseSpec, PoolSpec, normalize_layers

    total = 0
    for s in normalize_layers(layers):
        if isinstance(s, ConvSpec):
            if s.activation == "relu":
                ho, wo = out_hw(s)
                total += int(s.weight.shape[0]) * ho * wo
        elif isinstance(s, DenseSpec):
            total += int(s.weight.shape[0])             # ReLU / argmax comparisons ≈ output dim
        elif isinstance(s, PoolSpec) and s.pool == "max":
            total += int(s.size) * int(s.size) - 1      # per-window comparisons (scaled by windows below)
    return max(total, 1)


def _project_large(med: BenchResult, med_layers, large_layers, workload: str) -> BenchResult:
    """Extrapolate the 'large' tier from the measured 'medium' per-op cost (clearly flagged)."""
    fr = _macs(large_layers) / max(_macs(med_layers), 1)
    cr = _cmps(large_layers) / max(_cmps(med_layers), 1)
    fhe = (med.fhe_linear_ms or 0) * fr
    mpc = (med.mpc_nonlinear_ms or 0) * cr
    comm = (med.mpc_comm_kb or 0) * cr
    return BenchResult(
        path="MPC-FHE", backend=f"{med.backend} (projected)", workload=workload, model_size="large",
        latency_ms=round(fhe + mpc, 3), throughput_ips=round(1000 / max(fhe + mpc, 1e-3), 2),
        accuracy=med.accuracy, ciphertext_plaintext_ratio=med.ciphertext_plaintext_ratio,
        security_model=med.security_model, security_score=med.security_score,
        fhe_linear_ms=round(fhe, 3), mpc_nonlinear_ms=round(mpc, 3), mpc_comm_kb=round(comm, 3),
        projected=True, device=med.device,
        extra={"projected_from": "medium", "fhe_scale": round(fr, 2), "mpc_scale": round(cr, 2),
               "linear_under": "FHE/CKKS", "nonlinear_under": "MPC/mpyc-shamir"},
    )


def run_benchmark(workload: str = "linear", *, out_dir: str | Path = "results",
                  n_samples: int = 30, seed: int = 0, include_openfhe: bool = True,
                  model_size: str = "medium") -> list[BenchResult]:
    """Run the full classical-vs-quantum benchmark and persist artifacts.

    The classical **MPC-FHE** path is the faithful 2026 hybrid: linear layers under CKKS (FHE,
    GPU-capable) + non-linear (ReLU/argmax) under MPyC secret-sharing (MPC). ``model_size`` selects
    a lightweight/medium/large tier; lightweight+medium run for real, 'large' is projected.
    """
    from ..config import get_config

    bcfg = get_config().benchmark
    results: list[BenchResult] = []
    model, X, y = _linear_model_and_data(seed)
    n_hybrid = min(n_samples, bcfg.hybrid_samples)   # MPC ReLU/argmax is per-sample; keep run responsive

    if model_size == "large":
        # project the 'large' tier from a quick 'medium' calibration (keeps the page responsive)
        med_layers = _tier_mlp("medium", seed, base=model)
        med = run_hybrid(med_layers, X, y, backend_name="tenseal", workload=workload,
                         model_size="medium", n_samples=min(n_hybrid, 4))
        results.append(_project_large(med, med_layers, _tier_mlp("large", seed, base=model), workload))
    else:
        layers = _tier_mlp(model_size, seed, base=model)
        # hybrid on TenSEAL (Apache-2.0, CPU FHE) + MPyC (MIT, CPU MPC non-linear)
        results.append(run_hybrid(layers, X, y, backend_name="tenseal", workload=workload,
                                  model_size=model_size, n_samples=n_hybrid))
        # OpenFHE variant (BSD-2; live in the Docker image / Ubuntu 24.04, skipped otherwise)
        if include_openfhe:
            try:
                results.append(run_hybrid(layers, X, y, backend_name="openfhe", workload=workload,
                                          model_size=model_size, n_samples=n_hybrid))
            except Exception as e:  # pragma: no cover
                print(f"[bench] OpenFHE backend unavailable: {e}")

    # optional high-performance GPU MPC (HEonGPU + hpmpc) via our glue
    try:
        from .gpu_mpc_adapter import run_gpu_mpc

        gpu = run_gpu_mpc(workload=workload)
        if gpu:
            results.extend(gpu)
    except Exception as e:  # pragma: no cover
        print(f"[bench] GPU-MPC adapter skipped: {e}")

    # quantum QSMPC-FHE path
    q = run_quantum(X, y, n_qubits=bcfg.quantum_qubits,
                    n_samples=min(n_samples, bcfg.quantum_samples), seed=seed)
    q.model_size = model_size
    results.append(q)

    # BENCH_LATENCY was declared with a "path" label and had no reader anywhere - not even inside
    # the exporter that defined it. These are its rows.
    for r in results:
        record_benchmark(r.path, float(r.latency_ms))

    _persist(results, out_dir, workload)
    return results


def compute_advantage(results: list[BenchResult]) -> dict | None:
    """Quantum-advantage of the QSMPC-FHE path vs the classical CPU MPC-FHE baseline."""
    from .quantum_advantage import quantum_advantage

    classical = next((r for r in results if r.path == "MPC-FHE" and r.device == "cpu"), None)
    quantum = next((r for r in results if r.path == "QSMPC-FHE"), None)
    if classical and quantum:
        return quantum_advantage(classical, quantum)
    return None


def _persist(results: list[BenchResult], out_dir: str | Path, workload: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r.to_row() for r in results]
    (out / f"benchmark_{workload}.json").write_text(json.dumps(rows, indent=2, default=str))
    if rows:
        with (out / f"benchmark_{workload}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, dict) else v)
                            for k, v in r.items()})


def main(argv: list[str] | None = None) -> int:
    import argparse

    from rich.console import Console
    from rich.table import Table

    p = argparse.ArgumentParser(description="MPC-FHE vs QSMPC-FHE benchmark")
    p.add_argument("--workload", default="linear", help="linear | cnn | transformer")
    p.add_argument("--model-size", default="medium", choices=MODEL_SIZES)
    p.add_argument("--out", default="results/")
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--no-openfhe", action="store_true")
    args = p.parse_args(argv)

    results = run_benchmark(args.workload, out_dir=args.out, n_samples=args.samples,
                            include_openfhe=not args.no_openfhe, model_size=args.model_size)
    console = Console()
    table = Table(title=f"MPC-FHE vs QSMPC-FHE ({args.workload} / {args.model_size})")
    for col in ("path", "backend", "size", "latency_ms", "fhe_lin_ms", "mpc_nl_ms",
                "accuracy", "security", "device"):
        table.add_column(col)
    for r in results:
        table.add_row(r.path, r.backend, str(r.model_size), f"{r.latency_ms}",
                      str(r.fhe_linear_ms), str(r.mpc_nonlinear_ms),
                      str(r.accuracy), r.security_model, r.device + (" *proj" if r.projected else ""))
    console.print(table)
    console.print(f"[green]artifacts written to {args.out}[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

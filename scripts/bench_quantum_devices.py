#!/usr/bin/env python3
"""Where a GPU state-vector simulator overtakes the CPU, on this machine, for the circuits this project runs.

A local measurement for `docs/models-and-benchmarks.md`, not something the demo does. The circuit is four layers of RY on
every wire and a CNOT chain, and the result is <Z_0>. Reading that float forces the simulation to finish, so a GPU backend
cannot be timed before its work is done. Every backend runs the same angles, so their values must agree: this prints the
largest disagreement, and a backend that disagrees has measured something else.

The three families do not share an environment (different CUDA runtimes), so each is run in its own and the outputs merged:

    python scripts/bench_quantum_devices.py --family pennylane -o pl.json
    python scripts/bench_quantum_devices.py --family cudaq -o cq.json
    python scripts/bench_quantum_devices.py --family aer -o aer.json
    python scripts/bench_quantum_devices.py --merge pl.json cq.json aer.json

CUDA-Q's `qpp-cpu` is not timed: on this 14-core hybrid CPU it took anywhere from 1 ms to 1 s at the same small register, depending on
OpenMP's thread count and wait policy and on what else was running, and nothing here uses it.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

LAYERS = 4
REPEATS = 7
#: The largest register each backend is timed at. CPU simulators stop where a run takes minutes, not where they fail.
LIMITS = {"default.qubit": 20, "lightning.qubit": 26, "lightning.gpu": 30, "nvidia": 30, "nvidia-fp64": 30, "aer": 30}
QUBITS = (4, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30)


def _angles(n: int) -> np.ndarray:
    return np.random.default_rng(n).uniform(0, 3, (LAYERS, n))


def _time(fn) -> dict:
    """Median of the runs after the first, kept apart: a first call compiles, loads and allocates."""
    samples, value = [], None
    for _ in range(REPEATS + 1):
        t0 = time.perf_counter()
        value = float(fn())
        samples.append((time.perf_counter() - t0) * 1e3)
    return {"first_ms": round(samples[0], 3), "median_ms": round(statistics.median(samples[1:]), 3),
            "min_ms": round(min(samples[1:]), 3), "max_ms": round(max(samples[1:]), 3), "value": value}


def run_pennylane() -> list[dict]:
    import pennylane as qml

    rows = []
    for name in ("default.qubit", "lightning.qubit", "lightning.gpu"):
        for n in QUBITS:
            if n > LIMITS[name]:
                continue
            theta = _angles(n)

            def circuit(theta=theta, n=n):
                for layer in range(LAYERS):
                    for i in range(n):
                        qml.RY(theta[layer, i], wires=i)
                    for i in range(n - 1):
                        qml.CNOT(wires=[i, i + 1])
                return qml.expval(qml.PauliZ(0))

            qnode = qml.QNode(circuit, qml.device(name, wires=n), diff_method=None)
            rows.append({"backend": f"PennyLane {name}", "qubits": n, **_time(qnode)})
            print(rows[-1], flush=True)
    return rows


def run_cudaq() -> list[dict]:
    import cudaq

    @cudaq.kernel
    def kernel(n: int, theta: list[float]):
        q = cudaq.qvector(n)
        for layer in range(4):
            for i in range(n):
                ry(theta[layer * n + i], q[i])  # noqa: F821 - a CUDA-Q kernel intrinsic
            for i in range(n - 1):
                x.ctrl(q[i], q[i + 1])  # noqa: F821

    z0 = cudaq.spin.z(0)
    rows = []
    for name in ("nvidia", "nvidia-fp64"):
        cudaq.set_target(name)
        for n in QUBITS:
            if n > LIMITS[name]:
                continue
            theta = _angles(n).ravel().tolist()
            rows.append({"backend": f"CUDA-Q {name}", "qubits": n,
                         **_time(lambda n=n, theta=theta: cudaq.observe(kernel, z0, n, theta).expectation())})
            print(rows[-1], flush=True)
    return rows


def run_aer() -> list[dict]:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Pauli
    from qiskit_aer import AerSimulator

    sim = AerSimulator(method="statevector", device="GPU", precision="double")
    rows = []
    for n in QUBITS:
        if n > LIMITS["aer"]:
            continue
        theta = _angles(n)
        qc = QuantumCircuit(n)
        for layer in range(LAYERS):
            for i in range(n):
                qc.ry(float(theta[layer, i]), i)
            for i in range(n - 1):
                qc.cx(i, i + 1)
        qc.save_expectation_value(Pauli("I" * (n - 1) + "Z"), range(n))
        compiled = transpile(qc, sim, optimization_level=0)
        rows.append({"backend": "Qiskit Aer GPU", "qubits": n,
                     **_time(lambda c=compiled: sim.run(c).result().data()["expectation_value"])})
        print(rows[-1], flush=True)
    return rows


def machine() -> dict:
    query = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, check=False).stdout.strip()
    return {"gpu": query, "kernel": platform.release(), "python": platform.python_version(),
            "date": time.strftime("%Y-%m-%d")}


def merge(paths: list[str]) -> int:
    rows, versions = [], []
    for path in paths:
        data = json.loads(Path(path).read_text())
        rows += data["rows"]
        versions.append(data["versions"])  # per environment: two of them can hold different builds of one package
    by_qubits: dict[int, list[dict]] = {}
    for row in rows:
        by_qubits.setdefault(row["qubits"], []).append(row)
    backends = sorted({r["backend"] for r in rows})
    print("| qubits | " + " | ".join(backends) + " | largest disagreement |")
    print("|---|" + "---|" * (len(backends) + 1))
    for n in sorted(by_qubits):
        cells = {r["backend"]: r for r in by_qubits[n]}
        values = [r["value"] for r in by_qubits[n]]
        spread = max(values) - min(values)
        line = " | ".join(f'{cells[b]["median_ms"]:.2f}' if b in cells else "-" for b in backends)
        print(f"| {n} | {line} | {spread:.1e} |")
    print("\n", json.dumps(versions))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=["pennylane", "cudaq", "aer"])
    ap.add_argument("-o", "--output")
    ap.add_argument("--merge", nargs="+")
    args = ap.parse_args()
    if args.merge:
        return merge(args.merge)
    if not args.family or not args.output:
        ap.error("--family and -o, or --merge")
    rows = {"pennylane": run_pennylane, "cudaq": run_cudaq, "aer": run_aer}[args.family]()
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for package in ("pennylane", "pennylane-lightning", "pennylane-lightning-gpu", "custatevec-cu12", "cuda-quantum-cu13",
                    "custatevec-cu13", "qiskit", "qiskit-aer-gpu"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    Path(args.output).write_text(json.dumps({"machine": machine(), "versions": versions, "rows": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

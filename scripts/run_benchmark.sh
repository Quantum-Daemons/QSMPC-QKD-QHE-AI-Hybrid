#!/usr/bin/env bash
# Run the MPC-FHE vs QSMPC-FHE benchmark and write CSV/JSON to results/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
WORKLOAD="${1:-linear}"
echo "==> Benchmark workload=$WORKLOAD"
"$PY" -m qsh.benchmark.runner --workload "$WORKLOAD" --out results/ "${@:2}"
echo "==> Artifacts:"
ls -1 results/benchmark_"$WORKLOAD".* 2>/dev/null || true

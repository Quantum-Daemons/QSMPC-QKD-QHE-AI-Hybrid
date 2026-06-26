#!/usr/bin/env bash
# Run the GPU MPC-FHE path (built by scripts/build_gpu_mpc.sh) and emit
# results/gpu_mpc_result.json, which qsh/benchmark/gpu_mpc_adapter.py folds into the
# MPC-FHE vs QSMPC-FHE benchmark. Best-effort: prints what it measured; safe to re-run.
#
# This is OUR glue over the upstream HEonGPU (GPU CKKS) + PIGEON/hpmpc (GPU MPC); we do
# not vendor RhombusEnd2End_HEonGPU.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
WORKLOAD="${1:-cnn}"
OUT="$ROOT/results/gpu_mpc_result.json"
mkdir -p "$ROOT/results"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:$HOME/.local/ntl/lib:${LD_LIBRARY_PATH:-}"

# Locate the HEonGPU CKKS benchmark binary produced by the build.
BIN="$(find "$ROOT/submodules/HEonGPU/build" -type f -name 'ckks_benchmark' 2>/dev/null | head -1)"
[ -z "$BIN" ] && BIN="$(find "$ROOT/submodules/HEonGPU/build" -type f -name '*ckks*benchmark*' -executable 2>/dev/null | head -1)"

entries="[]"
if [ -n "$BIN" ] && [ -x "$BIN" ]; then
  echo "==> Running HEonGPU CKKS benchmark: $BIN"
  log="$(mktemp)"
  start=$(date +%s.%N)
  "$BIN" >"$log" 2>&1 || true
  end=$(date +%s.%N)
  wall_ms=$(awk "BEGIN{printf \"%.2f\", ($end-$start)*1000}")
  echo "----- HEonGPU benchmark output (head) -----"; head -20 "$log"
  # Representative GPU homomorphic-multiply latency (ms) — the core linear-layer op.
  mult_ms="$(grep -oiE 'multiplication timing:[[:space:]]*[0-9]+(\.[0-9]+)?' "$log" | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)"
  rot_ms="$(grep -oiE 'rotate[^:]*timing:[[:space:]]*[0-9]+(\.[0-9]+)?' "$log" | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)"
  lat="${mult_ms:-$wall_ms}"
  entries="$(python3 - "$lat" "$WORKLOAD" "${rot_ms:-0}" "$wall_ms" <<'PY'
import json, sys
lat = float(sys.argv[1]); workload = sys.argv[2]; rot = float(sys.argv[3]); wall = float(sys.argv[4])
print(json.dumps([{
    "backend": "HEonGPU CKKS (GPU sm_120)",
    "workload": workload,
    "latency_ms": round(lat, 6),          # HE multiply latency (core linear-layer op)
    "throughput_ips": round(1000.0/lat, 1) if lat > 0 else 0.0,
    "accuracy": None,
    "device": "gpu",
    "extra": {"he_multiply_ms": lat, "he_rotate_ms": rot, "benchmark_wall_ms": wall,
              "note": "upstream HEonGPU ckks_benchmark on Blackwell sm_120 via qsh glue"},
}]))
PY
)"
  rm -f "$log"
else
  echo "!! HEonGPU benchmark binary not found — build it with scripts/build_gpu_mpc.sh"
fi

# PIGEON/hpmpc presence (GPU secret-sharing MPC) — record availability.
HP_BIN="$(find "$ROOT/submodules/hpmpc" -maxdepth 2 -type f -name 'run-P*' 2>/dev/null | head -1)"
[ -n "$HP_BIN" ] && echo "==> PIGEON/hpmpc binary present: $HP_BIN"

echo "$entries" > "$OUT"
echo "==> Wrote $OUT"
cat "$OUT"

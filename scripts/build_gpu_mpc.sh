#!/usr/bin/env bash
# Builds hpmpc / PIGEON (GPU secret-sharing MPC, MIT), the optional GPU submodule nothing here runs yet.
#
# HEonGPU is not built here since Round 86: scripts/build_heongpu.sh builds it with this project's binding, which is what
# the GPU CKKS backend imports. The section this replaced built HEonGPU's own examples, which only the removed
# scripts/run_gpu_mpc.sh ran.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

ARCH="${CUDA_ARCH:-120}"                      # Blackwell sm_120
JOBS="$(nproc)"

echo "==> Ensuring submodules are present (full set)"
QSH_SET=all bash scripts/bootstrap_submodules.sh || true

# --------------------------------------------------------------------------- #
#  hpmpc / PIGEON (GPU secret-sharing MPC)
# --------------------------------------------------------------------------- #
HP="$ROOT/submodules/hpmpc"
if [ -d "$HP" ]; then
  echo "==> Building hpmpc / PIGEON (best-effort; CUTLASS arch=sm_$ARCH)"
  make -C "$HP" -j"$JOBS" CUDA_ARCH="sm_$ARCH" 2>/dev/null \
    && echo "==> hpmpc build OK" \
    || echo "!! hpmpc build skipped/failed (needs CUTLASS + arch config; non-blocking)."
fi

echo "==> Done. The GPU CKKS backend is built separately: bash scripts/build_heongpu.sh"

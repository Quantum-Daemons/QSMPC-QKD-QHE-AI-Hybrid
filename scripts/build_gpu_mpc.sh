#!/usr/bin/env bash
# Build the optional high-performance GPU MPC-FHE submodules for the benchmark.
#   - HEonGPU  (GPU BFV/CKKS, Apache-2.0) -- GPU-accelerated linear-layer HE
#   - hpmpc / PIGEON (GPU secret-sharing MPC, MIT) -- non-linear-layer optimization
#
# Targets NVIDIA Blackwell (sm_120) + CUDA 13. OPTIONAL: the benchmark runs on
# TenSEAL/DESILO without these. We integrate the UPSTREAM projects with our own glue
# (qsh/benchmark/gpu_mpc_adapter.py); we do not vendor RhombusEnd2End_HEonGPU.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

ARCH="${CUDA_ARCH:-120}"                      # Blackwell sm_120
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
NTL_INC="${NTL_INC:-$HOME/.local/ntl/include}"
JOBS="$(nproc)"

echo "==> Ensuring submodules are present (full set)"
QSH_SET=all bash scripts/bootstrap_submodules.sh || true

# --------------------------------------------------------------------------- #
#  HEonGPU
# --------------------------------------------------------------------------- #
HE="$ROOT/submodules/HEonGPU"
if [ -d "$HE" ]; then
  echo "==> Patching HEonGPU RMM tag for CUDA 13 / CCCL 3.0 (branch-25.08 -> branch-25.10)"
  # RMM <25.10 is incompatible with CCCL 3.0 (CUDA 13's bundled CCCL). Idempotent.
  sed -i 's/branch-25\.08/branch-25.10/g' "$HE/thirdparty/CMakeLists.txt" 2>/dev/null || true

  NTL_FLAG=""
  [ -d "$NTL_INC" ] && NTL_FLAG="-I$NTL_INC"
  # CUDA 13 relocates Thrust/CUB/libcudacxx under <cuda>/include/cccl/. HEonGPU's
  # FindThrust doesn't know that path, so point it there explicitly.
  CCCL_INC=""
  for c in "$CUDA_HOME/targets/x86_64-linux/include/cccl" "$CUDA_HOME/include/cccl"; do
    [ -f "$c/thrust/version.h" ] && CCCL_INC="$c" && break
  done
  THRUST_FLAG=""; [ -n "$CCCL_INC" ] && THRUST_FLAG="-DThrust_DIR=$CCCL_INC -DTHRUST_INCLUDE_DIR=$CCCL_INC"
  GEN=""; command -v ninja >/dev/null 2>&1 && GEN="-GNinja"
  echo "==> Configuring HEonGPU (arch=$ARCH, nvcc=$NVCC, ntl=${NTL_INC}, cccl=${CCCL_INC:-?}, gen=${GEN:-Make})"
  cmake -S "$HE" -B "$HE/build" $GEN \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER="$NVCC" \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
        -DHEONGPU_CUDA_ARCH_FORCE_MANUAL=ON \
        -DHEonGPU_BUILD_BENCHMARKS=ON -DHEonGPU_BUILD_EXAMPLES=ON \
        $THRUST_FLAG \
        -DCMAKE_CXX_FLAGS="$NTL_FLAG ${CCCL_INC:+-I$CCCL_INC}" \
        -DCMAKE_CUDA_FLAGS="$NTL_FLAG ${CCCL_INC:+-I$CCCL_INC}" \
        -DCMAKE_EXE_LINKER_FLAGS="${NTL_INC:+-L$(dirname "$NTL_INC")/lib}" \
    && cmake --build "$HE/build" -j"$JOBS" \
    && cmake --build "$HE/build" --target ckks_benchmark bfv_benchmark -j"$JOBS" \
    && echo "==> HEonGPU build OK" \
    || echo "!! HEonGPU build FAILED (see log). Benchmark falls back to DESILO/TenSEAL."
fi

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

echo "==> Done. Run the GPU MPC-FHE benchmark step: bash scripts/run_gpu_mpc.sh cnn"

#!/usr/bin/env bash
# Builds the GPU CKKS backend: HEonGPU (the pinned submodule, v2.1.1) and this project's pybind11 module over it,
# native/heongpu_py, into native/heongpu_py/build. qsh.fhe.backends.HeongpuBackend imports it from there.
#
# Why (Round 86, the owner's choice of the whole GPU path): HEonGPU has no Python API, so the backend was a stub that raised.
# The module exposes the seven operations the backend protocol needs and nothing else.
#
# Needs an NVIDIA GPU, CUDA 12.8 or later (HEonGPU's CMake refuses older), CMake 4, Ninja or Make, GMP's development headers,
# and pybind11 in the Python the tests use:
#
#   pip install "cmake>=4" ninja pybind11        # into the Python that will import the module
#   bash scripts/build_heongpu.sh
#
# NTL (LGPL-2.1-or-later) is built here as position-independent code, because the module is a shared object and NTL's
# default static build is not. It links into the module, which this project does not distribute; DATA_LICENSES.md says so.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${WORK:-$REPO/.cache/heongpu}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
CMAKE="${CMAKE:-cmake}"
NTL_VERSION=11.5.1
NTL_SHA256=210d06c31306cbc6eaf6814453c56c776d9d8e8df36d74eb306f6a523d1c6a8a
HE="$REPO/submodules/HEonGPU"
OUT="$REPO/native/heongpu_py/build"

major=$("$CMAKE" --version | head -1 | sed -E 's/[^0-9]*([0-9]+)\..*/\1/')
[ "$major" -ge 4 ] || { echo "HEonGPU needs CMake 4 or later; $CMAKE is $("$CMAKE" --version | head -1). pip install 'cmake>=4' and set CMAKE." >&2; exit 2; }
"$PYTHON" -m pybind11 --cmakedir >/dev/null || { echo "pybind11 is not installed in $PYTHON" >&2; exit 2; }
command -v nvcc >/dev/null || [ -x /usr/local/cuda/bin/nvcc ] || { echo "no nvcc: HEonGPU needs the CUDA toolkit" >&2; exit 2; }
NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
# The GPU's own architecture (12.0 -> 120), or CUDA_ARCH. Left to CMake, the wrapper project took nvcc's default of 75 and the
# module ran as Turing code the driver recompiled at load; the check after the build would have caught that.
ARCH="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)}"
[ -n "$ARCH" ] || { echo "no GPU architecture: set CUDA_ARCH, for example 120" >&2; exit 2; }
CUOBJDUMP="$(dirname "$NVCC")/cuobjdump"

echo "==> HEonGPU and its submodules (GPU-FFT, GPU-NTT, RNGonGPU)"
git -C "$REPO" submodule update --init submodules/HEonGPU
git -C "$HE" submodule update --init --recursive
echo "    $(git -C "$HE" describe --tags 2>/dev/null || git -C "$HE" rev-parse --short HEAD)"

mkdir -p "$WORK"
if [ ! -f "$WORK/ntl/lib/libntl.a" ]; then
  echo "==> NTL $NTL_VERSION, position-independent"
  curl -sSL -o "$WORK/ntl.tar.gz" "https://libntl.org/ntl-$NTL_VERSION.tar.gz"
  echo "$NTL_SHA256  $WORK/ntl.tar.gz" | sha256sum -c - >/dev/null
  rm -rf "$WORK/ntl-src" && mkdir -p "$WORK/ntl-src" && tar -xzf "$WORK/ntl.tar.gz" -C "$WORK/ntl-src" --strip-components=1
  (cd "$WORK/ntl-src/src" && ./configure PREFIX="$WORK/ntl" NTL_GMP_LIP=on SHARED=off NATIVE=off CXXFLAGS="-O2 -fPIC" \
     >"$WORK/ntl-configure.log" 2>&1 && make -j"$(nproc)" >"$WORK/ntl-make.log" 2>&1 && make install >"$WORK/ntl-install.log" 2>&1)
fi

echo "==> the module (HEonGPU builds inside it)"
GEN=""; command -v ninja >/dev/null 2>&1 && GEN="-GNinja"
"$CMAKE" -S "$REPO/native/heongpu_py" -B "$OUT" $GEN -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DHEONGPU_SRC="$HE" -DNTL_ROOT="$WORK/ntl" -DPython_EXECUTABLE="$PYTHON" \
  -Dpybind11_DIR="$("$PYTHON" -m pybind11 --cmakedir)" >"$WORK/module-configure.log" 2>&1
"$CMAKE" --build "$OUT" -j"$(nproc)" >"$WORK/module-build.log" 2>&1
MODULE="$(ls "$OUT"/qsh_heongpu*.so)"
echo "    $MODULE"
built="$("$CUOBJDUMP" -all -lelf "$MODULE" | grep -oE 'sm_[0-9]+[a-z]?' | sort -u | tr '\n' ' ')"
case " $built " in
  *" sm_$ARCH "*) echo "    GPU code: $built" ;;
  *) echo "the module carries GPU code for: ${built:-nothing}, not sm_$ARCH. A stale $OUT/CMakeCache.txt keeps the old value:" \
       "delete $OUT and build again." >&2; exit 1 ;;
esac
echo "==> Done. get_backend('heongpu') now loads it; tests/test_r86_heongpu_backend.py runs against it."

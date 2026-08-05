#!/usr/bin/env bash
# Build OpenFHE + its Python bindings from source on RHEL 9 / AlmaLinux 9.
#
# WHY THIS EXISTS
#   The published `openfhe` wheels come in two flavours and neither loads on EL9 + Python 3.12:
#     * openfhe==X.Y.Z.0.24.4  (Ubuntu 24.04 build, cpython-312) needs GLIBCXX_3.4.32
#     * openfhe==X.Y.Z.0.22.4  (Ubuntu 22.04 build) ships a cpython-310 .so
#   EL9's system libstdc++.so.6 tops out at GLIBCXX_3.4.29, so the 24.04 wheel fails with
#     ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.32' not found
#   Building here with gcc-toolset-14 links the newer C++ bits statically (libstdc++_nonshared.a),
#   so the resulting extension needs only GLIBCXX_3.4.29 and loads against the stock system library.
#
#   Without this, `qsh.fhe.backends.OpenFHEBackend` is unavailable on the dev host and the FHE layer
#   would quietly run every "openfhe-cpu" selection on TenSEAL instead. The project's no-silent-
#   fallback policy makes that unacceptable, so we fix the host rather than hide the failure.
#   (The Docker images are unaffected: python:3.12-slim-trixie already ships GLIBCXX_3.4.32+.)
#
# PREREQUISITES
#   sudo dnf install -y gcc-toolset-14 gcc-toolset-14-gcc-c++ cmake make git
#
# USAGE
#   bash env/build_openfhe_el9.sh                 # build + install into ./.venv
#   OPENFHE_VERSION=1.5.1 bash env/build_openfhe_el9.sh
#
set -euo pipefail

OPENFHE_VERSION="${OPENFHE_VERSION:-1.5.1}"
# openfhe-python tags carry a fourth component; keep the two in lockstep or its
# find_package(OpenFHE <ver> REQUIRED) will reject the C++ build.
OPENFHE_PY_TAG="${OPENFHE_PY_TAG:-v${OPENFHE_VERSION}.0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv}"
WORK="${WORK:-${TMPDIR:-/tmp}/openfhe-build-$OPENFHE_VERSION}"
PREFIX="$WORK/prefix"
JOBS="${JOBS:-$(nproc)}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "error: no venv at $VENV (run env/setup_venv.sh first)" >&2
  exit 1
fi

TOOLSET=/opt/rh/gcc-toolset-14/enable
if [ ! -f "$TOOLSET" ]; then
  echo "error: gcc-toolset-14 not found. Install it with:" >&2
  echo "  sudo dnf install -y gcc-toolset-14 gcc-toolset-14-gcc-c++" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$TOOLSET"
echo "compiler: $(gcc --version | head -1)"

mkdir -p "$PREFIX"
cd "$WORK"

# --- OpenFHE C++ ------------------------------------------------------------
if [ ! -d openfhe-development ]; then
  git clone --depth 1 --branch "v$OPENFHE_VERSION" \
    https://github.com/openfheorg/openfhe-development.git
fi
cmake -S openfhe-development -B openfhe-development/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
  -DBUILD_SHARED=ON
cmake --build openfhe-development/build -j "$JOBS"
cmake --install openfhe-development/build

# --- Python bindings --------------------------------------------------------
if [ ! -d openfhe-python ]; then
  git clone --depth 1 --branch "$OPENFHE_PY_TAG" \
    https://github.com/openfheorg/openfhe-python.git
fi
cmake -S openfhe-python -B openfhe-python/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenFHE_DIR="$PREFIX/lib/OpenFHE" \
  -DPython3_EXECUTABLE="$VENV/bin/python" \
  -DCMAKE_PREFIX_PATH="$PREFIX"
cmake --build openfhe-python/build -j "$JOBS"

# --- Install over the wheel -------------------------------------------------
# The wheel supplies openfhe/__init__.py and the openfhe/lib layout; we replace the
# extension and the bundled shared libraries with the ones just built. The extension's
# RUNPATH already starts with $ORIGIN/lib, so the package stays self-contained.
SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [ ! -d "$SITE/openfhe" ]; then
  echo "error: openfhe wheel not installed. Run: $VENV/bin/pip install -r requirements-fhe.txt" >&2
  exit 1
fi
EXT="$(find openfhe-python/build -maxdepth 1 -name 'openfhe.cpython-*.so' | head -1)"
[ -n "$EXT" ] || { echo "error: built extension not found" >&2; exit 1; }
install -m 0755 "$EXT" "$SITE/openfhe/"
mkdir -p "$SITE/openfhe/lib"
cp -P "$PREFIX"/lib/libOPENFHE*.so.1* "$SITE/openfhe/lib/"

echo "=== verifying real CKKS arithmetic ==="
"$VENV/bin/python" - <<'PY'
import openfhe
p = openfhe.CCParamsCKKSRNS()
p.SetMultiplicativeDepth(2); p.SetScalingModSize(50); p.SetBatchSize(8)
cc = openfhe.GenCryptoContext(p)
cc.Enable(openfhe.PKESchemeFeature.PKE)
cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)
kp = cc.KeyGen(); cc.EvalMultKeyGen(kp.secretKey)
ct = cc.Encrypt(kp.publicKey, cc.MakeCKKSPackedPlaintext([1.5, 2.5, 3.5, 4.0]))
res = cc.Decrypt(kp.secretKey, cc.EvalMult(ct, ct)); res.SetLength(4)
got = [round(v, 4) for v in res.GetRealPackedValue()]
want = [2.25, 6.25, 12.25, 16.0]
assert all(abs(a - b) < 1e-3 for a, b in zip(got, want)), f"{got} != {want}"
print("OpenFHE CKKS square:", got, "OK")
PY
echo "=== done. Build tree left at $WORK (safe to delete) ==="

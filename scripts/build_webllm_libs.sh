#!/usr/bin/env bash
# Builds the WebLLM model libraries the demo serves from web/public/webllm/, from source, with their
# manifest and third-party notices:
#
#   web/public/webllm/<version>/<model>.wasm      one library per model in web.webllm_models
#   web/public/webllm/<version>/MANIFEST.json     what each was built from, with its size and hash
#   web/public/licenses/webllm-libs.txt           the licences and notices of what is linked into them
#
# Why from source (Round 86): @mlc-ai/web-llm fetches its prebuilt libraries from the repository
# mlc-ai/binary-mlc-llm-libs, which has no licence file, while MLC-LLM and TVM are Apache-2.0.
#
# The pins are upstream's own. The commit adding v0_2_87's libraries says "Compiled at TVM apache/tvm@e345029,
# MLC-LLM mlc-ai/mlc-llm@44fd06d"; that MLC-LLM commit's tvm submodule is mlc-ai/relax@629cc658, e345029 plus CI
# commits. Emscripten is the version TVM's CI installs (docker/install/ubuntu_install_emscripten.sh). MLC-LLM's
# docs still suggest 3.1.56, whose LLVM 19 cannot read the bitcode an LLVM 21 TVM writes.
#
# Needs git, cmake >= 3.24, cargo, npm, python3.12 and llvm-config (LLVM >= 15 with the WebAssembly target), about
# 15 GB and half an hour. Installs nothing outside the work directory. Run from anywhere:
#
#   bash scripts/build_webllm_libs.sh [work-dir]
#
# Then copy the printed `lib_sri` values to web.webllm_models in src/qsh/config/defaults.py and run
# scripts/export_config.py. tests/test_r86_browser_runtime_stays_same_origin.py holds them to the files.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-$REPO/.cache/webllm-build}"
MLC_LLM_COMMIT=44fd06d04f22c87820e205b0f2b2e5e6a1f6ca85
EMSDK_VERSION=4.0.23
PREFILL_CHUNK_SIZE=1024
PYTHON="${PYTHON:-python3.12}"
JOBS="${JOBS:-8}"

mkdir -p "$WORK"
cd "$WORK"

echo "==> Emscripten $EMSDK_VERSION"
[ -d emsdk ] || git clone --depth 1 https://github.com/emscripten-core/emsdk.git emsdk
export EMSDK_PYTHON
EMSDK_PYTHON="$(command -v "$PYTHON")"
(cd emsdk && ./emsdk install "$EMSDK_VERSION" && ./emsdk activate "$EMSDK_VERSION") >emsdk.log 2>&1
# shellcheck disable=SC1091
EMSDK_QUIET=1 . emsdk/emsdk_env.sh >/dev/null 2>&1
export EMSDK_PYTHON
EMSDK_PYTHON="$(command -v "$PYTHON")"

echo "==> MLC-LLM $MLC_LLM_COMMIT and its submodules"
[ -d mlc-llm ] || git clone --filter=blob:none https://github.com/mlc-ai/mlc-llm.git mlc-llm
git -C mlc-llm checkout -q "$MLC_LLM_COMMIT"
git -C mlc-llm submodule update --init --recursive --filter=blob:none
TVM="$WORK/mlc-llm/3rdparty/tvm"

echo "==> Python environment"
[ -d venv ] || "$PYTHON" -m venv venv
export PATH="$WORK/venv/bin:$PATH"
pip install -q -U pip
pip install -q "$TVM/3rdparty/tvm-tirx/3rdparty/tvm-ffi"
pip install -q fastapi shortuuid safetensors tiktoken transformers requests prompt_toolkit openai uvicorn tqdm pandas \
  sentencepiece psutil ml_dtypes
pip install -q torch --index-url https://download.pytorch.org/whl/cpu

echo "==> TVM compiler"
if [ ! -f "$TVM/build/lib/libtvm_compiler.so" ]; then
  # CMake finds LLVM through llvm-config by itself; this TVM has no config.cmake to edit.
  mkdir -p "$TVM/build"
  (cd "$TVM/build" && cmake .. -DCMAKE_BUILD_TYPE=Release >cmake.log 2>&1 && make -j "$JOBS" >make.log 2>&1)
fi
mkdir -p "$TVM/3rdparty/tvm-tirx/build"
ln -sfn "$TVM/build/lib" "$TVM/3rdparty/tvm-tirx/build/lib"

echo "==> MLC-LLM native library"
if [ ! -f mlc-llm/build/libmlc_llm.so ]; then
  mkdir -p mlc-llm/build
  cat >mlc-llm/build/config.cmake <<'CMAKE'
set(TVM_SOURCE_DIR 3rdparty/tvm)
set(CMAKE_BUILD_TYPE Release)
set(USE_CUDA OFF)
set(USE_CUTLASS OFF)
set(USE_CUBLAS OFF)
set(USE_ROCM OFF)
set(USE_VULKAN OFF)
set(USE_METAL OFF)
set(USE_OPENCL OFF)
set(USE_OPENCL_ENABLE_HOST_PTR OFF)
set(USE_THRUST OFF)
CMAKE
  (cd mlc-llm/build && cmake .. >cmake.log 2>&1 && make -j "$JOBS" >make.log 2>&1)
fi

export MLC_LLM_SOURCE_DIR="$WORK/mlc-llm" TVM_SOURCE_DIR="$TVM"

echo "==> WebAssembly runtimes"
# What web/prep_emcc_deps.sh runs, with one flag it has no way to take. Clang writes source paths into assertion
# messages and function names, so the build directory, and the home it sits in, would reach a public file (upstream's
# libraries carry their builder's). Mapping it also makes the output independent of the directory.
EMCC_MAPPED="emcc -ffile-prefix-map=$WORK=/build"
if [ ! -f "$TVM/web/dist/wasm/wasm_runtime.bc" ] || [ ! -f mlc-llm/web/dist/wasm/mlc_wasm_runtime.bc ]; then
  (cd mlc-llm/web && make EMCC="$EMCC_MAPPED") >emcc-deps.log 2>&1
  (cd "$TVM/web" && TVM_HOME="$TVM" make EMCC="$EMCC_MAPPED") >>emcc-deps.log 2>&1
fi

export PYTHONPATH="$TVM/python:$TVM/3rdparty/tvm-tirx/python:$WORK/mlc-llm/python"
export TVM_LIBRARY_PATH="$TVM/build/lib" LD_LIBRARY_PATH="$TVM/build/lib"

echo "==> Model libraries"
python - "$REPO" "$WORK" "$MLC_LLM_COMMIT" "$EMSDK_VERSION" "$PREFILL_CHUNK_SIZE" <<'PY'
import base64, hashlib, json, subprocess, sys, urllib.request
from pathlib import Path

repo, work, mlc_commit, emsdk, prefill = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
web = json.loads((repo / "web/src/lib/config.defaults.json").read_text(encoding="utf-8"))["web"]
package = json.loads((repo / "web/package.json").read_text(encoding="utf-8"))
out = repo / "web/public" / web["webllm_lib_dir"]
out.mkdir(parents=True, exist_ok=True)

libs = {}
for model in web["webllm_models"]:
    config_dir = work / "configs" / model["id"]
    config_dir.mkdir(parents=True, exist_ok=True)
    config = config_dir / "mlc-chat-config.json"
    url = f"https://huggingface.co/mlc-ai/{model['id']}/raw/{model['revision']}/mlc-chat-config.json"
    urllib.request.urlretrieve(url, config)
    target = out / model["lib"]
    subprocess.run([sys.executable, "-m", "mlc_llm", "compile", str(config), "--device", "webgpu",
                    "--overrides", f"prefill_chunk_size={prefill}", "-o", str(target)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    data = target.read_bytes()
    libs[model["lib"]] = {"model_id": model["id"], "weights_revision": model["revision"], "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()}
    print(f'  {model["lib"]}: {len(data)} bytes, lib_sri '
          f'"sha256-{base64.b64encode(hashlib.sha256(data).digest()).decode()}"')

    assert str(work).encode() not in data, f"{target.name} carries the build directory: see EMCC_MAPPED"

tvm = work / "mlc-llm/3rdparty/tvm"
head = lambda path: subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, text=True,
                                   capture_output=True).stdout.strip()
llvm = subprocess.run(["llvm-config", "--version"], check=True, text=True, capture_output=True).stdout.strip()
(out / "MANIFEST.json").write_text(json.dumps({
    "runtime": f'@mlc-ai/web-llm@{package["dependencies"]["@mlc-ai/web-llm"]}',
    "built_with": {"mlc-llm": head(work / "mlc-llm"), "tvm (mlc-ai/relax)": head(tvm), "emscripten": emsdk,
                   "tvm's llvm": llvm},
    "prefill_chunk_size": int(prefill),
    "build_script": "scripts/build_webllm_libs.sh",
    "libs": libs,
}, indent=2) + "\n", encoding="utf-8")

parts = [("MLC-LLM (Apache-2.0)", work / "mlc-llm/LICENSE"), ("MLC-LLM NOTICE", work / "mlc-llm/NOTICE"),
         ("Apache TVM (Apache-2.0, with its third-party list)", tvm / "LICENSE"), ("Apache TVM NOTICE", tvm / "NOTICE"),
         ("DLPack", tvm / "3rdparty/tvm-tirx/3rdparty/tvm-ffi/3rdparty/dlpack/LICENSE"),
         ("Emscripten, whose libc, libc++ and compiler-rt are linked in",
          work / "emsdk/upstream/emscripten/LICENSE")]
text = ("The WebLLM model libraries under /webllm/ are built from source by scripts/build_webllm_libs.sh; what each\n"
        "was built from is in its directory's MANIFEST.json. These are the licences and notices of what they link.\n")
for title, path in parts:
    text += f"\n{'=' * 78}\n{title}\n{'=' * 78}\n\n{path.read_text(encoding='utf-8')}"
(repo / "web/public/licenses/webllm-libs.txt").write_text(text, encoding="utf-8")
PY
echo "==> Done. Copy the lib_sri values above to web.webllm_models in src/qsh/config/defaults.py."

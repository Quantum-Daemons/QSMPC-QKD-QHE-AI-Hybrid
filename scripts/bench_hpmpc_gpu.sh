#!/usr/bin/env bash
# hpmpc's secret-shared matrix multiplication on this machine's CPU and on its GPU, and the checks that make the
# GPU figure mean something (Round 86, G3).
#
#   bash scripts/bench_hpmpc_gpu.sh OUT_JSON [WORKDIR]
#   REPORT_ONLY=1 bash scripts/bench_hpmpc_gpu.sh OUT_JSON WORKDIR    # rewrite the record from a finished WORKDIR
#
# hpmpc (MIT) is the MPC framework under PIGEON. Its function 36 multiplies two secret-shared N x N matrices 20 times
# under the Trio protocol (PROTOCOL=5), three parties on loopback, either with its tiled CPU loop (USE_CUDA_GEMM=0) or
# with a CUTLASS GEMM on the GPU (USE_CUDA_GEMM=1). Nothing here ships it or offers it on a page: this is a recorded
# measurement.
#
# It builds in WORKDIR (default: a new temporary directory), never in submodules/hpmpc:
#   - a clone of the submodule at its pinned commit, with only nn/PIGEON (flexNN, MIT) initialised, whose headers the
#     benchmark includes; never nn/Pygeon, which carries no licence (DATA_LICENSES.md);
#   - CUTLASS at CUTLASS_TAG (BSD-3-Clause), headers only.
# One line of hpmpc is changed in the clone, and the script says so: the GPU branch of function 36 calls `T::GEMM`
# where no `T` is declared (the benchmark's share type is `S`), at the pinned commit and at upstream's head on
# 2026-11-06, so that branch does not compile there.
#
# What it refuses to report, because each happened while it was being written:
#   - a figure from a build that failed: hpmpc's Makefile keeps going after a compiler error and leaves the previous
#     executable in place, so a GPU "benchmark" ran the matrix-multiplication TEST from the build before it. Each build
#     here is checked for a compiler error and for a fresh executable, and each run for the test's output;
#   - a GPU figure without the GPU: the benchmark checks no result, so with the GPU hidden it still finishes. The
#     correctness test (function 57) is run with the GPU visible, where it must pass, and hidden, where its matrix
#     multiplication must fail, and each benchmark run's sent volume is recorded beside the CPU run's at the same N.
set -euo pipefail
OUT="${1:?usage: bench_hpmpc_gpu.sh OUT_JSON [WORKDIR]}"
# Absolute before the cd below: a relative path was written into WORKDIR, and a finished run lost its record.
OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
W="${2:-$(mktemp -d -t qsh-hpmpc-XXXXXX)}"
CUTLASS_TAG="${CUTLASS_TAG:-v4.8.1}"
CPU_SIZES="${CPU_SIZES:-128 256 512 1024 2048}"
GPU_SIZES="${GPU_SIZES:-128 256 512 1024 2048 4096}"
CPU_LARGE="${CPU_LARGE-4096}"        # measured once: it takes minutes (CPU_LARGE= skips it)
RUNS="${RUNS:-3}"
# A CPU time measures whatever else the machine is doing. Before each run the script waits for the 1-minute load
# to fall under MAX_LOAD (the benchmark's own three parties add about 3 while they run), records it, and stops after
# WAIT_MAX seconds rather than time a run beside another job: one did, a 6-worker job that arrived mid-measurement and
# took the CPU figure at N = 1024 from 4.8 s to 8-10 s.
MAX_LOAD="${MAX_LOAD:-4}"
WAIT_MAX="${WAIT_MAX:-3600}"
export LC_ALL=C                      # compiler messages in English, so the error check reads them
mkdir -p "$W"; cd "$W"
die() { echo "FAIL $*" >&2; exit 1; }

command -v nvcc >/dev/null || die "nvcc not found"
CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
ARCH="sm_${CAP}"

[ -d hpmpc ] || git clone -q "$REPO/submodules/hpmpc" hpmpc
git -C hpmpc submodule update --init nn/PIGEON >/dev/null 2>&1
[ -z "$(ls -A hpmpc/nn/Pygeon 2>/dev/null)" ] || die "nn/Pygeon is populated; it carries no licence"
[ -d cutlass ] || git clone -q --depth 1 --branch "$CUTLASS_TAG" https://github.com/NVIDIA/cutlass.git cutlass

BENCH=hpmpc/programs/benchmarks/bench_nn.hpp
if grep -q "        T::GEMM(A, B, C, mul_m, mul_n, mul_k, true);" "$BENCH"; then
  sed -i 's/        T::GEMM(A, B, C, mul_m, mul_n, mul_k, true);/        S::GEMM(A, B, C, mul_m, mul_n, mul_k, true);/' "$BENCH"
fi
[ "$(git -C hpmpc diff --numstat | wc -l)" = 1 ] || die "the clone differs from the pin by more than the one line"
echo "local change: $(git -C hpmpc diff | grep '^[-+] ' | tr '\n' ' ')"

# The GEMM object only: USE_CUDA_GEMM=1 links nothing else.
make -s -C hpmpc/core/cuda bin/gemm_cutlass_int.o arch="$ARCH" CUDA_PATH="$(dirname "$(dirname "$(command -v nvcc)")")" \
  CUTLASS_PATH="$W/cutlass" > gemm_build.log 2>&1 || die "the GEMM object did not build (gemm_build.log)"
cuobjdump --list-elf hpmpc/core/cuda/bin/gemm_cutlass_int.o | grep -q "$ARCH" || die "the GEMM object carries no $ARCH code"

build() {  # function gemm n
  local t0 log="$W/build_$1_$2_$3.log"; t0=$(date +%s); sleep 1
  ( cd hpmpc && make -j6 PARTY=all FUNCTION_IDENTIFIER="$1" PROTOCOL=5 USE_CUDA_GEMM="$2" NUM_INPUTS="$3" ) > "$log" 2>&1 || true
  ! grep -q -E "error:|fatal" "$log" || die "build $1/$2/$3 has a compiler error ($log)"
  for p in 0 1 2; do [ "$(stat -c %Y "hpmpc/executables/run-P$p.o")" -ge "$t0" ] || die "build $1/$2/$3 left a stale run-P$p"; done
}
quiet() {
  local waited=0
  while awk -v m="$MAX_LOAD" '{exit !($1 >= m)}' /proc/loadavg; do
    [ "$waited" -lt "$WAIT_MAX" ] || die "the machine stayed busy (load $(cut -d' ' -f1 /proc/loadavg)); a CPU time would measure another job"
    sleep 30; waited=$((waited + 30))
  done
}
run() {  # log [env]
  ( cd hpmpc && env ${2:-} timeout 3600 bash scripts/run.sh -p all -n 3 ) > "$1" 2>&1 || true
}

if [ "${REPORT_ONLY:-0}" != 1 ]; then
# 1. Correctness, GPU visible and hidden.
build 57 1 1
run "$W/test57_gpu.log"
[ "$(grep -c 'Passed 3 out of 3 tests' "$W/test57_gpu.log")" = 3 ] || die "the matrix-multiplication test did not pass on the GPU"
run "$W/test57_hidden.log" "CUDA_VISIBLE_DEVICES="
grep -q "Matrix Multiplication Failed" "$W/test57_hidden.log" || die "with the GPU hidden the test still passed: the GPU was not used"

# 2. Function 36.
: > runs.tsv
measure() {  # gemm n runs
  build 36 "$1" "$2"
  for r in $(seq 1 "$3"); do
    local log="$W/run36_$1_$2_$r.log" load; quiet; load=$(cut -d' ' -f1 /proc/loadavg); run "$log"
    ! grep -q "Testing" "$log" || die "run $1/$2 printed the test's output: a stale executable"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$r" \
      "$(grep 'P0, PID0: Time measured to perform computation chrono' "$log" | awk '{print $NF}' | tr -d s)" \
      "$(grep 'P0, PID0: Time measured to perform computation clock' "$log" | awk '{print $NF}' | tr -d s)" \
      "$(grep '^P0, ONLINE, PID0: Sending' "$log" | sed 's/.*players://' | cut -d, -f1 | tr -d ' ')" "$load" >> runs.tsv
  done
}
for n in $GPU_SIZES; do measure 1 "$n" "$RUNS"; done
for n in $CPU_SIZES; do measure 0 "$n" "$RUNS"; done
for n in $CPU_LARGE; do measure 0 "$n" 1; done
fi

python3 - "$OUT" "$W" "$ARCH" "$CUTLASS_TAG" "$MAX_LOAD" <<'PY'
import json, platform, statistics, subprocess, sys, time
out, w, arch, tag, max_load = sys.argv[1:6]
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
rows = [l.split("\t") for l in open(f"{w}/runs.tsv").read().splitlines() if l]
res = {}
for g, n, r, ch, cl, sent, load in rows:
    res.setdefault((int(g), int(n)), []).append({"run": int(r), "wall_s": float(ch), "cpu_s": float(cl), "p0_sent": sent,
                                                 "load1_before": float(load)})
table = []
for (g, n), rs in sorted(res.items()):
    table.append({"device": "gpu" if g else "cpu", "n": n, "runs": rs,
                  "median_wall_s": statistics.median(r["wall_s"] for r in rs)})
med = {(t["device"], t["n"]): t["median_wall_s"] for t in table}
speed = {str(n): round(med[("cpu", n)] / med[("gpu", n)], 2) for (d, n) in med if d == "gpu" and ("cpu", n) in med}
rec = {
    "machine": {"gpu": sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"), "arch": arch,
                "nvcc": sh("nvcc --version | tail -1"), "cpu": sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2").strip(),
                "hardware_threads": int(sh("nproc --all")), "kernel": platform.release(), "date": time.strftime("%Y-%m-%d")},
    "versions": {"hpmpc": sh(f"git -C {w}/hpmpc rev-parse HEAD"), "flexNN (nn/PIGEON)": sh(f"git -C {w}/hpmpc/nn/PIGEON rev-parse HEAD"),
                 "cutlass": f"{tag} {sh(f'git -C {w}/cutlass rev-parse HEAD')}"},
    "setup": {"function": 36, "what": "N x N secret-shared matrix multiplication, 20 times per run",
              "protocol": "Trio (PROTOCOL=5), three parties on loopback, one GPU shared by the three",
              "bitlength": 32, "local_change": sh(f"git -C {w}/hpmpc diff | grep '^[-+] '"),
              "gemm_object_only": "USE_CUDA_GEMM=1 links the GEMM object alone; the convolution objects were not built",
              "each_run_waited_for_load1_below": float(max_load)},
    "checks": {"test57_gpu": f"{open(f'{w}/test57_gpu.log').read().count('Passed 3 out of 3 tests')} of 3 parties passed 3 of 3 tests",
               "test57_gpu_hidden": ("matrix multiplication failed" if "Matrix Multiplication Failed" in open(f"{w}/test57_hidden.log").read()
                                     else "MATRIX MULTIPLICATION PASSED WITHOUT THE GPU"),
               "sent_volume": "each GPU run's P0 sent volume recorded beside the CPU run's at the same N"},
    "results": table, "cpu_over_gpu_median": speed,
}
json.dump(rec, open(out, "w"), indent=1); open(out, "a").write("\n")
print(json.dumps(speed))
PY
echo "wrote $OUT (work in $W)"

#!/usr/bin/env bash
# hpmpc's secret-shared matrix multiplication and convolution on this machine's CPU and on its GPU, and the checks that
# make each GPU figure mean something (Round 86, G3).
#
#   bash scripts/bench_hpmpc_gpu.sh OUT_JSON [WORKDIR]                # both parts (PARTS="gemm conv")
#   PARTS=conv bash scripts/bench_hpmpc_gpu.sh OUT_JSON [WORKDIR]     # one part, its record at OUT_JSON
#   REPORT_ONLY=1 bash scripts/bench_hpmpc_gpu.sh OUT_JSON WORKDIR    # rewrite the record(s) from a finished WORKDIR
#
# With both parts the convolution's record goes beside OUT_JSON, "-conv" before ".json".
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
# THE CONVOLUTION (functions 33 to 35, the layers of CryptGPU's figure 1): one flexNN Conv2d layer under the same
# protocol, on three paths: the CPU (USE_CUDA_GEMM=0); im2col on the CPU and the GEMM on the GPU (=1, the GEMM object
# function 36 uses); and the whole convolution on the GPU (=2, CUTLASS's implicit-GEMM convolution). The last needs
# more of the clone, each printed like the line above:
#   - CUTLASS 4.8.1's convolution header calls a function from its problem-size headers without including them, so the
#     convolution object is compiled with `-include` of both: a compiler flag, no source change;
#   - the 16-bit instantiation of hpmpc's convolution fails a CUTLASS static assertion, so it is commented out in the
#     clone (upstream comments out the 8-bit one for the same reason; the benchmark's shares are 32-bit);
#   - functions 33 and 34 do not compile with hpmpc's default PRINT_TIMINGS=1: core/utils/print.hpp declares the
#     layer-stats functions only for FUNCTION_IDENTIFIER >= 35, and flexNN's simple_nn.h calls one with no dependent
#     argument (at the pin and at upstream's head on 2026-11-06). The flag governs per-layer stats, which a one-layer
#     benchmark never prints, and the time comes from protocol_executer.hpp either way, so all three convolution
#     functions are built with PRINT_TIMINGS=0 and share one configuration.
# hpmpc has no correctness test for the convolution, so this script carries one, inserted into function 33 for its own
# builds and taken out before anything is timed: small fixed integers through the secret-shared layer (the input in
# fixed point, the weights as integers, so the truncated output is the plain convolution), revealed and compared with a
# plain convolution in the program and, as a sum, with an independent NumPy one. Each path must reproduce it; with the
# GPU hidden, neither GPU path may.
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
PARTS="${PARTS:-gemm conv}"
CONV_SIZES_33="${CONV_SIZES_33:-64 128 256 512 1024}"   # n: an n x n x 3 input, 11 x 11 kernel, 64 out, stride 4, pad 2
CONV_SIZES_34="${CONV_SIZES_34:-8 32 128 512}"          # k: a batch of k 32 x 32 x 3 inputs, the same layer
CONV_SIZES_35="${CONV_SIZES_35:-8 16 32 64}"            # n: an n x n x 512 input, 3 x 3 kernel, 512 out, stride 1, pad 1
CONV_CHECK_N=16
# A CPU time measures whatever else the machine is doing. Before each run the script waits for the 1-minute load
# to fall under MAX_LOAD (the benchmark's own three parties add about 3 while they run), records it, and stops after
# WAIT_MAX seconds rather than time a run beside another job: one did, a 6-worker job that arrived mid-measurement and
# took the CPU figure at N = 1024 from 4.8 s to 8-10 s.
MAX_LOAD="${MAX_LOAD:-4}"
WAIT_MAX="${WAIT_MAX:-3600}"
export LC_ALL=C                      # compiler messages in English, so the error check reads them
has() { case " $PARTS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
GEMM_OUT="$OUT"; CONV_OUT="$OUT"
if has gemm && has conv; then CONV_OUT="${OUT%.json}-conv.json"; fi
mkdir -p "$W"; cd "$W"
die() { echo "FAIL $*" >&2; exit 1; }

command -v nvcc >/dev/null || die "nvcc not found"
CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
ARCH="sm_${CAP}"
CUDA_PATH="$(dirname "$(dirname "$(command -v nvcc)")")"

[ -d hpmpc ] || git clone -q "$REPO/submodules/hpmpc" hpmpc
git -C hpmpc submodule update --init nn/PIGEON >/dev/null 2>&1
[ -z "$(ls -A hpmpc/nn/Pygeon 2>/dev/null)" ] || die "nn/Pygeon is populated; it carries no licence"
[ -d cutlass ] || git clone -q --depth 1 --branch "$CUTLASS_TAG" https://github.com/NVIDIA/cutlass.git cutlass

BENCH=hpmpc/programs/benchmarks/bench_nn.hpp
CONV_CU=hpmpc/core/cuda/conv_cutlass_int.cu
fix_gemm_line() {
  if grep -q "        T::GEMM(A, B, C, mul_m, mul_n, mul_k, true);" "$BENCH"; then
    sed -i 's/        T::GEMM(A, B, C, mul_m, mul_n, mul_k, true);/        S::GEMM(A, B, C, mul_m, mul_n, mul_k, true);/' "$BENCH"
  fi
}
fix_gemm_line
if has conv; then
  python3 - "$CONV_CU" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text()
head = "template void conv2d_cutlass<uint16_t>("
if head in s and "/* removed in this clone" not in s:
    a = s.index(head); b = s.index(";", a) + 1
    s = s[:a] + "/* removed in this clone: CUTLASS 4.8.1 rejects the 16-bit convolution with a static assertion\n" + s[a:b] + " */" + s[b:]
    p.write_text(s)
PY
fi
clone_is_only_ours() {
  local want="programs/benchmarks/bench_nn.hpp"
  has conv && want="core/cuda/conv_cutlass_int.cu programs/benchmarks/bench_nn.hpp"
  [ "$(git -C hpmpc diff --name-only | sort | xargs)" = "$want" ] || die "the clone differs from the pin beyond this script's changes"
  [ "$(git -C hpmpc diff --numstat -- programs/benchmarks/bench_nn.hpp | cut -f1,2 | tr '\t' ' ')" = "1 1" ] \
    || die "the clone differs from the pin by more than the one line"
}
clone_is_only_ours
echo "local change: $(git -C hpmpc diff -- "${BENCH#hpmpc/}" | grep '^[-+] ' | tr '\n' ' ')"
has conv && echo "local change: the 16-bit instantiation in ${CONV_CU#hpmpc/} is commented out"

# The GEMM object: USE_CUDA_GEMM=1 links nothing else, and =2 links it beside the convolution object.
make -s -C hpmpc/core/cuda bin/gemm_cutlass_int.o arch="$ARCH" CUDA_PATH="$CUDA_PATH" \
  CUTLASS_PATH="$W/cutlass" > gemm_build.log 2>&1 || die "the GEMM object did not build (gemm_build.log)"
cuobjdump --list-elf hpmpc/core/cuda/bin/gemm_cutlass_int.o | grep -q "$ARCH" || die "the GEMM object carries no $ARCH code"
if has conv; then
  make -s -C hpmpc/core/cuda bin/conv_cutlass_int_NCHW.o arch="$ARCH" CUDA_PATH="$CUDA_PATH" CUTLASS_PATH="$W/cutlass" \
    NVCCOPTs="-arch=$ARCH --expt-relaxed-constexpr -std=c++20 -include cutlass/conv/conv2d_problem_size.h -include cutlass/conv/conv3d_problem_size.h" \
    > conv_build.log 2>&1 || die "the convolution object did not build (conv_build.log)"
  cuobjdump --list-elf hpmpc/core/cuda/bin/conv_cutlass_int_NCHW.o | grep -q "$ARCH" \
    || die "the convolution object carries no $ARCH code"
fi

build() {  # function gemm n [make arguments]
  local f="$1" g="$2" n="$3"; shift 3
  local t0 log="$W/build_${f}_${g}_${n}.log"; t0=$(date +%s); sleep 1
  ( cd hpmpc && make -j6 PARTY=all FUNCTION_IDENTIFIER="$f" PROTOCOL=5 USE_CUDA_GEMM="$g" NUM_INPUTS="$n" "$@" ) > "$log" 2>&1 || true
  ! grep -q -E "error:|fatal" "$log" || die "build $f/$g/$n has a compiler error ($log)"
  for p in 0 1 2; do [ "$(stat -c %Y "hpmpc/executables/run-P$p.o")" -ge "$t0" ] || die "build $f/$g/$n left a stale run-P$p"; done
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
record() {  # function gemm n run log load -> one row of runs.tsv
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" \
    "$(grep 'P0, PID0: Time measured to perform computation chrono' "$5" | awk '{print $NF}' | tr -d s)" \
    "$(grep 'P0, PID0: Time measured to perform computation clock' "$5" | awk '{print $NF}' | tr -d s)" \
    "$(grep '^P0, ONLINE, PID0: Sending' "$5" | sed 's/.*players://' | cut -d, -f1 | tr -d ' ')" "$6"
}

if [ "${REPORT_ONLY:-0}" != 1 ] && has gemm; then
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
    record 36 "$1" "$2" "$r" "$log" "$load" | cut -f2- >> runs.tsv
  done
}
for n in $GPU_SIZES; do measure 1 "$n" "$RUNS"; done
for n in $CPU_SIZES; do measure 0 "$n" "$RUNS"; done
for n in $CPU_LARGE; do measure 0 "$n" 1; done
fi

if [ "${REPORT_ONLY:-0}" != 1 ] && has conv; then
# 3. The convolution's correctness, on every path, against NumPy; then hidden.
"$REPO/.venv/bin/python" - "$CONV_CHECK_N" > conv_reference.txt <<'PY'
import sys
import numpy as np
H = int(sys.argv[1]); C, K, O, S, P = 3, 11, 64, 4, 2
OH = (H + 2 * P - K) // S + 1
x = np.fromfunction(lambda c, h, w: ((c * 7 + h * 3 + w) % 5) + 1, (C, H, H), dtype=np.int64)
w = np.fromfunction(lambda o, c, kh, kw: ((o + c * 2 + kh * 3 + kw) % 3) + 1, (O, C, K, K), dtype=np.int64)
xp = np.pad(x, ((0, 0), (P, P), (P, P)))
out = np.array([[[np.sum(w[o] * xp[:, i * S:i * S + K, j * S:j * S + K]) for j in range(OH)] for i in range(OH)]
                for o in range(O)])
print(out.size, int(out.sum()))
PY
python3 - "$BENCH" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text()
anchor = "        conv.set_layer(input_shape);\n        conv.forward(input, false);\n        dummy_reveal<Share>();\n    }\n}\n#endif"
assert anchor in s, "function 33's body is not where this check expects it"
check = r'''        conv.set_layer(input_shape);
#if FUNCTION_IDENTIFIER == 33
        // QSH CONV CHECK (bench_hpmpc_gpu.sh, check builds only): fixed small integers through the secret-shared
        // layer, the input in fixed point and the weights as integers, so the truncated output is the plain convolution.
        const int qC = 3, qH = NUM_INPUTS, qK = 11, qO = 64, qS = 4, qP = 2, qOH = (qH + 2 * qP - qK) / qS + 1;
        auto qx = [&](int c, int h, int w) -> long { return ((c * 7 + h * 3 + w) % 5) + 1; };
        auto qw = [&](int o, int c, int kh, int kw) -> long { return ((o + c * 2 + kh * 3 + kw) % 3) + 1; };
        for (int c = 0; c < qC; c++)
            for (int h = 0; h < qH; h++)
                for (int w = 0; w < qH; w++)
                    input(0, (c * qH + h) * qH + w) = S((UINT_TYPE)(qx(c, h, w) << FRACTIONAL));
        for (int o = 0; o < qO; o++)
            for (int c = 0; c < qC; c++)
                for (int kh = 0; kh < qK; kh++)
                    for (int kw = 0; kw < qK; kw++)
                        conv.kernel(o, (c * qK + kh) * qK + kw) = S((UINT_TYPE)qw(o, c, kh, kw));
        for (int o = 0; o < qO; o++)
            conv.bias(o) = S((UINT_TYPE)0);
#endif
        conv.forward(input, false);
#if FUNCTION_IDENTIFIER == 33
        if (i == 0)
        {
            const int qN = qO * qOH * qOH;
            for (int j = 0; j < qN; j++)
                conv.output(j / (qOH * qOH), j % (qOH * qOH)).prepare_reveal_to_all();
            Share::communicate();
            long bad = 0, maxerr = 0, sum = 0;
            for (int j = 0; j < qN; j++)
            {
                DATATYPE v = conv.output(j / (qOH * qOH), j % (qOH * qOH)).complete_reveal_to_all();
                UINT_TYPE got[DATTYPE / BITLENGTH];
                unorthogonalize_arithmetic(&v, got, 1);
                const int o = j / (qOH * qOH), oh = (j % (qOH * qOH)) / qOH, ow = j % qOH;
                long ref = 0;
                for (int c = 0; c < qC; c++)
                    for (int kh = 0; kh < qK; kh++)
                        for (int kw = 0; kw < qK; kw++)
                        {
                            const int r = oh * qS - qP + kh, q = ow * qS - qP + kw;
                            if (r >= 0 && r < qH && q >= 0 && q < qH)
                                ref += qx(c, r, q) * qw(o, c, kh, kw);
                        }
                const long err = (long)(int32_t)(got[0] - (UINT_TYPE)ref);
                maxerr = std::max(maxerr, std::abs(err));
                bad += std::abs(err) > 1;
                sum += (long)(int32_t)got[0];
            }
            std::cout << "QSH_CONV_CHECK P" << PARTY << " n=" << qN << " bad=" << bad << " maxerr=" << maxerr
                      << " sum=" << sum << std::endl;
        }
#endif
        dummy_reveal<Share>();
    }
}
#endif'''
p.write_text(s.replace(anchor, check, 1))
PY
for g in 0 1 2; do
  build 33 "$g" "$CONV_CHECK_N" PRINT_TIMINGS=0
  run "$W/convcheck_$g.log"
  [ "$g" = 0 ] || run "$W/convcheck_${g}_hidden.log" "CUDA_VISIBLE_DEVICES="
done
"$REPO/.venv/bin/python" - "$W" <<'PY' || die "the convolution check failed (convcheck_*.log)"
import re, sys
w = sys.argv[1]
n, ref = map(int, open(f"{w}/conv_reference.txt").read().split())
pat = r"QSH_CONV_CHECK P(\d) n=(\d+) bad=(\d+) maxerr=(\d+) sum=(-?\d+)"
for g in (0, 1, 2):
    good = [m for m in re.findall(pat, open(f"{w}/convcheck_{g}.log").read()) if int(m[2]) == 0 and int(m[1]) == n]
    assert {m[0] for m in good} == {"0", "1", "2"}, f"path {g}: not every party reproduced the convolution"
    for m in good:
        assert abs(int(m[4]) - ref) <= int(m[3]) * n, f"path {g} P{m[0]}: sum {m[4]} against NumPy's {ref}"
    if g:
        hidden = re.findall(pat, open(f"{w}/convcheck_{g}_hidden.log").read())
        assert all(int(m[2]) > 0 for m in hidden), f"path {g}: with the GPU hidden it still reproduced the convolution"
print(f"convolution check: every path reproduces NumPy's {n} outputs (sum {ref}); hidden, neither GPU path does")
PY
# Out again before anything is timed.
git -C hpmpc checkout -q -- "${BENCH#hpmpc/}"
fix_gemm_line
clone_is_only_ours

# 4. Functions 33 to 35 on the three paths.
: > conv_runs.tsv
measure_conv() {  # function gemm n runs
  build "$1" "$2" "$3" PRINT_TIMINGS=0
  for r in $(seq 1 "$4"); do
    local log="$W/run$1_$2_$3_$r.log" load; quiet; load=$(cut -d' ' -f1 /proc/loadavg); run "$log"
    ! grep -q -E "Testing|QSH_CONV_CHECK" "$log" || die "run $1/$2/$3 printed a test's output: a stale executable"
    record "$1" "$2" "$3" "$r" "$log" "$load" >> conv_runs.tsv
  done
}
for f in 33 34 35; do
  sizes="CONV_SIZES_$f"
  for n in ${!sizes}; do for g in 0 1 2; do measure_conv "$f" "$g" "$n" "$RUNS"; done; done
done
fi

if has gemm; then
python3 - "$GEMM_OUT" "$W" "$ARCH" "$CUTLASS_TAG" "$MAX_LOAD" "$PARTS" <<'PY'
import json, platform, statistics, subprocess, sys, time
out, w, arch, tag, max_load, parts = sys.argv[1:7]
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
              "bitlength": 32, "local_change": sh(f"git -C {w}/hpmpc diff -- programs/benchmarks/bench_nn.hpp | grep '^[-+] '"),
              "gemm_object_only": "USE_CUDA_GEMM=1 links the GEMM object alone"
                                  + ("" if "conv" in parts.split() else "; the convolution objects were not built"),
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
echo "wrote $GEMM_OUT (work in $W)"
fi

if has conv; then
python3 - "$CONV_OUT" "$W" "$ARCH" "$CUTLASS_TAG" "$MAX_LOAD" <<'PY'
import json, platform, re, statistics, subprocess, sys, time
out, w, arch, tag, max_load = sys.argv[1:6]
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
PATHS = {0: "cpu", 1: "gpu_gemm", 2: "gpu_conv"}
rows = [l.split("\t") for l in open(f"{w}/conv_runs.tsv").read().splitlines() if l]
res = {}
for f, g, n, r, ch, cl, sent, load in rows:
    res.setdefault((int(f), int(g), int(n)), []).append(
        {"run": int(r), "wall_s": float(ch), "cpu_s": float(cl), "p0_sent": sent, "load1_before": float(load)})
table = [{"function": f, "path": PATHS[g], "n": n, "runs": rs, "median_wall_s": statistics.median(x["wall_s"] for x in rs)}
         for (f, g, n), rs in sorted(res.items())]
med = {(t["function"], t["path"], t["n"]): t["median_wall_s"] for t in table}
speed = {}
for (f, p, n), m in med.items():
    if p != "cpu" and (f, "cpu", n) in med:
        speed.setdefault(str(f), {}).setdefault(str(n), {})[f"cpu_over_{p}"] = round(med[(f, "cpu", n)] / m, 2)
# The same protocol on each path: the same volume sent at the same size.
sent = {}
for t in table:
    sent.setdefault((t["function"], t["n"]), set()).update(x["p0_sent"] for x in t["runs"])
unequal = sorted(f"{f}/{n}" for (f, n), v in sent.items() if len(v) != 1)
n_ref, sum_ref = map(int, open(f"{w}/conv_reference.txt").read().split())
pat = r"QSH_CONV_CHECK P(\d) n=(\d+) bad=(\d+) maxerr=(\d+) sum=(-?\d+)"
def check(log):
    found = re.findall(pat, open(f"{w}/{log}").read())
    good = sorted({m[0] for m in found if int(m[2]) == 0})
    return {"parties_reproducing": len(good), "lines": [" ".join(m) for m in found]}
rec = {
    "machine": {"gpu": sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"), "arch": arch,
                "nvcc": sh("nvcc --version | tail -1"), "cpu": sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2").strip(),
                "hardware_threads": int(sh("nproc --all")), "kernel": platform.release(), "date": time.strftime("%Y-%m-%d")},
    "versions": {"hpmpc": sh(f"git -C {w}/hpmpc rev-parse HEAD"), "flexNN (nn/PIGEON)": sh(f"git -C {w}/hpmpc/nn/PIGEON rev-parse HEAD"),
                 "cutlass": f"{tag} {sh(f'git -C {w}/cutlass rev-parse HEAD')}"},
    "setup": {
        "functions": {"33": "an n x n x 3 input, 11 x 11 kernel, 64 output channels, stride 4, padding 2 (CryptGPU fig. 1a)",
                      "34": "a batch of k 32 x 32 x 3 inputs, the same layer (fig. 1b)",
                      "35": "an n x n x 512 input, 3 x 3 kernel, 512 output channels, stride 1, padding 1 (fig. 1c)"},
        "paths": {"cpu": "USE_CUDA_GEMM=0: im2col and the GEMM on the CPU",
                  "gpu_gemm": "USE_CUDA_GEMM=1: im2col on the CPU, the GEMM on the GPU (the object function 36 uses)",
                  "gpu_conv": "USE_CUDA_GEMM=2: the whole convolution on the GPU (CUTLASS implicit-GEMM convolution)"},
        "protocol": "Trio (PROTOCOL=5), three parties on loopback, one GPU shared by the three",
        "bitlength": 32, "fractional_bits": 5, "build_flags": "PRINT_TIMINGS=0 for every convolution build",
        "local_changes": [sh(f"git -C {w}/hpmpc diff -- programs/benchmarks/bench_nn.hpp | grep '^[-+] '"),
                          "core/cuda/conv_cutlass_int.cu: the 16-bit instantiation commented out",
                          "the convolution object compiled with -include of cutlass/conv/conv2d_problem_size.h and "
                          "conv3d_problem_size.h"],
        "each_run_waited_for_load1_below": float(max_load)},
    "checks": {
        "reference": {"outputs": n_ref, "sum": sum_ref, "by": "NumPy, independently of the program's own plain convolution"},
        "cpu": check("convcheck_0.log"), "gpu_gemm": check("convcheck_1.log"), "gpu_conv": check("convcheck_2.log"),
        "gpu_gemm_hidden": check("convcheck_1_hidden.log"), "gpu_conv_hidden": check("convcheck_2_hidden.log"),
        "sent_volume_unequal_at": unequal},
    "results": table, "cpu_over_gpu_median": speed,
}
json.dump(rec, open(out, "w"), indent=1); open(out, "a").write("\n")
print(json.dumps(speed))
if unequal:
    raise SystemExit(f"the paths sent different volumes at {unequal}: not the same protocol work")
PY
echo "wrote $CONV_OUT (work in $W)"
fi

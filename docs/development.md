# Development environment

## Quickstart, with its notes

```bash
# 1. Python 3.12 venv + core (crypto + quantum CPU) stack
bash env/setup_venv.sh      # or: make setup
source .venv/bin/activate

# 2. Run the test suite (crypto correctness, FHE accuracy, E2E smoke)
make test             # pytest prints the count; the Node/OpenFHE-gated slow tests are deselected
# No count is quoted here. This line read "187 collected, 183 passed" while the suite had grown to
# 393 - and `Makefile` two files over already carries the same lesson from its own help text, which
# said 174 when the truth was 236. A count maintained by hand in a comment drifts; pytest reports it.
# `make test` already passes -m "not slow", which is what deselects the gated ones.
# `make test-all` runs every test. The OpenFHE bootstrap test runs in a subprocess, so a native crash
# fails that test rather than the suite; scripts/ci_local.sh and CI run it too.

# 3. Optional tiers
pip install -r requirements-ai.txt  # torch (CUDA 13 auto), PennyLane (QFL), XGBoost, MedMNIST, ONNX, MCP SDK
pip install -r requirements-fhe.txt  # TenSEAL + OpenFHE (commercial-OK CKKS)
pip install -r requirements-api.txt  # FastAPI backend

# 4. Run an E2E use case (writes results/ artifacts)
python -m qsh.orchestration.pipeline --usecase finance_aml --out results/

# 5. Classical-vs-quantum benchmark
bash scripts/run_benchmark.sh linear

# 6. Web: API + SPA (two terminals)
uvicorn api.main:app --port 8000      # backend
cd web && npm install && npm run dev    # SPA at http://localhost:5173
```

GPU quantum simulation (Blackwell): see **Dev environment** below - use CUDA-Q /
PennyLane `lightning.gpu`, **not** `qiskit-aer-gpu`.

---

## Dev environment

Target box: **AlmaLinux 9.7, i5-13600K, 128 GB DDR5, NVIDIA RTX PRO 6000 Blackwell 96 GB,
CUDA 13.0**. Verified: Python 3.12.11, gcc 11.5, cmake 3.31, Node 22, Docker 29,
`torch 2.13.0+cu130` with `cuda.is_available() == True`, nvcc V13.0.88, driver 580.105.08
(CUDA 13 wheels require 580.65.06 or newer on Linux). Re-measured 2026-08-14; the line read
`torch 2.12+cu130` until then. CUDA 13.0 became a **stable** PyTorch build target in 2.11, so this
box is no longer on a leading edge - the capability is verified by launching a real kernel and
checking that `get_arch_list()` contains the device's `sm_120`, rather than by trusting a version
string here.

### NOTE GPU quantum simulation on Blackwell
`qiskit-aer-gpu` (cu11/cu12 wheels) **does not support Blackwell `sm_120`**. Use:
- **CUDA-Q** (`cudaq`) and/or **PennyLane `lightning.gpu`** (wraps cuQuantum cuStateVec, SM 7.0+).
- **Qiskit / Qiskit-Aer (CPU)** for circuit authoring and small simulations (used here).
- See [`requirements-gpu.txt`](../requirements-gpu.txt).

### FHE backends (commercial-OK only)
- **TenSEAL** (Apache-2.0, CPU CKKS via Microsoft SEAL) - the always-available default (`TenSEALBackend`).
- **OpenFHE** (BSD-2-Clause, CPU CKKS **with approximate bootstrapping**) - the live
  `OpenFHEBackend` via the `openfhe` wheel, pinned to the 1.5 series (`openfhe>=1.5.1,<1.6`: OpenFHE
  released 1.6.1 on 2026-10-09 and its 1.6.1 wheel on 2026-10-27, and that wheel's module lacks the
  `FHECKKSRNS` the bootstrap path calls). The
  wheel needs a recent libstdc++ (Ubuntu 24.04 / the Debian-trixie Docker image); on RHEL 9 build the
  bindings with `env/build_openfhe_el9.sh`, and on an older host the import fails with a clear
  `ImportError`.
- **HEonGPU** (Apache-2.0, GPU CKKS/BFV + collective bootstrapping) - a build-gated reference, not a
  running path: `HeongpuBackend` raises because HEonGPU exposes no Python API, `gpu_mpc_adapter`
  returns no rows unless the submodule has been built with `scripts/build_gpu_mpc.sh`, and no measured
  HEonGPU run is recorded in this repository. It is not a per-run selector option.

> **DESILO `desilofhe` was removed.** Its license is **non-commercial** - commercial use requires a separate
> paid license ([fhe.desilo.dev/latest/license](https://fhe.desilo.dev/latest/license/)). To keep the stack
> unambiguously commercial-OK, the FHE layer uses TenSEAL + OpenFHE, with HEonGPU as the build-gated GPU reference.

### Non-linear MPC backend (the hybrid's other half)
- **MPyC** (MIT, pure-Python, v0.11 2026, a pip dependency - the submodule was dropped in Round 17) - Shamir secret-sharing with secure
  comparison drives the **non-linear** layers (ReLU/argmax) on shares, the canonical CrypTen `ReLU([x]) =
  [x], [x>0]` pattern. Wired in `src/qsh/mpc/secret_inference.py` and run as three parties, each its own
  process over loopback TCP (`src/qsh/mpc/parties.py`, t = 1, k = 40); selectable as the `mpc` crypto-agility family
  (`mpyc-shamir` default; MP-SPDZ / hpmpc-gpu as heavyweight references). The linear layers stay under FHE - see
  the hybrid benchmark in [`docs/models-and-benchmarks.md`](models-and-benchmarks.md).

### Rust/WASM toolchain
Since Round 65 the demo's garbled-circuit tier is a WebAssembly module built from Rust:
`wasm/qsh-mpc/` (Polytune), loaded by `web/src/lib/clientMPC.ts` from `web/src/wasm/qsh_mpc/`. That
directory is generated, never committed, so a fresh checkout builds it once before `npm run dev`:

```bash
scripts/build_wasm.sh
```

It runs the Rust stage of `docker/Dockerfile.web` - the same one the demo image runs - so the host needs
only Docker. The stage uses the official `rust:1.99.1-bookworm` image pinned by digest, adds the
`wasm32-unknown-unknown` target, `wasm-pack` 0.15.0 and `wasm-bindgen-cli` 0.2.129, runs the crate's
`cargo test`, builds the module, and writes third-party notices for the 69 crates compiled into it to
`web/public/licenses/qsh-mpc.txt`, failing if any licence falls outside its permissive list.
`CARGO_JOBS` sets the compile's parallelism (2 by default, as on the shared demo host). The tests run in
every image build, the public demo's included.

A host build also works for crates whose minimum Rust version the host compiler meets:

```bash
sudo dnf install rust-std-static-wasm32-unknown-unknown
```

```bash
cargo install --locked wasm-pack
```

Polytune needs Rust 1.89 and this box's system Rust is 1.88, which is why the build uses the
container. The measurements that led to the module are in `prototypes/polytune-argmax/` (in one tab)
and `prototypes/net-bench/` (between two tabs over netem-shaped TCP).

### Dependency advisories
One script checks every locked dependency against the published advisories, and fails on any. Since
Round 67, `.github/workflows/audit.yml` runs it daily, on demand, and whenever a manifest, a lockfile
or the policy changes:

```bash
bash scripts/audit_deps.sh
```

- `web/`: `npm audit` at the lowest level, development tooling included.
- `wasm/qsh-mpc/`: cargo-deny 0.20.2, run in the same pinned Rust image as the build. Every RustSec
  advisory fails the check (vulnerability, unsound, unmaintained or yanked) unless
  `wasm/qsh-mpc/deny.toml` ignores it with a reason. Only crates.io and the pinned Polytune repository
  may supply crates. One advisory is ignored: RUSTSEC-2025-0141, which says bincode 2.0.1 (pulled by
  Polytune) is unmaintained. That advisory reports no vulnerability, and there is no safe upgrade.

`bash scripts/audit_deps.sh npm` runs only the npm check, and `bash scripts/audit_deps.sh rust` only the
Rust one. The Rust check needs Docker, and caches cargo-deny in `wasm/qsh-mpc/.cargo-home/`. GitHub
disables a public repository's scheduled workflows after 60 days without activity, so a long pause
needs the workflow re-enabled in the Actions tab.

### Languages
Python-first, plus **TypeScript** for the browser tier. **C++** is used only via the optional GPU-MPC
submodules (HEonGPU / hpmpc). Go was evaluated and not needed. **Rust** builds the demo's MPC module
(`wasm/qsh-mpc`, Polytune, MIT) and the measurement prototypes in `prototypes/`; the client-side Rust
decline recorded in the register concerned tfhe-rs (no FHE evaluation in its WASM build, and a licence
that withholds patent rights) and still stands.

**Julia: DECLINED 2026-08-12 (Round 33), and it should have been declined twenty-nine plans ago.** It was
listed as "optional (defer)" in the first plan and this line said "optional and deferred" ever since. The
word appears in exactly one of thirty plan files and in no instruction after the first, and **no condition
was ever attached to it** - so it could not even fail the three-round rule in
[`docs/deliverable-verification.md`](deliverable-verification.md), because an item with no condition
never becomes due. Two reasons, both checkable:

- **No in-browser path exists.** The browser tier is the deliverable, and Julia's WebAssembly target is
  still experimental in 2026: `ExportWebAssembly.jl` is marked work-in-progress, Julia's task runtime needs
  its stacks managed explicitly in the wasm heap rather than on the native WebAssembly stack, and only a
  few packages are built for the wasm platform through BinaryBuilder's experimental support. This is the
  same test that declined Rust for the client-side stack: not "is the language good" but "can the tab load
  it and evaluate".
- **There is no numerical gap for it to close.** The numerical work runs through numpy, torch and
  PennyLane, and the paths that dominate wall clock are already compiled native code - SEAL through
  Emscripten (`node-seal`), torch's kernels, Polytune compiled from Rust. Julia's argument is speed against interpreted
  Python, and the hot loops here are not interpreted Python.

**Reopening condition (both halves required, so it can actually fire):** a numerical kernel appears here
that is neither a SEAL/`node-seal` call nor a torch/numpy call and is measured to be a run's bottleneck in
pure Python, **and** Julia's wasm target stops being experimental - a Julia release listing wasm32 as a
supported platform, or `ExportWebAssembly.jl` out of WIP.

---

## OSS / submodules

Only the two submodules that code actually builds against are vendored:

```bash
QSH_SET=all bash scripts/bootstrap_submodules.sh  # HEonGPU + hpmpc (the GPU MPC-FHE benchmark path)
# The bare command defaults to QSH_SET=min and both entries are tagged `all`, so it adds nothing.
```

Round 17 removed ten others (CQC-QHE, AMLSim, shark, mpyc, oqs-provider, openfhe, SEAL, liboqs,
MP-SPDZ, synthea). None was referenced by any import, script or Dockerfile; five were superseded by
the pip/npm package of the same project; and `.dockerignore` excludes `submodules/` wholesale, so
none of them ever reached an image. Cloning ~10 GB of C++ that nothing links against is a cost with
no matching capability. They remain **cited as upstream work** in
[`references/REFERENCES.md`](../references/REFERENCES.md), which is the accurate relationship - using a
library through its published package does not require vendoring its source tree.

| Dependency | How it is used | How it is obtained | License |
|---|---|---|---|
| `liboqs-python` | PQC ML-KEM / ML-DSA (SLH-DSA is not in this liboqs build, and no code here uses it) | pip | MIT |
| `tenseal` | CKKS (default FHE backend) | pip | Apache-2.0 (over Microsoft SEAL, MIT) |
| `openfhe` | CKKS + approximate bootstrapping | pip; on RHEL 9 build with [`env/build_openfhe_el9.sh`](../env/build_openfhe_el9.sh) | BSD-2-Clause |
| `mpyc` | **non-linear MPC** - Shamir ReLU/argmax, the hybrid's non-linear half | pip | MIT |
| `pennylane` | variational quantum layers (QFL, QML-IDS) | pip | Apache-2.0 |
| `qiskit` / `qiskit-aer` | circuit authoring + state-vector simulation | pip | Apache-2.0 |
| **Alisah-Ozcan/HEonGPU** | GPU CKKS for the MPC-FHE benchmark | **submodule** | Apache-2.0 |
| **chart21/hpmpc** (PIGEON) | GPU secret-sharing MPC for the same benchmark | **submodule** | MIT |

**Client-side / AI OSS (npm, all commercial-OK):** `@noble/post-quantum` **0.7.0** (MIT) for
in-browser ML-KEM and ML-DSA. The pin matters: **0.2.1 accepted all 2,595 malformed encapsulation
keys** in the C2SP/CCTV modulus set, because its polynomial coder took the compression path for
d=12 where FIPS 203 specifies plain ByteEncode12/ByteDecode12, making the section 7.2 modulus check
a no-op. 0.6.1 and 0.7.0 reject all 2,595, and `npm run check:pqc` asserts it - now with a
keygen/encapsulate/decapsulate round trip and the FIPS 203 Table 3 ciphertext lengths on the same
line, because the rejection count alone reads identically for a working library and a totally broken
one.

**Moved from 0.6.1 to 0.7.0 in Round 56, and what that release is NOT.** `^0.6.1` does not admit
0.7.0 - a caret on a 0.x version means `>=0.6.1 <0.7.0` - so this was a deliberate bump, verified by
running `check:pqc` unmodified against 0.7.0: identical rejection counts, identical key/ciphertext
/signature sizes, exit 0. The breaking changes do not reach this project (the removed hybrid aliases
and `utils.afunction` appear zero times here) and the argument order is unchanged. **Its
`MultiplyNTTs` note is a PERFORMANCE fix, not a security one** - 0.7.0's own comment says the wide
intermediate "forc[es] JS engines into slow float fmod", and 0.6.1 reduces before truncating, so
0.6.1 is not numerically wrong. Recorded because that changelog line reads like a vulnerability and
would otherwise be transcribed as one. What 0.7.0 genuinely improves for this repo's usage: ML-DSA
draws randomness before decoding the secret key, and ML-KEM detaches the caller's public key before
caching. Note the honest scope: noble
is self-audited and carries no side-channel protection, so this is conformance evidence, not
production assurance. Also `node-seal` 7.0.0, which bundles **Microsoft
SEAL 4.1.2**. Verified 2026-10-30: node-seal 7.0.0 (2025-11-08) is still the newest node-seal on npm,
so this is not a stale pin here - but the gap to upstream SEAL is wide and widening: Microsoft SEAL
is at **4.7.2 (2026-10-28)**. `CHANGES.md`
calls **4.4.0 a "Critical security update: all users should upgrade ... as soon as possible"**, which
hardened untrusted deserialization and fixed native memory-safety and concurrency defects across
ciphertext evaluation; 4.4.3 fixes null-output-pointer dereferences and C++ exceptions leaking across
the **C API** boundary (issues #753-#755); 4.4.4 adds input validation to several evaluator
functions; and 4.7.2 fixes an off-by-one in the rejection sampling of `util::sample_poly_uniform`,
which made its output very slightly non-uniform. The C-API items do not reach a WASM build, but
deserialization, ciphertext evaluation and sampling are core C++, so 4.4.0's two most serious items
and 4.7.2's sampling fix **do**. Full
analysis and what it means for this demo: [`docs/deliverable-verification.md`](deliverable-verification.md)
and [`docs/threat-model.md`](threat-model.md). **No CVE and no GHSA advisory
exists for SEAL**, so the version is recorded as a fact and no vulnerability is claimed
(SEAL -> WASM CKKS, MIT) +
`polytune` with `garble_lang` (sine-fdn, garbled-circuit 2PC, MIT; built from `wasm/qsh-mpc`, its
third-party notices served at `/licenses/qsh-mpc.txt`) for the demo's real in-browser FHE/MPC; `onnxruntime-web` (MIT) for the published industry models; `@huggingface/transformers` +
`@mlc-ai/web-llm` (Apache-2.0) for in-browser AI; `mcp` Python SDK (MIT) + the official
`@modelcontextprotocol/server-filesystem` (MIT) for the real MCP server (full deployment). Full table
+ dataset licenses in [`DATA_LICENSES.md`](../DATA_LICENSES.md).

**Originality note:** the GPU MPC-FHE benchmark is our own integration over the *upstream*
projects (`src/qsh/benchmark/gpu_mpc_adapter.py` + `scripts/build_gpu_mpc.sh`). We do **not**
clone or vendor any mixed-authorship aggregate repo.

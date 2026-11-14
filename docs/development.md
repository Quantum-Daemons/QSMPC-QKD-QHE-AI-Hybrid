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

GPU quantum simulation (Blackwell): see **Dev environment** below. CUDA-Q, PennyLane `lightning.gpu` and
Qiskit Aer GPU 0.18.2 all run on it; `quantum_sim.device` chooses the Python tiers' simulator.

---

## Dev environment

Target box: **AlmaLinux 9.8, i5-13600K, 128 GB DDR5, NVIDIA RTX PRO 6000 Blackwell 96 GB**. Read
2026-11-05: kernel 5.14.0-687.65.1, Python 3.12.15, gcc 11.5, cmake 3.31, Node 22.23, Docker 29.9,
`torch 2.13.0+cu130` with `cuda.is_available() == True`, nvcc 13.5.37, driver 620.70.06
(CUDA 13 wheels require 580.65.06 or newer on Linux). A GPU figure elsewhere in this repository dated before
then ran on AlmaLinux 9.7, nvcc 13.0.88 and driver 580.105.08. CUDA 13.0 became a **stable** PyTorch build target in 2.11, so this
box is no longer on a leading edge - the capability is verified by launching a real kernel and
checking that `get_arch_list()` contains the device's `sm_120`, rather than by trusting a version
string here.

### NOTE GPU quantum simulation on Blackwell
Measured on this machine on 2026-11-06 ([State-vector simulators, measured](models-and-benchmarks.md#state-vector-simulators-measured)):
- **CUDA-Q** (`cuda-quantum-cu13` 0.16.0.post1) and **PennyLane `lightning.gpu`** 0.45.0 (cuQuantum cuStateVec, on its bundled CUDA 12
  runtime) both run on `sm_120`, and so does **Qiskit Aer GPU** 0.18.2 (2026-10-26, CUDA 12 runtime). Earlier records said Aer's wheels did
  not target `sm_120` and that it had not been released since 2024; both were true of what was available then.
- `quantum_sim.device` (`qsh.quantum_device`) chooses the Python tiers' simulator: `default.qubit` by default, `lightning.gpu` for a large
  experiment. `tests/test_r86_quantum_device.py` holds every simulator to the same outputs.
- CUDA-Q 0.16.0.post1 needs `custatevec-cu13` 1.14 and `cuquantum-python-cu13` 26.9.0 needs 1.15 or later, so the two cannot share one
  environment. See [`requirements-gpu.txt`](../requirements-gpu.txt); `requirements-gpu.lock` pins the part this machine trains and simulates with, by hash.

### FHE backends (commercial-OK only)
- **TenSEAL** (Apache-2.0, CPU CKKS via Microsoft SEAL) - the always-available default (`TenSEALBackend`).
- **OpenFHE** (BSD-2-Clause, CPU CKKS **with approximate bootstrapping**) - the live
  `OpenFHEBackend` via the `openfhe` wheel, `openfhe>=1.6.1,<1.7` since Round 86 (never 1.6.0, whose issue
  #1364 is this repository's bootstrap path; the 1.6 bindings moved `GetBootstrapDepth` from `FHECKKSRNS` to
  `CryptoContext`, and the bootstrap path calls it there). The
  wheel needs a recent libstdc++ (Ubuntu 24.04 / the Debian-trixie Docker image); on RHEL 9 build the
  bindings with `env/build_openfhe_el9.sh`, and on an older host the import fails with a clear
  `ImportError`.
- **HEonGPU** (Apache-2.0, GPU CKKS/BFV + collective bootstrapping) - the GPU `HeongpuBackend` since
  Round 86. HEonGPU has no Python API, so this project binds the seven operations the hybrid uses
  (`native/heongpu_py`), and `bash scripts/build_heongpu.sh` builds the pinned v2.1.1 with that
  module for the GPU it finds (CUDA 12.8 or later, CMake 4, GMP's headers; it fetches NTL and checks
  its hash). The script fails if the module carries no code for that GPU's architecture: the first
  build carried only Turing's, which the driver recompiled at load. The backend runs where the module
  is built and a GPU is present, and only there is `heongpu-gpu` an `fhe` option. CI has neither, so
  `tests/test_r86_heongpu_backend.py` skips there. What it measured is in
  [the benchmark document](models-and-benchmarks.md#ckks-on-each-backend-measured).

> **DESILO `desilofhe` was removed.** Its license is **non-commercial** - commercial use requires a separate
> paid license ([fhe.desilo.dev/latest/license](https://fhe.desilo.dev/latest/license/)). To keep the stack
> unambiguously commercial-OK, the FHE layer uses TenSEAL + OpenFHE, and HEonGPU on a GPU where it is built.

### Non-linear MPC backend (the hybrid's other half)
- **MPyC** (MIT, pure-Python, v0.11 2026, a pip dependency - the submodule was dropped in Round 17) - Shamir secret-sharing with secure
  comparison drives the **non-linear** layers (ReLU/argmax) on shares, the `ReLU([x]) =
  [x] * [x>0]` pattern. Wired in `src/qsh/mpc/secret_inference.py` and run as three parties, each its own
  process over loopback TCP (`src/qsh/mpc/parties.py`, t = 1, k = 40); the `mpc` crypto-agility family's one
  option, `mpyc-shamir` (MP-SPDZ and hpmpc were names in it that nothing ran, until Round 86). The linear layers stay under FHE - see
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

Since Round 86 the same script also runs the image's Go stage, which builds the threshold-aggregation module
`wasm/qsh-fedthresh` (Lattigo v6.2.0, Apache-2.0) for the medical page's encrypted-aggregation comparison. The stage
uses the official `golang:1.27.4-bookworm` image pinned by digest, verifies the module checksums (`go mod verify`),
runs `go vet` and the module's `go test` (threshold BGV exact, each CKKS setting within its bound, every setting clear
of the Gaussian-sampler defect reported as tuneinsight/lattigo#596), builds it with `GOOS=js GOARCH=wasm`, copies
Go's `wasm_exec.js` beside it into `web/public/wasm/qsh_fedthresh/`, and writes the notices for the ten Go modules
and Go itself compiled into it to `web/public/licenses/qsh-fedthresh.txt`, failing on a licence outside Apache-2.0,
BSD, ISC or MIT. Both outputs are gitignored. `npm run check:fed-agg` then runs the comparison in Node against them.

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
Python-first, plus **TypeScript** for the browser tier. **C++ and CUDA** only for the GPU: the
HEonGPU binding (`native/heongpu_py`) and the submodules it and hpmpc build from. Go was evaluated and not needed. **Rust** builds the demo's MPC module
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


## The lending page's checks

`npm run check:lending` holds the market to closed forms, Aave's own worked example, its knots and its books. `npm run check:lending-private` runs the private check on the real node-seal and Polytune against the same check in the clear: every checkpoint's bit, positions built one stable unit either side of zero, the masks, the circuit's size, a tampered, replayed or wrongly signed submission, and the worker pool's own path through a stub `Worker`. Both run in CI and in `scripts/ci_local.sh`.

## Bills of materials

**SBOMs.** The `sbom` job in `.github/workflows/ci.yml` runs `scripts/generate_sbom.sh` on every push and keeps the output as a 90-day workflow artefact: CycloneDX 1.7 documents for `requirements.lock` (cyclonedx-py), `web/package-lock.json` without development dependencies (cyclonedx-npm) `wasm/qsh-mpc/Cargo.lock` (cargo-cyclonedx), and, since Round 86, the threshold-aggregation module `wasm/qsh-fedthresh`: the Go modules `go list` compiles into its wasm and Go's standard library, written by its own `sbom.py` with the licences read from each module's licence files, the same reading its `notices.py` makes for the notices it ships. The three tools are Apache-2.0. No container image is built in CI, so there is no SBOM of an image.

**CBOM.** `web/public/cbom/qsh.cdx.json` is a CycloneDX 1.7 list of the cryptography this repository's code calls, generated by `scripts/generate_cbom.py` from `src/qsh/crypto_inventory.py` and rendered on the Validations page. It is hand-written and checked, not scanned: `tests/test_r86_cbom.py` requires each entry's evidence to be in the source and fails on a cryptographic import no entry accounts for. A scanner was not used because CBOMkit's sonar-cryptography carries SonarSource's source-available licence. Neither kind of document is a claim of conformance to anything.

`scripts/validate_cyclonedx.py` checks all of them against the schema vendored in `schemas/cyclonedx/` (tag 1.7.2, Apache-2.0).

## OSS / submodules

**The maintenance rule, in one place** (owner's choice, 2026-11-05): a dependency's project must have moved within the last
**12 months**, and an advisory with no fix fails a dependency at any age. The window is wide so that a stable cryptographic
library with few releases is not refused for being finished. Commercial use must be allowed.

**One recorded exception (the owner's choice, 2026-11-08).** `RUSTSEC-2025-0141`: bincode 2.0.1, reached through Polytune, is
unmaintained. No vulnerability is reported against it and no fixed version exists, so `wasm/qsh-mpc/deny.toml` ignores that one id. Reopens if a
vulnerability is reported against bincode 2.x, or if Polytune changes the dependency; `.github/workflows/audit.yml` runs `cargo deny` on a schedule.

**The GPU tier's licences and the published timings (read 2026-11-08; a reading of the texts, not legal advice).** The cuQuantum SDK's "NVIDIA SOFTWARE LICENSE (version
August 18, 2021)", the CUDA Toolkit EULA and cuTensorNet's shipped LICENSE contain no clause on benchmarks or on publishing results, and CUDA-Q is Apache-2.0, so
the timings in `models-and-benchmarks.md` stand. A different NVIDIA agreement, the enterprise-software license, does bar disclosing benchmark results; it is not the licence these
libraries carry. Re-read on an upgrade: https://docs.nvidia.com/cuda/cuquantum/latest/license.html and https://docs.nvidia.com/cuda/eula/index.html.

Only the two submodules that code actually builds against are vendored:

```bash
QSH_SET=all bash scripts/bootstrap_submodules.sh  # HEonGPU (the GPU CKKS backend) + hpmpc
# The bare command defaults to QSH_SET=min and both entries are tagged `all`, so it adds nothing.
```

Round 17 removed ten others (CQC-QHE, AMLSim, shark, mpyc, oqs-provider, openfhe, SEAL, liboqs,
MP-SPDZ, synthea). None was referenced by any import, script or Dockerfile; five were superseded by
the pip/npm package of the same project; and `.dockerignore` excludes `submodules/` wholesale, so
none of them ever reached an image. Cloning ~10 GB of C++ that nothing links against is a cost with
no matching capability. They remain **cited as upstream work** in
[`references/REFERENCES.md`](../references/REFERENCES.md), which is the accurate relationship - using a
library through its published package does not require vendoring its source tree.

| Dependency | How it is used | How it is obtained | License | Locked, and its release |
|---|---|---|---|---|
| `liboqs-python` | PQC ML-KEM / ML-DSA. liboqs 0.16.0, in the Docker image and on the development machine (built from the same tag with the same flags, 2026-11-09), also builds SLH-DSA (in liboqs since 0.15.0) and enables HQC by default. No code here registers SLH-DSA or HQC | pip; the C library built from the 0.16.0 tag in `docker/Dockerfile.api` | MIT | 0.16.0.2, 2026-10-28 |
| `pysodium` | libsodium's ristretto255 for the full deployment's RFC 9497 OPRF PSI (`qsmpc/oprf.py`, Round 85); without the system library the OPRF falls back to its pure-Python code, which is not constant-time, and the run says so (`psi_oprf_backend`) | pip, over the system libsodium 1.0.18 (`libsodium23` in `docker/Dockerfile.api` and CI; EL9 `libsodium`) | BSD-2-Clause (libsodium: ISC) | 0.7.19, 2026-10-17 |
| `tenseal` | CKKS (default FHE backend). It bundles its own SEAL: 4.1.1 in 0.3.16, the version the venv runs, and 4.3.3 from 0.3.17 on, which the `>=0.3.15` floor admits; both predate SEAL 4.4.0 | pip | Apache-2.0 (over Microsoft SEAL, MIT) | 0.3.18, 2026-09-12 |
| `openfhe` | CKKS + approximate bootstrapping | pip; on RHEL 9 build with [`env/build_openfhe_el9.sh`](../env/build_openfhe_el9.sh) | BSD-2-Clause | 1.6.1.0.24.4, 2026-10-27 |
| `mpyc` | **non-linear MPC** - Shamir ReLU/argmax, the hybrid's non-linear half | pip | MIT | 0.11, 2026-02-09 |
| `pennylane` | variational quantum layers (QFL, QML-IDS) | pip | Apache-2.0 | 0.45.1, 2026-06-26 |
| `qiskit` / `qiskit-aer` | circuit authoring + state-vector simulation | pip | Apache-2.0 | 2.6.2, 2026-11-05 / 0.18.2, 2026-10-26 |

The last column is the version `requirements.lock` pins and the date PyPI gives for that release, read 2026-11-10; each is the newest on PyPI that day, so every row
has a release inside the last year, which checklist rows TR5 and G3 ask. A development venv can lag the lock: this machine's ran TenSEAL 0.3.16, Qiskit 2.5.1 and Qiskit Aer
0.17.2 that day, while CI and the Docker image install from the lock. TenSEAL 0.3.18's notes name no SEAL bump after 0.3.17's move to 4.3.3.

HEonGPU builds the GPU CKKS backend and hpmpc is vendored for a GPU MPC run not yet made; their pins,
licences, what the HEonGPU build links and the unlicensed submodule one level below hpmpc are in
[`DATA_LICENSES.md`](../DATA_LICENSES.md#6a-git-submodules).

**Client-side / AI OSS (npm, all commercial-OK):** `@noble/post-quantum` **0.7.1** (MIT) for
in-browser ML-KEM and ML-DSA, with `@noble/curves`, `@noble/hashes` and `@noble/ciphers` at 2.4.0, the
exact versions 0.7.1 depends on (Round 85: moved together, so one copy of each is bundled). The pin
matters: **0.2.1 accepted all 2,595 malformed encapsulation keys** in the C2SP/CCTV modulus set,
because its polynomial coder took the compression path for
d=12 where FIPS 203 specifies plain ByteEncode12/ByteDecode12, making the section 7.2 modulus check
a no-op. 0.6.1, 0.7.0 and 0.7.1 reject all 2,595, and `npm run check:pqc` asserts it - now with a
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
is at **4.7.2 (2026-10-28)**. node-seal's main branch merged SEAL 4.7.2 on 2026-10-31 (PR #168; its
`CHANGES.md` names the release 7.1.0, which also rejects NaN and infinities at CKKS encode and
renames `jsArrayFromVecModulus` and `correctionFactor`, neither of which `clientFHE.ts` calls); npm
still served 7.0.0 on 2026-11-02. `bash scripts/audit_deps.sh seal`, which the audit workflow runs
daily, fails once npm's newest node-seal is not the one locked, and the upgrade is owed then: move
`web/`, then rerun `tests/test_r84_medtel_ckks_decrypts_only_outputs.py` and
`tests/test_r86_private_inference_review.py` (Round 86). One check belongs to the upgrade itself
(review PI-14): that the masked slots' imaginary parts carry double-precision round-off only, which
is what lets `clientFHE.ts` flood the real parts alone. It has been shown on TenSEAL's SEAL, whose
encoder node-seal shares, and not on node-seal's build, whose node harness decodes real parts only.
A real slot vector encodes to coefficients with $`m_i = -m_{N-i}`$, so the check can be made on the
encoded flood without trusting the decoder. `CHANGES.md`
calls **4.4.0 a "Critical security update: all users should upgrade ... as soon as possible"**, which
hardened untrusted deserialization and fixed native memory-safety and concurrency defects across
ciphertext evaluation; 4.4.3 fixes null-output-pointer dereferences and C++ exceptions leaking across
the **C API** boundary (issues #753-#755); 4.4.4 adds input validation to several evaluator
functions; and 4.7.2 fixes an off-by-one in the rejection sampling of `util::sample_poly_uniform`,
which made its output very slightly non-uniform. Read in SEAL's source on 2026-11-14: v4.1.2, the version node-seal 7.0.0
bundles, bounds the draw at `max_random - barrett_reduce_64(max_random, modulus) - 1`, and v4.7.2 drops the `- 1`, so
each draw rejects one 64-bit value more than it needs to; by this project's reading one residue loses one of its
preimages, a statistical distance from uniform of about $`2^{-64}`$ per coefficient, far below any security margin. The C-API items do not reach a WASM build, but
deserialization, ciphertext evaluation and sampling are core C++, so 4.4.0's two most serious items
and 4.7.2's sampling fix **do**. Full
analysis and what it means for this demo: [`docs/deliverable-verification.md`](deliverable-verification.md)
and [`docs/threat-model.md`](threat-model.md). **No CVE and no GHSA advisory
exists for SEAL**, so the version is recorded as a fact and no vulnerability is claimed
(SEAL -> WASM CKKS, MIT) +
`polytune` with `garble_lang` (sine-fdn, garbled-circuit 2PC, MIT; built from `wasm/qsh-mpc`, its
third-party notices served at `/licenses/qsh-mpc.txt`) for the demo's real in-browser FHE/MPC; Lattigo v6.2.0 (Tune Insight, Apache-2.0; built from `wasm/qsh-fedthresh`, its notices served at `/licenses/qsh-fedthresh.txt`) for threshold CKKS and BGV in the medical page's aggregation comparison; `onnxruntime-web` (MIT) for the published industry models (the notices of every npm package in the page's JavaScript, the workers' included, are written at build time to `/licenses/js-bundle.txt` and linked from the sidebar); `@huggingface/transformers` +
`@mlc-ai/web-llm` (Apache-2.0) for in-browser AI; `mcp` Python SDK (MIT) + the official
`@modelcontextprotocol/server-filesystem` (MIT and Apache-2.0, moving from one to the other) for the real MCP server (full deployment). Full table
+ dataset licenses in [`DATA_LICENSES.md`](../DATA_LICENSES.md).

**Originality note:** the GPU CKKS backend is this project's binding over the *upstream* HEonGPU
(`native/heongpu_py`, `scripts/build_heongpu.sh`, `qsh.fhe.backends.HeongpuBackend`).

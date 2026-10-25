#!/bin/sh
# Builds the garbled-circuit tier's WebAssembly for local development and CI by running the SAME Rust
# stage docker/Dockerfile.web runs, so the module a developer or CI tests is the one the demo ships:
#
#   web/src/wasm/qsh_mpc/            the wasm-pack package web/src/lib/clientMPC.ts imports
#   web/public/licenses/qsh-mpc.txt  its third-party notices, served with the demo
#
# That stage also runs the crate's `cargo test`, so this fails if the argmax no longer matches the
# plaintext one. Needs Docker with BuildKit; installs nothing on the host. CARGO_JOBS (default 2, as on
# the shared demo host) sets the compile's parallelism. Both outputs are gitignored.
set -eu
cd "$(dirname "$0")/.."
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
docker build -f docker/Dockerfile.web --target wasm-out \
  --build-arg CARGO_JOBS="${CARGO_JOBS:-2}" --output "type=local,dest=$out" .
rm -rf web/src/wasm/qsh_mpc
mkdir -p web/src/wasm web/public/licenses
cp -r "$out/qsh_mpc" web/src/wasm/qsh_mpc
cp "$out/qsh-mpc-notices.txt" web/public/licenses/qsh-mpc.txt
echo "wasm: web/src/wasm/qsh_mpc ($(du -sh web/src/wasm/qsh_mpc | cut -f1)); notices: web/public/licenses/qsh-mpc.txt ($(grep -c '^== ' web/public/licenses/qsh-mpc.txt) crates)"

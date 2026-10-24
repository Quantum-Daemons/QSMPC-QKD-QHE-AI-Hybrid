#!/bin/sh
# Builds and tests the prototype inside the official Rust container. Needs Docker; installs nothing on
# the host. Runs as the calling user, so target/ and pkg/ are not root-owned, and keeps Cargo's
# download cache in .cargo-home/ here (gitignored) so a rebuild does not fetch the crates again.
#
#   ./build.sh          native test, then the wasm package in pkg/
#   ./build.sh test     native test only
set -eu
cd "$(dirname "$0")"
IMAGE=qsh-polytune-builder:rust-1.99.1
docker build -q -t "$IMAGE" .
run() {
  docker run --rm -u "$(id -u):$(id -g)" \
    -v "$PWD":/work -w /work \
    -e CARGO_HOME=/work/.cargo-home -e HOME=/tmp \
    -e RUSTFLAGS='--cfg getrandom_backend="wasm_js"' \
    "$IMAGE" sh -c "$1"
}
# The native test does not need the getrandom wasm flag, and must not receive it.
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/work -w /work \
  -e CARGO_HOME=/work/.cargo-home -e HOME=/tmp "$IMAGE" \
  cargo test --release -- --nocapture
[ "${1:-}" = "test" ] && exit 0
run "wasm-pack build --target web --release"
ls -l pkg/*.wasm

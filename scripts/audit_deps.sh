#!/bin/sh
# Checks every locked dependency against the published advisories, and fails on any. The same script
# runs locally and in .github/workflows/audit.yml (daily, and whenever a manifest, a lockfile or the
# policy changes), so the two cannot drift apart.
#
#   bash scripts/audit_deps.sh          both checks
#   bash scripts/audit_deps.sh npm      web/: npm audit, development tooling included
#   bash scripts/audit_deps.sh rust     wasm/qsh-mpc: cargo-deny, advisories and sources
#
# npm audit needs Node and network access. The Rust check needs Docker and network access, and
# installs nothing on the host: it runs in the same digest-pinned Rust image as the web image's wasm
# stage, as the calling user, with cargo-deny and its advisory database cached in
# wasm/qsh-mpc/.cargo-home (gitignored, kept out of the Docker context).
#
# AUDIT_WEB_DIR points the npm check at another project, which is how its failure path is checked.
set -eu
cd "$(dirname "$0")/.."

RUST_IMAGE=rust:1.99.1-bookworm@sha256:48b7f435b8fc6378ce43b44f15e3a47fd8919b2825f835c39bd4bd3da50af5b8
CARGO_DENY=0.20.2

audit_npm() {
  # No --omit=dev: the dev server and the build tooling run on developer machines and in CI, and
  # Round 59 counted them on purpose. info is the lowest level, so any advisory fails.
  (cd "${AUDIT_WEB_DIR:-web}" && npm audit --audit-level=info)
}

audit_rust() {
  # Policy in wasm/qsh-mpc/deny.toml: every advisory fails (vulnerability, unsound, unmaintained,
  # yanked) unless it is ignored there with its reason, and only crates.io and the pinned Polytune
  # repository may supply crates.
  mkdir -p wasm/qsh-mpc/.cargo-home
  docker run --rm -u "$(id -u):$(id -g)" \
    -v "$PWD/wasm/qsh-mpc":/work -w /work \
    -e CARGO_HOME=/work/.cargo-home -e HOME=/tmp -e CARGO_BUILD_JOBS="${CARGO_JOBS:-2}" \
    "$RUST_IMAGE" sh -c "
      cargo install --locked --quiet cargo-deny@$CARGO_DENY &&
      /work/.cargo-home/bin/cargo-deny --color never check advisories sources"
}

case "${1:-all}" in
  npm) audit_npm ;;
  rust) audit_rust ;;
  all) audit_npm && audit_rust ;;
  *) echo "usage: $0 [npm|rust]" >&2; exit 2 ;;
esac

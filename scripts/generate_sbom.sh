#!/bin/sh
# Software bills of materials for the three dependency trees this repository ships, in CycloneDX 1.7 (Round 86):
#
#   python.cdx.json   requirements.lock, the full deployment's pinned set
#   web.cdx.json      web/package-lock.json without development dependencies, what the browser bundle can contain
#   wasm.cdx.json     wasm/qsh-mpc/Cargo.lock, the garbled-circuit module
#
# Needs cyclonedx-py (cyclonedx-bom 8.0.1), cargo-cyclonedx 0.5.10 and npx. CI installs them; all three are Apache-2.0.
# Each file is validated against the vendored schema by scripts/validate_cyclonedx.py afterwards. The output directory is
# the first argument, sbom-out by default, and is gitignored.
set -eu
cd "$(dirname "$0")/.."
out=$(mkdir -p "${1:-sbom-out}" && cd "${1:-sbom-out}" && pwd)

cyclonedx-py requirements --sv 1.7 --output-reproducible --of JSON -o "$out/python.cdx.json" requirements.lock
( cd web && npx --yes @cyclonedx/cyclonedx-npm@6.0.1 --sv 1.7 --omit dev --output-reproducible --package-lock-only -o "$out/web.cdx.json" )
( cd wasm/qsh-mpc && SOURCE_DATE_EPOCH=0 cargo cyclonedx --format json --spec-version 1.7 --override-filename qsh-mpc && mv qsh-mpc.json "$out/wasm.cdx.json" )
echo "wrote $out/python.cdx.json, web.cdx.json and wasm.cdx.json"

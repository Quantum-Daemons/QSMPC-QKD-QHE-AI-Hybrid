#!/bin/sh
# Software bills of materials for the three dependency trees this repository ships, in CycloneDX 1.7 (Round 86):
#
#   python.cdx.json   requirements.lock, the full deployment's pinned set
#   web.cdx.json      web/package-lock.json without development dependencies, what the browser bundle can contain
#   wasm.cdx.json     wasm/qsh-mpc/Cargo.lock, the garbled-circuit module
#   fedthresh.cdx.json  wasm/qsh-fedthresh, the threshold-aggregation module: the Go modules compiled into it and Go's
#                     standard library (Round 86), written by its own sbom.py from `go list`
#
# Needs cyclonedx-py (cyclonedx-bom 9.0.0, with cyclonedx-python-lib 11.12), cargo-cyclonedx 0.5.10, npx and Go. CI installs them; the first three are
# Apache-2.0, Go is BSD-3-Clause.
# Each file is validated against the vendored schema by scripts/validate_cyclonedx.py afterwards. The output directory is
# the first argument, sbom-out by default, and is gitignored.
set -eu
cd "$(dirname "$0")/.."
out=$(mkdir -p "${1:-sbom-out}" && cd "${1:-sbom-out}" && pwd)

cyclonedx-py requirements --sv 1.7 --output-reproducible --of JSON -o "$out/python.cdx.json" requirements.lock
( cd web && npx --yes @cyclonedx/cyclonedx-npm@6.0.1 --sv 1.7 --omit dev --output-reproducible --package-lock-only -o "$out/web.cdx.json" )
( cd wasm/qsh-mpc && SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)" cargo cyclonedx --format json --spec-version 1.7 --override-filename qsh-mpc && mv qsh-mpc.json "$out/wasm.cdx.json" )
( cd wasm/qsh-fedthresh && go mod download && python3 sbom.py "$out/fedthresh.cdx.json" )
# The metadata the generators leave out, reproducibly: the commit's time, a content-derived serial number, the supplier,
# the lifecycle phase, the Python SBOM's primary component; the web SBOM's component hashes from web/package-lock.json;
# and, when QSH_PYTHON_ENVIRONMENT_SBOM names cyclonedx-py's environment-mode SBOM of an interpreter where
# requirements.lock is installed (CI's python job writes one), the Python SBOM's licences (scripts/sbom_metadata.py,
# Round 86). In GitHub Actions it also writes a line per SBOM to the job summary.
env_sbom=${QSH_PYTHON_ENVIRONMENT_SBOM:-}
python3 scripts/sbom_metadata.py ${env_sbom:+--python-environment "$env_sbom"} ${GITHUB_STEP_SUMMARY:+--summary "$GITHUB_STEP_SUMMARY"} \
  "$out/python.cdx.json" "$out/web.cdx.json" "$out/wasm.cdx.json" "$out/fedthresh.cdx.json"
echo "wrote $out/python.cdx.json, web.cdx.json, wasm.cdx.json and fedthresh.cdx.json"

#!/usr/bin/env bash
# Known vulnerabilities in every lockfile this repository tracks, from OSV-Scanner's offline databases (Round 86, the
# owner's choice of 2026-11-23). CI downloads the databases each run (job `osv` in .github/workflows/ci.yml, with
# --download-offline-databases); scripts/ci_local.sh runs the same scan on a local copy. Exceptions, each with its
# reason and a date to look again, are in osv-scanner.toml. tests/test_r86_osv_scan_covers_every_lockfile.py fails when
# a tracked lockfile is missing from the list below.
#
# Usage: bash scripts/scan_lockfiles.sh <osv-scanner binary> [extra osv-scanner flags]
set -euo pipefail
scanner="$1"
shift
exec "$scanner" scan source --offline-vulnerabilities --config osv-scanner.toml "$@" \
  -L requirements.txt:requirements.lock \
  -L requirements.txt:requirements-gpu.lock \
  -L requirements.txt:walkthroughs/requirements.lock \
  -L web/package-lock.json \
  -L docker/mcp-fs/package-lock.json \
  -L prototypes/net-bench/bench/package-lock.json \
  -L wasm/qsh-mpc/Cargo.lock \
  -L prototypes/polytune-argmax/Cargo.lock \
  -L wasm/qsh-fedthresh/go.mod

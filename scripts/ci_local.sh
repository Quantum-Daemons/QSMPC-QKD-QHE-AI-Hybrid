#!/usr/bin/env bash
# Run every BLOCKING gate .github/workflows/ci.yml runs, locally, in CI's own order.
#
# Why this exists: for a whole round the local gate was `pytest -m "not slow"` + `tsc --noEmit` +
# `npm run build`, all three green, while CI was RED on every push. The failure was a single ruff
# E702 in a file the blocking lint covers. Local-green and CI-green are different claims, and the
# difference is silent - `git push` succeeds either way, and the demo deploys fine from a red commit
# because the VPS pulls the branch rather than a CI artifact.
#
# Usage:  bash scripts/ci_local.sh
# Exit 0 means every blocking CI gate would pass on this tree.

set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
fail=0
step() { printf '\n=== %s ===\n' "$1"; }
ok()   { printf '  PASS  %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n' "$1"; fail=1; }

step "1/7 test suite (pytest -m 'not slow')"
HF_HUB_OFFLINE=1 "$PY" -m pytest -m "not slow" -q -p no:warnings >/tmp/ci_pytest.log 2>&1 \
  && ok "tests" || { bad "tests"; tail -20 /tmp/ci_pytest.log; }

step "2/7 config parity (defaults.py -> config.defaults.json)"
"$PY" scripts/export_config.py >/dev/null 2>&1
git diff --quiet -- web/src/lib/config.defaults.json \
  && ok "config.defaults.json in sync" \
  || bad "config.defaults.json is stale - commit the regenerated file"

step "3/7 lint, BLOCKING scope (this is the one that was red)"
"$PY" -m ruff check src/qsh/models scripts/train scripts/publish_hf.py scripts/data \
  && ok "ruff blocking scope" || bad "ruff blocking scope"

step "4/7 no confidential files tracked"
make check-secrets >/dev/null 2>&1 && ok "no confidential files" || bad "confidential files tracked"

step "5/7 no Japanese in public files"
if grep -rIlP '[\x{3040}-\x{309F}\x{30A0}-\x{30FF}\x{4E00}-\x{9FFF}]' \
     --include='*.py' --include='*.ts' --include='*.tsx' --include='*.md' \
     --include='*.yml' --include='*.sh' \
     src api web/src tests scripts docs README.md 2>/dev/null; then
  bad "Japanese characters in public files (listed above)"
else
  ok "no Japanese in public files"
fi

step "6/7 no model artifacts committed"
if git ls-files | grep -E '\.(onnx|safetensors|pt|pth|h5|npy)$'; then
  bad "model artifacts committed - publish to the Hub instead"
else
  ok "no model artifacts committed"
fi

step "7/7 web typecheck + build"
( cd web && npx tsc --noEmit >/tmp/ci_tsc.log 2>&1 ) \
  && ok "tsc --noEmit" || { bad "tsc --noEmit"; tail -15 /tmp/ci_tsc.log; }
( cd web && npm run build >/tmp/ci_build.log 2>&1 ) \
  && ok "npm run build" || { bad "npm run build"; tail -15 /tmp/ci_build.log; }

printf '\n=== advisory (not blocking in CI, printed so it is driven down deliberately) ===\n'
"$PY" -m ruff check src tests api scripts --output-format=concise 2>/dev/null | tail -3

if [ "$fail" -eq 0 ]; then
  printf '\nALL BLOCKING CI GATES PASS\n'; exit 0
else
  printf '\nAT LEAST ONE BLOCKING CI GATE FAILS - do not push and call it green\n'; exit 1
fi

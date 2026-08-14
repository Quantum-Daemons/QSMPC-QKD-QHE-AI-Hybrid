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

step "1/14 test suite (pytest -m 'not slow')"
HF_HUB_OFFLINE=1 "$PY" -m pytest -m "not slow" -q -p no:warnings >/tmp/ci_pytest.log 2>&1 \
  && ok "tests" || { bad "tests"; tail -20 /tmp/ci_pytest.log; }

step "2/14 config parity (defaults.py -> config.defaults.json)"
"$PY" scripts/export_config.py >/dev/null 2>&1
git diff --quiet -- web/src/lib/config.defaults.json \
  && ok "config.defaults.json in sync" \
  || bad "config.defaults.json is stale - commit the regenerated file"

step "3/14 generated artifacts match their generator and their history"
"$PY" scripts/check_generated_prose.py \
  && ok "prose matches generators; no recorded field vanished" \
  || bad "a prose field was hand-edited, or a measurement disappeared"

step "4/14 lint, BLOCKING scope (this is the one that was red)"
"$PY" -m ruff check src/qsh/models scripts/train scripts/publish_hf.py scripts/data \
  && ok "ruff blocking scope" || bad "ruff blocking scope"

step "5/14 no confidential files tracked"
make check-secrets >/dev/null 2>&1 && ok "no confidential files" || bad "confidential files tracked"

step "6/14 no Japanese in public files"
# Scope is EVERY TRACKED FILE, with `grep -I` skipping binaries. It was a six-extension list
# (*.py *.ts *.tsx *.md *.yml *.sh) until Round 34, and the comment above it claimed the Round 33 fix
# had made "public mean exactly tracked". It had not: public meant tracked AND one of six extensions,
# leaving 122 of 356 tracked files unscanned - 59 .json, 7 .txt, 7 .tsv, 5 .tex, 3 .yaml. A
# Japanese-language confidential note saved as notes.txt or strategy.json passed both this gate and
# check-secrets. Claiming a scope the code does not have is the "gate blind to its own scope" defect
# this project keeps finding, and it was in the comment I wrote when fixing the previous instance.
#
# LATENT HAZARD, recorded rather than fixed blind: `xargs` exits 123 if ANY batch exits non-zero, so a
# match in batch 1 with no match in batch 2 still yields 123 and the `if` takes the ok branch - a false
# pass. Harmless today because the file list is one batch (about 9 KB against a 128 KB command buffer),
# and it would take roughly a 14x growth in tracked files to split. Re-check if that changes.
if git ls-files -sz | awk -v RS='\0' -v ORS='\0' '$1 != "160000" { sub(/^[^\t]*\t/, ""); print }' \
     | xargs -0 -r grep -IlP '[\x{3040}-\x{309F}\x{30A0}-\x{30FF}\x{4E00}-\x{9FFF}]'; then
  bad "Japanese characters in public files (listed above)"
else
  ok "no Japanese in public files"
fi

step "7/14 no model artifacts committed"
if git ls-files | grep -E '\.(onnx|safetensors|pt|pth|h5|npy)$'; then
  bad "model artifacts committed - publish to the Hub instead"
else
  ok "no model artifacts committed"
fi

step "8/14 web typecheck + build"
( cd web && npx tsc --noEmit >/tmp/ci_tsc.log 2>&1 ) \
  && ok "tsc --noEmit" || { bad "tsc --noEmit"; tail -15 /tmp/ci_tsc.log; }
( cd web && npm run build >/tmp/ci_build.log 2>&1 ) \
  && ok "npm run build" || { bad "npm run build"; tail -15 /tmp/ci_build.log; }

step "9/14 PQC conformance (FIPS 203 s7.2 negatives + FIPS 204 sizes)"
# Offline by design: reads the vendored CC0 snapshot in web/scripts/vectors. It was ungated for
# several rounds because it used to fetch its vectors, and the only way to gate a network check is a
# skip-on-failure fallback -- which for THIS check would be worse than not running it, since it
# exists to catch a library that silently accepts malformed keys.
( cd web && npm run --silent check:pqc >/tmp/ci_pqc.log 2>&1 ) \
  && ok "PQC conformance" || { bad "PQC conformance"; tail -12 /tmp/ci_pqc.log; }

step "10/14 QKD model plumbing and identities"
# Separate from the build for the same reason the gate above is: a green typecheck says nothing about
# whether config reaches the model. It did not - `error_correction_eff` had no reader in web/src and
# the browser ran on a literal that happened to match. This mutates the live config and requires the
# output to follow BY THE PREDICTED RATIO; a before/after diff is passed by a reader that reads
# nothing. Also asserts the CHSH/Werner identity at full precision, after the published relation
# turned out not to be the implemented one and to have passed only because both printed as "2.7".
( cd web && npm run --silent check:qkd >/tmp/ci_qkd.log 2>&1 ) \
  && ok "QKD plumbing" || { bad "QKD plumbing"; tail -12 /tmp/ci_qkd.log; }

step "11/14 Federated secure aggregation"
# The sharpest case of the same class. The diagram said "pairwise masking (demo keys)" and the DAG
# said "(secure aggregation)" while clientQFL summed plaintext updates, with `secureAgg: true` a
# literal on the return line. Every gate above stayed green throughout, because none of them can see
# a figure asserting a property of code in another file. This asserts it on the function: masks
# applied, keyed by the session key, fresh per round, cancelling to a bit-identical weighted mean,
# and no plaintext path to fall back into.
( cd web && npm run --silent check:qfl >/tmp/ci_qfl.log 2>&1 ) \
  && ok "federated secure aggregation" || { bad "federated secure aggregation"; tail -12 /tmp/ci_qfl.log; }

step "12/14 Hybrid-KEM combiner parity (Python vectors vs the browser module)"
(cd web && npm run --silent check:combiner) \
  && ok "combiner parity" \
  || bad "backend and browser derive different session keys"

step "13/14 combiner vectors regenerated from the Python side"
# The gate above reproduces the COMMITTED vectors, so it stays green against stale bytes when Python
# changed and nobody regenerated. This is the other direction - the mechanism gate 2 uses for
# config.defaults.json.
#
# It is a separate step here because CI's copy was originally placed in the Node job, where `python`
# is not installed: it failed with ModuleNotFoundError the first time it ever ran, a round after
# being written. Locally the venv is always present, so this script would have gone on saying PASS
# while CI could only ever say FAIL. That asymmetry is what this file's header is about, and a gate
# nobody has watched run has an unknown result rather than a good one.
"$PY" scripts/export_combiner_vectors.py >/dev/null 2>&1
git diff --quiet -- web/scripts/vectors/hybrid_combiner_vectors.json \
  && ok "vectors in sync with the Python combiners" \
  || bad "hybrid_combiner_vectors.json is stale - commit the regenerated file"

step "14/14 QCNN cost metrics at EVERY site that claims the backend"
# Structural, not another careful read. The latency was added to the private_inference call site and
# not to the qml_ids one, so telecom reported `primary_inference_backend = qcnn-trained` with no
# latency while two checklist rows asserted it renders on every such page. tsc, the suite and all
# thirteen gates above were green; the deployed telecom page said "inference not run".
#
# Fourth instance of one-fix-to-one-of-two-twins in this repository: the combiner length tag, the ITS
# combiner PRG, the federated mask round index, and this. The relation between two well-formed
# branches is what no type checker can see.
( cd web && npm run --silent check:qcnn-metrics >/tmp/ci_qcnnm.log 2>&1 ) \
  && ok "qcnn cost metrics at every claiming site" \
  || { bad "a qcnn-trained site omits its cost metrics"; tail -6 /tmp/ci_qcnnm.log; }

printf '\n=== advisory (not blocking in CI, printed so it is driven down deliberately) ===\n'
"$PY" -m ruff check src tests api scripts --output-format=concise 2>/dev/null | tail -3

if [ "$fail" -eq 0 ]; then
  printf '\nALL BLOCKING CI GATES PASS\n'; exit 0
else
  printf '\nAT LEAST ONE BLOCKING CI GATE FAILS - do not push and call it green\n'; exit 1
fi

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

step "1/16 test suite (pytest -m 'not slow')"
HF_HUB_OFFLINE=1 "$PY" -m pytest -m "not slow" -q -p no:warnings >/tmp/ci_pytest.log 2>&1 \
  && ok "tests" || { bad "tests"; tail -20 /tmp/ci_pytest.log; }

step "2/16 config parity (defaults.py -> config.defaults.json)"
"$PY" scripts/export_config.py >/dev/null 2>&1
git diff --quiet -- web/src/lib/config.defaults.json \
  && ok "config.defaults.json in sync" \
  || bad "config.defaults.json is stale - commit the regenerated file"

step "3/16 generated artifacts match their generator and their history"
"$PY" scripts/check_generated_prose.py \
  && ok "prose matches generators; no recorded field vanished" \
  || bad "a prose field was hand-edited, or a measurement disappeared"

step "4/16 lint, BLOCKING scope (this is the one that was red)"
"$PY" -m ruff check src/qsh/models scripts/train scripts/publish_hf.py scripts/data \
  && ok "ruff blocking scope" || bad "ruff blocking scope"

step "5/16 no confidential files tracked"
make check-secrets >/dev/null 2>&1 && ok "no confidential files" || bad "confidential files tracked"

step "6/16 no Japanese in public files"
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

step "7/16 no model artifacts committed"
if git ls-files | grep -E '\.(onnx|safetensors|pt|pth|h5|npy)$'; then
  bad "model artifacts committed - publish to the Hub instead"
else
  ok "no model artifacts committed"
fi

step "8/16 web typecheck + build"
( cd web && npx tsc --noEmit >/tmp/ci_tsc.log 2>&1 ) \
  && ok "tsc --noEmit" || { bad "tsc --noEmit"; tail -15 /tmp/ci_tsc.log; }
( cd web && npm run build >/tmp/ci_build.log 2>&1 ) \
  && ok "npm run build" || { bad "npm run build"; tail -15 /tmp/ci_build.log; }

step "9/16 PQC conformance (FIPS 203 s7.2 negatives + FIPS 204 sizes)"
# Offline by design: reads the vendored CC0 snapshot in web/scripts/vectors. It was ungated for
# several rounds because it used to fetch its vectors, and the only way to gate a network check is a
# skip-on-failure fallback -- which for THIS check would be worse than not running it, since it
# exists to catch a library that silently accepts malformed keys.
( cd web && npm run --silent check:pqc >/tmp/ci_pqc.log 2>&1 ) \
  && ok "PQC conformance" || { bad "PQC conformance"; tail -12 /tmp/ci_pqc.log; }

step "10/16 QKD model plumbing and identities"
# Separate from the build for the same reason the gate above is: a green typecheck says nothing about
# whether config reaches the model. It did not - `error_correction_eff` had no reader in web/src and
# the browser ran on a literal that happened to match. This mutates the live config and requires the
# output to follow BY THE PREDICTED RATIO; a before/after diff is passed by a reader that reads
# nothing. Also asserts the CHSH/Werner identity at full precision, after the published relation
# turned out not to be the implemented one and to have passed only because both printed as "2.7".
( cd web && npm run --silent check:qkd >/tmp/ci_qkd.log 2>&1 ) \
  && ok "QKD plumbing" || { bad "QKD plumbing"; tail -12 /tmp/ci_qkd.log; }

step "11/16 Federated secure aggregation"
# The sharpest case of the same class. The diagram said "pairwise masking (demo keys)" and the DAG
# said "(secure aggregation)" while clientQFL summed plaintext updates, with `secureAgg: true` a
# literal on the return line. Every gate above stayed green throughout, because none of them can see
# a figure asserting a property of code in another file. This asserts it on the function: masks
# applied, keyed by the session key, fresh per round, cancelling to a bit-identical weighted mean,
# and no plaintext path to fall back into.
( cd web && npm run --silent check:qfl >/tmp/ci_qfl.log 2>&1 ) \
  && ok "federated secure aggregation" || { bad "federated secure aggregation"; tail -12 /tmp/ci_qfl.log; }

step "12/16 Hybrid-KEM combiner parity (Python vectors vs the browser module)"
(cd web && npm run --silent check:combiner) \
  && ok "combiner parity" \
  || bad "backend and browser derive different session keys"

step "13/16 combiner vectors regenerated from the Python side"
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

step "14/16 QCNN cost metrics at EVERY site that claims the backend"
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

step "15/16 the local-only checklist gates are actually running"
# THE GATE ON THE GATES. docs/browser-verification-checklist.md is gitignored by design - it must not
# be published - so the four tests that read it (prefix legend, WS index, CI1 step numbering, prose
# counts) SKIP in CI and in every clone. Measured on a fresh `git archive` checkout: four skips, and
# `pytest -q` still prints PASS. Their green tick in CI means only that the file was absent.
#
# This machine is therefore the only place those invariants are enforced, which makes a missing or
# renamed checklist here indistinguishable from four passing checks. That is the same
# scope-measured-on-one-side defect this project keeps finding, applied to its own enforcement point.
if [ -f docs/browser-verification-checklist.md ]; then
  skipped=$("$PY" -m pytest tests/test_checklist_prefix_legend.py tests/test_checklist_ws_index.py \
              tests/test_ci_local_step_numbering.py tests/test_checklist_prose_counts.py \
              -q -p no:warnings 2>&1 | grep -c "LOCAL-ONLY GATE, SKIPPED" || true)
  if [ "${skipped:-0}" = "0" ]; then
    ok "checklist present; its four gates ran here (they cannot run anywhere else)"
  else
    bad "the checklist gates skipped ON THIS MACHINE - they enforce nothing anywhere"
  fi
else
  bad "docs/browser-verification-checklist.md is missing; four invariants now enforce nothing, here or in CI"
fi

step "16/16 the out-of-repo business document has been reconciled with this round"
# THE FILE EVERY DOCUMENT SWEEP MISSED, and for a structural reason rather than a lazy one:
# the confidential business-recommendations document lives at the PARENT directory, outside this
# repository (its filename is Japanese, so it is described here rather than written), so
# every sweep that walks the repo walks past it. Twenty-one consecutive ORDER files (18 through 38)
# name it in the documentation-merge bullet. Its own title says it covers rounds 17 to 20. The
# project is at round 43.
#
# It is confidential-class like MONETIZATION*, so it cannot be moved in here and its CONTENT must
# never be quoted - not in the checklist, not in a commit, not in this script. What this reads is its
# presence and the reconciliation stamp in the register: facts about staleness that say nothing about
# what it contains.
#
# LOCAL-ONLY by construction, exactly as gate 15 is - the file is not in the repository, so CI cannot
# see it and neither can a clone. Said here rather than left to be discovered.
# The pattern is built from \u escapes rather than written literally. `make check-secrets` and gate 6
# forbid Japanese characters in every tracked public file, and this script is tracked - the first
# version of this gate put the filename in directly and gate 6 failed on gate 16, which is the rule
# catching the check written to enforce a different rule.
# Override with QSH_BIZ_DOC_GLOB if the file is named differently on another machine.
# The ORDER names this document specifically; a wider glob would also match the separate strategy
# document beside it, which carries no reconciliation obligation - matching that would invent one and
# then report it unmet.
BIZ_GLOB="${QSH_BIZ_DOC_GLOB:-$(printf '../\u91cf\u5b50\u6280\u8853\u8005_\u4e8b\u696d\u63a8\u5968\u4e8b\u9805_\u61f8\u5ff5\u4e8b\u9805_*.md')}"
BIZ_DOC=$(ls -1 $BIZ_GLOB 2>/dev/null | head -1)
RECONCILED=$(grep -oE 'BUSINESS DOC RECONCILED Round [0-9]+' docs/deliverable-verification.md | tail -1 | grep -oE '[0-9]+$')
CURRENT=$(grep -oE '^## Rounds? [0-9]+' docs/deliverable-verification.md | grep -oE '[0-9]+$' | sort -n | tail -1)
if [ -z "$BIZ_DOC" ]; then
  ok "no out-of-repo business document on this machine; nothing to reconcile here"
elif [ -z "$RECONCILED" ]; then
  bad "the register carries no 'BUSINESS DOC RECONCILED Round <N>' stamp; 21 ORDER files ask for this file to be kept current"
elif [ "${RECONCILED:-0}" -lt "${CURRENT:-0}" ]; then
  bad "business document last reconciled in Round $RECONCILED, register is at Round $CURRENT"
else
  ok "business document reconciled in Round $RECONCILED (register at $CURRENT)"
fi

printf '\n=== advisory (not blocking in CI, printed so it is driven down deliberately) ===\n'
"$PY" -m ruff check src tests api scripts --output-format=concise 2>/dev/null | tail -3

if [ "$fail" -eq 0 ]; then
  printf '\nALL BLOCKING CI GATES PASS\n'; exit 0
else
  printf '\nAT LEAST ONE BLOCKING CI GATE FAILS - do not push and call it green\n'; exit 1
fi

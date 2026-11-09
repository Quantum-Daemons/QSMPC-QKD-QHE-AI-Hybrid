#!/bin/sh
# Checks every locked dependency against the published advisories, and fails on any. The same script
# runs locally and in .github/workflows/audit.yml (daily, and whenever a manifest, a lockfile or the
# policy changes), so the two cannot drift apart.
#
#   bash scripts/audit_deps.sh          every check
#   bash scripts/audit_deps.sh npm      web/: npm audit, development tooling included
#   bash scripts/audit_deps.sh rust     wasm/qsh-mpc: cargo-deny, advisories and sources
#   bash scripts/audit_deps.sh seal     web/: the locked node-seal is still npm's newest
#   bash scripts/audit_deps.sh polytune upstream Polytune: its protocol source and the two issues that bear on this project
#   bash scripts/audit_deps.sh python   requirements.lock: pip-audit against the versions and hashes it names
#
# npm audit and the node-seal check need Node and network access. The Rust check needs Docker and network access, and
# installs nothing on the host: it runs in the same digest-pinned Rust image as the web image's wasm
# stage, as the calling user, with cargo-deny and its advisory database cached in
# wasm/qsh-mpc/.cargo-home (gitignored, kept out of the Docker context).
#
# AUDIT_WEB_DIR points the npm check at another project, which is how its failure path is checked.
set -eu
cd "$(dirname "$0")/.."

RUST_IMAGE=rust:1.99.1-bookworm@sha256:48b7f435b8fc6378ce43b44f15e3a47fd8919b2825f835c39bd4bd3da50af5b8
CARGO_DENY=0.20.2
PIP_AUDIT=2.10.1

audit_npm() {
  # No --omit=dev: the dev server and the build tooling run on developer machines and in CI, and
  # Round 59 counted them on purpose. info is the lowest level, so any advisory fails.
  (cd "${AUDIT_WEB_DIR:-web}" && npm audit --audit-level=info)
  (cd docker/mcp-fs && npm audit --audit-level=info)
}

audit_seal() {
  # Round 86 (review PI-2). The browser's CKKS is node-seal 7.0.0, Microsoft SEAL 4.1.2, while SEAL
  # 4.4.0 is a "Critical security update" and SEAL is at 4.7.2. No advisory names SEAL, so npm audit
  # passes; and node-seal's move to 4.7.2 (merged upstream 2026-10-31, released as 7.1.0 per its
  # CHANGES.md) is not on npm yet. This fails once npm's newest node-seal is not the locked one, so the
  # release cannot pass unseen: upgrade web/, rerun the CKKS tests, and update the residual risk in
  # docs/threat-model.md. AUDIT_WEB_DIR and NPM_VIEW (the command that prints npm's newest version)
  # are how its failure path is checked.
  dir="${AUDIT_WEB_DIR:-web}"
  locked=$(node -e "const l=require(require('path').resolve(process.argv[1], 'package-lock.json'));
    process.stdout.write((l.packages['node_modules/node-seal']||{}).version||'')" "$dir")
  newest=$(${NPM_VIEW:-npm view node-seal version})
  if [ -z "$locked" ] || [ -z "$newest" ]; then
    echo "node-seal: could not read the locked version ($locked) or npm's newest ($newest)" >&2
    return 1
  fi
  if [ "$locked" != "$newest" ]; then
    echo "node-seal $newest is published and $locked is locked: upgrade web/, rerun" \
      "tests/test_r84_medtel_ckks_decrypts_only_outputs.py and tests/test_r86_private_inference_review.py," \
      "and update the SEAL residual risk in docs/threat-model.md" >&2
    return 1
  fi
  echo "node-seal: $locked is locked and is npm's newest"
}

audit_polytune() {
  # Round 86 (audit item D3). The crate is pinned, and tests/test_the_polytune_integration_reports_what_happened.py fires
  # when the pin moves. Nothing noticed upstream moving, or an answer on the issues docs/threat-model.md waits on: #63,
  # the KOS consistency check, where a question was posted on 2026-10-29, and #100, the AesRng construction. Fails when
  # either issue changes, or when protocol source changes past the pin. Dependency bumps and CI files do not count.
  # POLYTUNE_API (a command taking the GitHub API path and printing JSON) is how its failure path is checked.
  rev=$(sed -n 's/^polytune = .*rev = "\([0-9a-f]\{40\}\)".*/\1/p' wasm/qsh-mpc/Cargo.toml)
  [ -n "$rev" ] || { echo "polytune: no pinned rev in wasm/qsh-mpc/Cargo.toml" >&2; return 1; }
  api=${POLYTUNE_API:-polytune_api}
  issue63=$($api "repos/sine-fdn/polytune/issues/63") || return 1
  issue100=$($api "repos/sine-fdn/polytune/issues/100") || return 1
  compare=$($api "repos/sine-fdn/polytune/compare/$rev...main") || return 1
  ISSUE63=$issue63 ISSUE100=$issue100 COMPARE=$compare node -e '
    // The comment counts read on 2026-11-09. A new one is what this exists to catch.
    const seen = { 63: 2, 100: 0 };
    const problems = [];
    for (const [n, name] of [[63, "ISSUE63"], [100, "ISSUE100"]]) {
      const i = JSON.parse(process.env[name]);
      if (i.state !== "open" || i.comments !== seen[n])
        problems.push(`issue #${n} is ${i.state} with ${i.comments} comments, and ${seen[n]} comments on an open issue were last read`);
    }
    const c = JSON.parse(process.env.COMPARE);
    // Protocol source: Rust under src/ of the crate or of a member crate, except the test utilities.
    const source = (c.files || []).map((f) => f.filename)
      .filter((f) => /(^|\/)src\/.*\.rs$/.test(f) && !f.startsWith("crates/polytune-test-utils/"));
    if (source.length) problems.push(`protocol source changed past the pin: ${source.join(", ")}`);
    if (problems.length) {
      console.error("polytune upstream needs a review:\n  " + problems.join("\n  ") +
        "\nRead it, update docs/threat-model.md, then update the counts here or the pin and REVIEWED_POLYTUNE_REV.");
      process.exit(1);
    }
    console.log(`polytune: ${c.ahead_by} commits past the pin, no protocol source among them; issues #63 and #100 unchanged`);
  '
}

polytune_api() {
  curl -fsSL -H 'Accept: application/vnd.github+json' ${GH_TOKEN:+-H "Authorization: Bearer $GH_TOKEN"} "https://api.github.com/$1"
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

audit_python() {
  # What the image and CI install, not what the floors would resolve to today. --disable-pip: the lock is complete.
  venv=$(mktemp -d)
  # pip-audit needs a newer Python than some systems' python3 (AlmaLinux 9's is 3.9).
  "$(command -v python3.12 || command -v python3)" -m venv "$venv"
  "$venv/bin/pip" install --quiet "pip-audit==$PIP_AUDIT"
  "$venv/bin/pip-audit" --disable-pip --require-hashes -r requirements.lock
  # torch and torchvision carry a +cpu local version, which PyPI's advisory lookup skips with only a note, so they are
  # asked about again under the public version they are built from.
  grep -E '^(torch|torchvision)==' requirements.lock | sed -E 's/\+cpu.*//' > "$venv/plain.txt"
  "$venv/bin/pip-audit" --disable-pip --no-deps -r "$venv/plain.txt"
  rm -rf "$venv"
}

case "${1:-all}" in
  npm) audit_npm ;;
  python) audit_python ;;
  rust) audit_rust ;;
  seal) audit_seal ;;
  polytune) audit_polytune ;;
  all) audit_npm && audit_seal && audit_polytune && audit_python && audit_rust ;;
  *) echo "usage: $0 [npm|rust|seal|polytune|python]" >&2; exit 2 ;;
esac

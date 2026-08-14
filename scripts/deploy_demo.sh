#!/usr/bin/env bash
# Deploy a branch to the public demo host, verify it, and RETURN THE BOX TO main.
#
# WHY THIS EXISTS
#
# Checklist row `DP7` requires that the demo host sit on this round's branch during verification and
# on `main` after the round's PR merges - never on a previous round's branch. It has failed **six
# consecutive rounds**, and not because anyone disagreed with it. The return-to-main happens AFTER
# the merge, which is after the session that ran the checklist ended, so the only person who could
# discharge it was never present when it came due.
#
# Round 39 rewrote the row's wording and left that structure intact, which is why it failed a sixth
# time. This is the other half of the fix the register keeps arriving at: **a rule with no executor
# is an intention**. Running this script IS the discharge - the last thing it does is put the box
# back on `main` and rebuild, so the state `DP7` asks for is the state the deploy leaves behind.
#
# CONNECTION DETAILS ARE NOT IN THIS FILE, and must never be. The demo host's address, user and key
# are confidential; this script is tracked. Everything host-specific comes from the environment:
#
#     QSH_DEMO_SSH   ssh destination, e.g. user@host
#     QSH_DEMO_KEY   path to the ssh key            (optional; ssh config may supply it)
#     QSH_DEMO_DIR   deploy tree on the host        (default ~/dev/QSMPC-QKD-QHE-AI-Hybrid)
#     QSH_DEMO_URL   public URL to verify against
#
# Put them in a gitignored file and source it. `make check-secrets` will not see this script because
# it carries nothing to see.
#
# Usage:
#     bash scripts/deploy_demo.sh <branch>          # deploy and verify, box left on <branch>
#     bash scripts/deploy_demo.sh --return-to-main  # after the PR merges: discharge DP7
#
set -uo pipefail

die() { printf '\n  FAIL  %s\n' "$1" >&2; exit 1; }
say() { printf '\n=== %s ===\n' "$1"; }
ok()  { printf '  PASS  %s\n' "$1"; }

: "${QSH_DEMO_SSH:?set QSH_DEMO_SSH (ssh destination); it is deliberately not in this file}"
: "${QSH_DEMO_URL:?set QSH_DEMO_URL (public demo URL); it is deliberately not in this file}"
DIR="${QSH_DEMO_DIR:-\$HOME/dev/QSMPC-QKD-QHE-AI-Hybrid}"
SSH=(ssh -o ConnectTimeout=20)
[ -n "${QSH_DEMO_KEY:-}" ] && SSH+=(-i "$QSH_DEMO_KEY")
SSH+=("$QSH_DEMO_SSH")

COMPOSE="docker compose -p qsh-demo -f docker/docker-compose.demo.yml -f docker/docker-compose.demo.behind-proxy.yml"
# `-p qsh-demo` AND the behind-proxy override, both. Without them the container lands on no shared
# network and the site 502s while `docker ps` cheerfully reports Up.

remote() { "${SSH[@]}" "cd $DIR && $*"; }

deploy_branch() {
  local branch="$1"
  say "fetch $branch from GitHub (never rsync - a copy ships untracked files and omits uncommitted dirs)"
  # `--prune`, and an explicit existence check against the REMOTE. Without them a branch deleted
  # upstream still resolves through a stale local `origin/<branch>` ref, so checkout and reset both
  # succeed against a commit that no longer exists and the script reports PASS while serving an old
  # bundle - a green run describing a branch nobody can fetch.
  remote "git ls-remote --exit-code --heads origin '$branch' >/dev/null" \
    || die "origin has no branch '$branch' - push it first, or the box would deploy a stale local ref"
  remote "git fetch --quiet --prune origin && git checkout --quiet '$branch' && git reset --hard --quiet origin/'$branch'" \
    || die "could not put the box on $branch (local modifications? run 'git status' there)"
  local head; head=$(remote "git rev-parse --short HEAD") || die "could not read the deployed HEAD"
  ok "box HEAD is $head"

  say "build (exit status captured, not assumed)"
  remote "$COMPOSE build" >/tmp/qsh_deploy_build.log 2>&1 \
    || { tail -20 /tmp/qsh_deploy_build.log; die "docker build failed"; }
  ok "image built"

  say "start"
  remote "$COMPOSE up -d" >/dev/null 2>&1 || die "compose up failed"
  local age; age=$(remote "docker inspect -f '{{.State.StartedAt}}' qsh-demo-web-1")
  ok "container restarted at $age"

  # The 502 trap, asserted rather than remembered: the container must be on the shared proxy network.
  local nets; nets=$(remote "docker inspect qsh-demo-web-1 --format '{{range \$k,\$v := .NetworkSettings.Networks}}{{\$k}} {{end}}'")
  case "$nets" in
    *_mgmt-net*|*proxy*) ok "on the shared proxy network ($nets)" ;;
    *) die "container is on '$nets' with no shared proxy network - the site will 502 while docker says Up" ;;
  esac

  say "verify the PUBLIC url, not localhost"
  local code; code=$(curl -s -o /dev/null -w '%{http_code}' "$QSH_DEMO_URL/?cb=$RANDOM")
  [ "$code" = "200" ] || die "public URL returned $code"
  local bundle; bundle=$(curl -s "$QSH_DEMO_URL/?cb=$RANDOM" | grep -oE '/assets/index-[A-Za-z0-9_-]+\.js' | head -1)
  [ -n "$bundle" ] || die "no bundle referenced by the served index.html"
  ok "200, serving $bundle"

  # DP7's pass condition is literally `git rev-parse --abbrev-ref HEAD` on the box, and only the
  # return half ran it. If `<branch>` is a tag or a sha, `git checkout` detaches HEAD and this path
  # reported PASS for a state the row does not accept - the deploy script's own two halves checking
  # different things, which is this round's subject inside the script written to close DP7.
  local on; on=$(remote "git rev-parse --abbrev-ref HEAD") || die "could not read the box's branch"
  [ "$on" = "$branch" ] || die "box reports branch '$on', not '$branch' (detached HEAD?) - DP7 needs a named branch"
  ok "box is on branch $on"

  # REPORTED, never deleted. `git reset --hard` leaves untracked trees behind, and the box carries
  # ten submodule directories against two the repository declares - traces of an ORDER-1-era mandate
  # that was reversed without a record. `.dockerignore` excludes `submodules/`, so they cannot reach
  # the image; deleting someone's working tree is not this script's call, but going quiet about it is
  # how "the box equals the commit" becomes an assumption instead of a fact.
  local untracked; untracked=$(remote "git status --porcelain | grep -c '^??'" || echo 0)
  if [ "${untracked:-0}" != "0" ]; then
    printf '  NOTE  %s untracked path(s) on the box that this commit does not describe:\n' "$untracked"
    remote "git status --porcelain | grep '^??' | head -12" | sed 's/^/          /'
    printf '        Excluded from the image by .dockerignore. Not deleted - that is your call.\n'
  fi
  printf '\n  Deployed %s at %s. Verify the pages in Chrome, then run:\n    bash %s --return-to-main\n' \
    "$branch" "$head" "$0"
}

return_to_main() {
  say "return the box to main - this is what DP7 asks for and what six rounds forgot"
  remote "git fetch --quiet origin && git checkout --quiet main && git reset --hard --quiet origin/main" \
    || die "could not return the box to main"
  local head; head=$(remote "git rev-parse --short HEAD")
  ok "box is on main at $head"

  say "rebuild from main, because leaving main checked out with a branch's image running is the same defect wearing a different label"
  remote "$COMPOSE build" >/tmp/qsh_deploy_build.log 2>&1 \
    || { tail -20 /tmp/qsh_deploy_build.log; die "rebuild from main failed"; }
  remote "$COMPOSE up -d" >/dev/null 2>&1 || die "compose up failed"

  # The same assertions the deploy half runs. They were asymmetric until Round 41: the network check
  # and the bundle extraction ran only on the way out, and the branch re-read only on the way back -
  # so each half trusted what the other verified, in the script written because a two-halves problem
  # went six rounds unnoticed.
  local nets; nets=$(remote "docker inspect qsh-demo-web-1 --format '{{range \$k,\$v := .NetworkSettings.Networks}}{{\$k}} {{end}}'") \
    || die "could not read the container's networks"
  case "$nets" in
    *_mgmt-net*|*proxy*) ok "on the shared proxy network ($nets)" ;;
    *) die "container is on '$nets' with no shared proxy network - the site will 502 while docker says Up" ;;
  esac

  local code; code=$(curl -s -o /dev/null -w '%{http_code}' "$QSH_DEMO_URL/?cb=$RANDOM")
  [ "$code" = "200" ] || die "public URL returned $code after returning to main"
  local bundle; bundle=$(curl -s "$QSH_DEMO_URL/?cb=$RANDOM" | grep -oE '/assets/index-[A-Za-z0-9_-]+\.js' | head -1)
  [ -n "$bundle" ] || die "no bundle referenced by the served index.html after returning to main"
  local branch; branch=$(remote "git rev-parse --abbrev-ref HEAD") || die "could not read the box's branch"
  [ "$branch" = "main" ] || die "box reports branch '$branch', not main"
  ok "DP7 discharged: box on main at $head, serving $bundle, public URL 200"
}

case "${1:-}" in
  --return-to-main) return_to_main ;;
  "" ) die "usage: $0 <branch> | $0 --return-to-main" ;;
  * ) deploy_branch "$1" ;;
esac

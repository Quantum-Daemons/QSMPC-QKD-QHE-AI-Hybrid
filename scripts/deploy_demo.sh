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
#     QSH_DEMO_DIR   deploy tree on the host
#     QSH_DEMO_URL   public URL to verify against
#     QSH_DEMO_ADDRESS  address the box serves on (optional; by default the address ssh would connect to)
#     PROXY_NETWORK  the network the host's reverse proxy owns
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
: "${PROXY_NETWORK:?set PROXY_NETWORK (the reverse proxy network on the host); it is deliberately not in this file}"
: "${QSH_DEMO_DIR:?set QSH_DEMO_DIR (the deploy tree on the host); it is deliberately not in this file}"
DIR="$QSH_DEMO_DIR"
SSH=(ssh -o ConnectTimeout=20)
[ -n "${QSH_DEMO_KEY:-}" ] && SSH+=(-i "$QSH_DEMO_KEY")
SSH+=("$QSH_DEMO_SSH")

COMPOSE="PROXY_NETWORK=$PROXY_NETWORK docker compose -p qsh-demo -f docker/docker-compose.demo.yml -f docker/docker-compose.demo.behind-proxy.yml"
# `-p qsh-demo` AND the behind-proxy override, both. Without them the container lands on no shared
# network and the site 502s while `docker ps` cheerfully reports Up.

remote() { "${SSH[@]}" "cd $DIR && $*"; }

# What THIS box serves under the public name, read through a resolve pinned to its address (checklist DP18). During a
# move between hosts the public name can resolve to another machine, and a PASS read through DNS alone then describes
# that machine: on 2026-11-19 the names pointed at a temporary host while the usual one was being redeployed. `ssh -G`
# applies the ssh config, so an alias in QSH_DEMO_SSH gives the address ssh itself would use; QSH_DEMO_ADDRESS
# overrides it where the box serves on another address. The address is never printed.
url_host=${QSH_DEMO_URL#*://}; url_host=${url_host%%/*}
url_port=443
case "$url_host" in *:*) url_port=${url_host##*:}; url_host=${url_host%:*} ;; esac

box_address() {
  local a="${QSH_DEMO_ADDRESS:-}"
  [ -n "$a" ] || a=$(ssh -G "$QSH_DEMO_SSH" 2>/dev/null | awk '$1 == "hostname" {print $2; exit}')
  if [[ ! "$a" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    a=$(getent ahostsv4 "$a" 2>/dev/null | awk 'NR == 1 {print $1}')
  fi
  printf '%s' "$a"
}

BUNDLE=""
verify_served() {  # $1 says when, for the messages; sets BUNDLE to the bundle the box serves
  local addr; addr=$(box_address)
  [ -n "$addr" ] || die "could not work out the box's address from QSH_DEMO_SSH; set QSH_DEMO_ADDRESS"
  local pin=(--resolve "$url_host:$url_port:$addr")
  local code; code=$(curl -s "${pin[@]}" -o /dev/null -w '%{http_code}' "$QSH_DEMO_URL/?cb=$RANDOM")
  [ "$code" = "200" ] || die "the box returned $code for the public name $1"
  BUNDLE=$(curl -s "${pin[@]}" "$QSH_DEMO_URL/?cb=$RANDOM" | grep -oE '/assets/index-[A-Za-z0-9_-]+\.js' | head -1)
  [ -n "$BUNDLE" ] || die "no bundle referenced by the index.html the box serves $1"
  ok "the box serves the public name: 200, $BUNDLE"
  local pcode pbundle
  pcode=$(curl -s -o /dev/null -w '%{http_code}' "$QSH_DEMO_URL/?cb=$RANDOM")
  pbundle=$(curl -s "$QSH_DEMO_URL/?cb=$RANDOM" | grep -oE '/assets/index-[A-Za-z0-9_-]+\.js' | head -1)
  if [ "$pcode" = "200" ] && [ "$pbundle" = "$BUNDLE" ]; then
    ok "the public name, as DNS resolves it here, gets the same build"
  else
    printf '  NOTE  the public name, as DNS resolves it here, gets %s %s: another machine answers it. During a\n' \
      "$pcode" "${pbundle:-with no bundle}"
    printf '        move between hosts, move the names only once every host they may reach serves %s (DP18).\n' "$BUNDLE"
  fi
}

# The security headers the box sends, against what the Caddyfile in this checkout writes (Round 86, the owner's choice of
# 2026-11-20). A Caddyfile change that dropped HSTS or a policy would otherwise deploy unnoticed until a walk. Read through
# the same pinned resolve as the bundle, for the app and for the walkthrough, which has its own policy. Every expected
# value comes from the Caddyfile, so the check moves with it.
CADDYFILE="$(cd "$(dirname "$0")/.." && pwd)/docker/env/Caddyfile.client-only"
TAB=$(printf '\t')
caddy_matched() {  # $1 matcher, $2 header name: the value `header @<matcher> ><name> "..."` sets
  sed -n "s/^${TAB}header @$1 >$2 \"\(.*\)\"\$/\1/p" "$CADDYFILE" | head -1
}
caddy_block() {  # $1 header name: the value a `header { ><name> "..." }` block sets
  sed -n "s/^${TAB}${TAB}>$1 \"\(.*\)\"\$/\1/p" "$CADDYFILE" | head -1
}
header_value() {  # $1 a header block, $2 a header name in any case: its value
  printf '%s\n' "$1" | tr -d '\r' | awk -v n="$(printf '%s' "$2" | tr 'A-Z' 'a-z')" \
    'index(tolower($0), n ":") == 1 { sub(/^[^:]*:[ \t]*/, ""); print; exit }'
}
verify_headers() {  # $1 says when
  local addr; addr=$(box_address)
  local pin=(--resolve "$url_host:$url_port:$addr")
  local app walk want block matcher name which
  app=$(curl -s -D - -o /dev/null "${pin[@]}" "$QSH_DEMO_URL/?cb=$RANDOM")
  walk=$(curl -s -D - -o /dev/null "${pin[@]}" "$QSH_DEMO_URL/walkthrough/?cb=$RANDOM")
  # A missing file: Caddy's own error response, which the Caddyfile's handle_errors gives the same headers (2026-11-20).
  local missing; missing=$(curl -s -D - -o /dev/null "${pin[@]}" "$QSH_DEMO_URL/assets/qsh-deploy-check-missing-$RANDOM.js")
  printf '%s\n' "$missing" | head -1 | grep -q ' 404' || die "a missing file did not get a 404 $1"
  for spec in "app_csp|Content-Security-Policy|app" "app_csp|Content-Security-Policy-Report-Only|app" \
              "walkthrough_csp|Content-Security-Policy|walkthrough"; do
    IFS='|' read -r matcher name which <<< "$spec"
    want=$(caddy_matched "$matcher" "$name")
    [ -n "$want" ] || die "the Caddyfile in this checkout sets no $name for @$matcher"
    block="$app"; [ "$which" = walkthrough ] && block="$walk"
    [ "$(header_value "$block" "$name")" = "$want" ] || die "the box's $name for the $which is not the Caddyfile's $1"
  done
  for name in Strict-Transport-Security X-Content-Type-Options Referrer-Policy Cross-Origin-Opener-Policy \
              Cross-Origin-Embedder-Policy Cross-Origin-Resource-Policy; do
    want=$(caddy_block "$name")
    [ -n "$want" ] || die "the Caddyfile in this checkout sets no $name"
    [ "$(header_value "$app" "$name")" = "$want" ] || die "the box does not send $name: $want $1"
    [ "$(header_value "$missing" "$name")" = "$want" ] || die "a missing file's 404 does not carry $name: $want $1"
  done
  ok "the box sends the Caddyfile's security headers: both policies, the Report-Only candidate, HSTS, nosniff, the referrer policy, COOP, COEP and CORP, and a missing file's 404 carries the last six"
  # The Caddyfile drops Server; one that still arrives was added in front of this container (the demo's front proxy drops
  # it too, since 2026-11-20). A note, not a failure: that proxy's configuration is not in this repository.
  local server; server=$(header_value "$app" Server)
  [ -n "$server" ] || server=$(header_value "$missing" Server)
  [ -z "$server" ] || printf '  NOTE  the box sends "Server: %s", which this Caddyfile drops: something in front of the container adds it %s\n' "$server" "$1"
}

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
  # Its output is kept, as the build's is. It went to /dev/null until Round 86, when a start failed because the host's
  # container runtime was being upgraded and restarted under it, and the script said only "compose up failed".
  remote "$COMPOSE up -d" >/tmp/qsh_deploy_up.log 2>&1 \
    || { tail -20 /tmp/qsh_deploy_up.log; die "compose up failed"; }
  local age; age=$(remote "docker inspect -f '{{.State.StartedAt}}' qsh-demo-web-1")
  ok "container restarted at $age"

  # The 502 trap, asserted rather than remembered: the container must be on the shared proxy network.
  local nets; nets=$(remote "docker inspect qsh-demo-web-1 --format '{{range \$k,\$v := .NetworkSettings.Networks}}{{\$k}} {{end}}'")
  case "$nets" in
    *"$PROXY_NETWORK"*) ok "on the shared proxy network ($nets)" ;;
    *) die "container is on '$nets' with no shared proxy network - the site will 502 while docker says Up" ;;
  esac

  say "verify the public name on this box (a pinned resolve), not localhost, then the name as DNS resolves it"
  verify_served "after the deploy"
  verify_headers "after the deploy"

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
  # `grep -c` prints 0 AND exits 1 when nothing matches, so `|| echo 0` appended a SECOND zero and
  # the count arrived as "0\n0" - non-empty, so the NOTE fired on every clean deploy with an empty
  # list under it. A warning that always fires is a warning nobody reads, which is the failure this
  # block was written to avoid. `grep -c ... || true` keeps the count and discards the status.
  local untracked; untracked=$(remote "git status --porcelain | grep -c '^??' || true")
  untracked=$(printf '%s' "$untracked" | tr -dc '0-9')
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
  remote "$COMPOSE up -d" >/tmp/qsh_deploy_up.log 2>&1 \
    || { tail -20 /tmp/qsh_deploy_up.log; die "compose up failed"; }

  # The same assertions the deploy half runs. They were asymmetric until Round 41: the network check
  # and the bundle extraction ran only on the way out, and the branch re-read only on the way back -
  # so each half trusted what the other verified, in the script written because a two-halves problem
  # went six rounds unnoticed.
  local nets; nets=$(remote "docker inspect qsh-demo-web-1 --format '{{range \$k,\$v := .NetworkSettings.Networks}}{{\$k}} {{end}}'") \
    || die "could not read the container's networks"
  case "$nets" in
    *"$PROXY_NETWORK"*) ok "on the shared proxy network ($nets)" ;;
    *) die "container is on '$nets' with no shared proxy network - the site will 502 while docker says Up" ;;
  esac

  verify_served "after returning to main"
  verify_headers "after returning to main"
  local branch; branch=$(remote "git rev-parse --abbrev-ref HEAD") || die "could not read the box's branch"
  [ "$branch" = "main" ] || die "box reports branch '$branch', not main"
  ok "DP7 discharged: box on main at $head, serving $BUNDLE under the public name"
}

case "${1:-}" in
  --return-to-main) return_to_main ;;
  "" ) die "usage: $0 <branch> | $0 --return-to-main" ;;
  * ) deploy_branch "$1" ;;
esac

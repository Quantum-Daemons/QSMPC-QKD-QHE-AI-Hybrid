#!/usr/bin/env bash
# Confidentiality guards for what the repository publishes: file NAMES and file CONTENTS.
#
# WHY THE CONTENT CHECKS (2026-10-29). The name checks passed for months while three things leaked
# through file contents: the demo's host name in a compose comment and in the verification log, a
# local home path (whose user name is the deploy login) inside two tracked LaTeX build files, and
# the private notes named in three prose documents. All three were removed from the whole git
# history that day; these checks keep them from coming back.
#
# The deploy values are read from .env.demo when it exists (it is git-ignored and per machine), so
# the values themselves never appear here. CI has no .env.demo and runs the other checks. A finding
# names the file, never the value. Run: make check-secrets   (or: bash scripts/check_secrets.sh)
set -u
fail=0
ok() { printf 'OK: %s\n' "$1"; }
bad() { printf 'FAIL: %s\n' "$1"; fail=1; }

# --- names --------------------------------------------------------------------------------------
if git ls-files | grep -qi 'MONETIZATION'; then bad "a private-notes file is tracked"; else ok "private notes untracked"; fi
if git ls-files | grep -q '\.docx$'; then bad "a .docx is tracked"; else ok "report .docx untracked"; fi
if git ls-files -z | tr '\0' '\n' | grep -qP '[\x{3040}-\x{309F}\x{30A0}-\x{30FF}\x{4E00}-\x{9FFF}]'; then
  bad "a file with a Japanese name is tracked - the confidential strategy notes are named this way"
else ok "no CJK-named file tracked"; fi
if git ls-files | grep -q 'browser-verification-checklist'; then
  bad "the browser verification checklist is tracked - it must never be published"
else ok "verification checklist untracked"; fi

# --- contents -----------------------------------------------------------------------------------
# An absolute home directory names a local account. /home/user/ is the placeholder and passes.
# `git grep -E` everywhere, then GNU grep's PCRE for the placeholder: a git built without PCRE would
# make `git grep -P` fail, and an empty result would read as a pass.
homes=$(git grep -nIE '/(home|Users)/[A-Za-z0-9._-]+/' -- . | grep -P '/(home|Users)/(?!user/)[A-Za-z0-9._-]+/' | cut -d: -f1 | sort -u)
if [ -n "$homes" ]; then bad "an absolute home path is tracked in: $(echo $homes)"; else ok "no absolute home paths"; fi

# The private notes may be named by the files that keep them out of the repository, and nowhere else.
named=$(git grep -Il 'MONETIZATION' -- . ':!.gitignore' ':!.dockerignore' ':!.github/workflows/ci.yml' \
          ':!Makefile' ':!scripts/ci_local.sh' ':!scripts/check_secrets.sh' ':!tests' 2>/dev/null)
if [ -n "$named" ]; then bad "the private notes are named in: $(echo $named)"; else ok "private notes named only by the guards"; fi

# The deploy target, from the per-machine env file. QSH_DEMO_PROVIDER is optional.
# Also the host's REGISTERED DOMAIN (2026-10-30): a sibling site under the same domain was linked from a
# page, and it resolved to the demo's own address, so naming the domain anywhere names the box. Two labels,
# or three under a generic second level (co.jp, ac.uk, com.au ...), so the check never widens to a TLD.
if [ -f .env.demo ]; then
  vals=$( (set -a; . ./.env.demo >/dev/null 2>&1; set +a
           url=${QSH_DEMO_URL:-}; h=${url#*://}; h=${h%%/*}; h=${h%%:*}
           IFS=. read -r -a lab <<< "$h"; n=${#lab[@]}; d=""
           if [ "$n" -ge 3 ] && [[ ${lab[n-2]} =~ ^(co|ac|go|or|ne|com|net|org|gov|edu)$ ]]; then
             d="${lab[n-3]}.${lab[n-2]}.${lab[n-1]}"
           elif [ "$n" -ge 2 ]; then d="${lab[n-2]}.${lab[n-1]}"; fi
           printf '%s\n%s\n%s\n%s\n' "$h" "${QSH_DEMO_SSH%%@*}" "${QSH_DEMO_PROVIDER:-}" "$d") )
  leaked=""
  while IFS= read -r v; do
    [ ${#v} -ge 4 ] || continue
    f=$(git grep -IlF -- "$v" 2>/dev/null)
    [ -n "$f" ] && leaked="$leaked $(echo $f)"
  done <<< "$vals"
  if [ -n "$leaked" ]; then bad "a value from .env.demo appears in tracked files:$leaked"; else ok "no deploy values from .env.demo tracked"; fi
else
  ok "no .env.demo on this machine; deploy values not checked here"
fi

exit $fail

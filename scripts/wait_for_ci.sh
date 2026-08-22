#!/usr/bin/env bash
# Wait for CI on a branch's CURRENT head and exit non-zero unless every run CONCLUDED SUCCESS.
#
# WHY THIS EXISTS. Round 55 merged a PR on CANCELLED CI. The ad-hoc wait loop used in that session
# was:
#
#     [ "$n" -ge 2 ] && [ "$n" = "$c" ] && break     # c = count of status=="completed"
#
# and `cancelled` IS `completed`. Pushing a follow-up commit cancels the in-flight runs, so the loop
# saw two completed runs, exited, and the next line merged. The head_sha check that session had
# established was performed and passed - the commit was right, the VERDICT was never read.
#
# That is this project's second signature defect in its own tooling: a check that reads the MECHANISM
# (did the run finish) and not the OUTCOME (did it pass). It is written down here so the next session
# inherits the check rather than the habit.
#
#   usage: bash scripts/wait_for_ci.sh <branch> [expected-sha] [min-runs]
set -euo pipefail
branch="${1:?usage: wait_for_ci.sh <branch> [sha] [min-runs]}"
want="${2:-$(git rev-parse HEAD)}"
min="${3:-2}"

# A BOUNDED wait. An unbounded one hangs forever when no run ever appears for the sha - which is a
# real state (a push that raced a workflow edit, a sha that was never pushed), and a hang reads as
# "still working" rather than as "no evidence exists".
deadline=$(( SECONDS + ${WAIT_FOR_CI_TIMEOUT:-2400} ))
while :; do
  runs=$(gh run list --branch "$branch" --limit 10 \
           --json headSha,status,conclusion \
           --jq "[.[] | select(.headSha==\"$want\")]")
  total=$(jq 'length' <<<"$runs")
  done_=$(jq '[.[] | select(.status=="completed")] | length' <<<"$runs")
  if [ "$total" -ge "$min" ] && [ "$total" = "$done_" ]; then break; fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "TIMED OUT after ${WAIT_FOR_CI_TIMEOUT:-2400}s: $done_/$total run(s) complete for ${want:0:7}."
    echo "No verdict is not a pass. Do not merge."
    exit 2
  fi
  sleep 60
done

jq -r '.[] | "  \(.headSha[0:7]) \(.status) \(.conclusion)"' <<<"$runs"

bad=$(jq '[.[] | select(.conclusion != "success")] | length' <<<"$runs")
if [ "$bad" -ne 0 ]; then
  echo "NOT GREEN: $bad of $total run(s) on $want did not conclude success."
  echo "A cancelled run is COMPLETED and is NOT a pass. Do not merge."
  exit 1
fi
echo "GREEN: $total/$total runs on ${want:0:7} concluded success."

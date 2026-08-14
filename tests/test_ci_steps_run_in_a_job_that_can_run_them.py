"""A CI step must live in a job that installs what the step runs.

WHY THIS EXISTS

Round 39 added "Combiner vectors are regenerated" - `python scripts/export_combiner_vectors.py`
followed by a `git diff --exit-code` - to the **web** job. That job runs `actions/setup-node` and
nothing else, so the step failed with `ModuleNotFoundError: No module named 'oqs'` the first time it
ever executed, one round after being written.

`scripts/ci_local.sh` could not have caught it and never will: locally the virtualenv is always on
the path, so the same command passes. That is exactly the local-green/CI-red asymmetry the script's
own header was written about, reappearing in the script that exists to close it. The gate was
correct, the command was correct, and it was in a job that could not run it.

This is a static check over the workflow file. It cannot verify that a job installs everything a
step needs - only that a step invoking an interpreter is in a job that sets that interpreter up -
but that is the failure that actually happened, and it is the one a reviewer reading a diff of
`ci.yml` is least likely to see, because the step looks right in isolation.

KNOWN LIMIT, stated rather than implied: this checks `python` and `npm`/`npx`. A step shelling out
to some other uninstalled tool is not covered, and no claim is made that it is.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is needed to parse the workflow")

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"

#: An interpreter invoked at the start of a command line (or after `&&`/`|`/newline).
_PYTHON = re.compile(r"(?:^|[\n;&|]\s*)python[3]?\s", re.M)
_NODE = re.compile(r"(?:^|[\n;&|]\s*)(?:npm|npx|node)\s", re.M)


def _jobs() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]


def _sets_up(job: dict, action: str) -> bool:
    return any(str(s.get("uses", "")).startswith(action) for s in job.get("steps", []))


def test_the_workflow_parses_into_jobs_with_steps() -> None:
    """Guard against a parse that silently yields nothing to check."""
    jobs = _jobs()
    assert jobs, "no jobs parsed from ci.yml"
    assert all(j.get("steps") for j in jobs.values()), "a job parsed with no steps"


@pytest.mark.parametrize("job_name", list(_jobs()))
def test_steps_run_in_a_job_that_sets_up_their_interpreter(job_name: str) -> None:
    job = _jobs()[job_name]
    has_python = _sets_up(job, "actions/setup-python")
    has_node = _sets_up(job, "actions/setup-node")

    offenders = []
    for step in job["steps"]:
        run = step.get("run")
        if not run:
            continue
        name = step.get("name", "<unnamed>")
        if _PYTHON.search(run) and not has_python:
            offenders.append(f"{name!r} runs python, but job {job_name!r} has no setup-python")
        if _NODE.search(run) and not has_node:
            offenders.append(f"{name!r} runs npm/npx/node, but job {job_name!r} has no setup-node")

    assert not offenders, (
        "CI steps are in a job that cannot run them:\n  " + "\n  ".join(offenders)
        + "\n\nThis exact mistake shipped once: a python step in the Node-only web job, which failed"
        " with ModuleNotFoundError the first time it ran. Move the step to the job that installs its"
        " interpreter - do not add an install line to make the wrong job work, because that"
        " duplicates a dependency set that is already maintained somewhere else."
    )


def test_the_check_can_fail() -> None:
    """The regexes must actually match a realistic step body.

    A test whose detector matches nothing passes for every workflow ever written, which is how a
    green tick becomes worthless. This feeds it the exact step that broke.
    """
    broke = "python scripts/export_combiner_vectors.py\ngit diff --exit-code -- web/x.json\n"
    assert _PYTHON.search(broke), "the python detector no longer matches the step that broke CI"
    assert _NODE.search("npm ci"), "the node detector no longer matches a plain npm invocation"
    assert not _PYTHON.search("make check-secrets"), "the python detector matches unrelated commands"

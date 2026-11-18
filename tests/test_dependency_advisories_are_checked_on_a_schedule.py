"""Every locked dependency is checked against the published advisories, daily and on every change.

WHY THIS EXISTS

Rounds 59 to 65 took npm audit from 16 flagged packages to 0, development tooling included, by hand:
each round ran it and recorded the count. Nothing kept the count at 0. An advisory published against
a package that is already locked changes nothing in the repository, so no push-triggered check can
see it, and the 69 Rust crates compiled into the MPC wasm (Round 65) had never been audited at all.

WHAT IS ASSERTED

- .github/workflows/audit.yml runs on a schedule, on demand, and on any change to a manifest, a
  lockfile, the cargo-deny policy, the script or the workflow; it can only read the repository; and
  both of its jobs run scripts/audit_deps.sh, not a separate recipe;
- the script audits web/ with development tooling included (no --omit=dev) at the lowest level, and
  runs a pinned cargo-deny in the same digest-pinned Rust image as the web image's wasm stage;
- the policy fails on every class of advisory and on any unexpected source, and every advisory it
  ignores is listed in IGNORED below with the condition that reopens it, so an ignore cannot be added
  to deny.toml without being reviewed here;
- the same for npm (Round 86, 2026-11-18): every advisory scripts/npm_audit_ignore.json lets through is
  listed in NPM_IGNORED below, and scripts/npm_audit_filter.mjs lets it through only on the packages named,
  before its date and while no release fixes it, each of which is checked to fail here.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import tomllib
from datetime import date
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "audit.yml"
SCRIPT = REPO / "scripts" / "audit_deps.sh"
POLICY = REPO / "wasm" / "qsh-mpc" / "deny.toml"

#: Each advisory deny.toml ignores, with what reopens it. Adding one to deny.toml without adding it
#: here fails; so does leaving one here after it has gone from deny.toml.
IGNORED = {
    "RUSTSEC-2025-0141": "bincode 2.0.1 is unmaintained; reopen when polytune moves off bincode, or if a "
                         "vulnerability is reported against it (that would be a separate advisory, not ignored)",
}

#: Each npm advisory scripts/npm_audit_ignore.json lets through, with what reopens it, as IGNORED is for deny.toml.
NPM_IGNORED = {
    "GHSA-fc5q-39xc-vjpf": "picomatch <= 4.0.7 (CVE-2026-112366), disputed and fixed in no release, reached only through "
                           "vite's build tooling; reopen when a newer picomatch is published, when it is reported through "
                           "another package, or after 2026-12-18",
}
NPM_IGNORE = REPO / "scripts" / "npm_audit_ignore.json"
NPM_FILTER = REPO / "scripts" / "npm_audit_filter.mjs"

WATCHED = {
    "web/package.json", "web/package-lock.json", "wasm/qsh-mpc/Cargo.toml", "wasm/qsh-mpc/Cargo.lock",
    "wasm/qsh-mpc/deny.toml", "scripts/audit_deps.sh", ".github/workflows/audit.yml",
    "scripts/npm_audit_ignore.json", "scripts/npm_audit_filter.mjs",
}


def _workflow() -> dict:
    wf = yaml.safe_load(WORKFLOW.read_text())
    wf["on"] = wf.pop(True, wf.get("on"))  # YAML 1.1 reads the bare key `on` as true
    return wf


def test_the_workflow_runs_daily_on_demand_and_on_every_watched_change() -> None:
    on = _workflow()["on"]
    crons = [s["cron"] for s in on.get("schedule", [])]
    assert crons and all(c.split()[2:] == ["*", "*", "*"] for c in crons), (
        f"the audit must run daily: an advisory is published without any change here ({crons})")
    assert "workflow_dispatch" in on
    for event in ("push", "pull_request"):
        paths = set(on[event]["paths"])
        assert WATCHED <= paths, f"{event} does not watch {sorted(WATCHED - paths)}"
    assert on["push"]["branches"] == ["main"]


def test_the_workflow_reads_only_and_runs_the_script() -> None:
    wf = _workflow()
    assert wf["permissions"] == {"contents": "read"}
    runs = {name: [s.get("run", "") for s in job["steps"]] for name, job in wf["jobs"].items()}
    assert any("scripts/audit_deps.sh npm" in r for r in runs.get("npm", [])), "the npm job no longer runs the script"
    assert any("scripts/audit_deps.sh rust" in r for r in runs.get("rust", [])), "the rust job no longer runs the script"
    assert any("scripts/audit_deps.sh python" in r for r in runs.get("python", [])), "the python job no longer runs the script"
    uses = {s["uses"].split("@")[0] for job in wf["jobs"].values() for s in job["steps"] if "uses" in s}
    assert uses <= {"actions/checkout", "actions/setup-node", "actions/setup-python", "actions/cache"}, (
        f"the audit runs through the script, not a third-party action: {sorted(uses)}")


def test_the_script_audits_everything_with_pinned_tools() -> None:
    src = SCRIPT.read_text()
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert re.search(r"npm audit --audit-level=info\b", code), "npm audit must fail at the lowest level"
    assert "--omit=dev" not in code and "--production" not in code, (
        "development tooling is part of what is audited (Round 59), so it must not be omitted")
    m = re.search(r"^CARGO_DENY=(\d+\.\d+\.\d+)$", code, re.M)
    assert m, "cargo-deny must be pinned to an exact version"
    assert "cargo install --locked --quiet cargo-deny@$CARGO_DENY" in code
    assert "check advisories sources" in code
    image = re.search(r"^RUST_IMAGE=(\S+)$", code, re.M)
    dockerfile = (REPO / "docker" / "Dockerfile.web").read_text()
    stage = re.search(r"^FROM\s+(\S+)\s+AS\s+wasm\s*$", dockerfile, re.M)
    assert image and stage and image.group(1) == stage.group(1), (
        "the audit must run in the same digest-pinned Rust image as the web image's wasm stage")
    assert "@sha256:" in image.group(1)


def test_the_policy_fails_on_every_advisory_class_and_unknown_source() -> None:
    policy = tomllib.loads(POLICY.read_text())
    adv, src = policy["advisories"], policy["sources"]
    assert adv.get("yanked") == "deny"
    assert adv.get("unmaintained") == "all", "unmaintained advisories must count, for every crate"
    assert "unsound" not in adv or adv["unsound"] == "all"
    assert src.get("unknown-registry") == "deny" and src.get("unknown-git") == "deny"
    assert src.get("allow-git") == ["https://github.com/sine-fdn/polytune"], (
        "a new git source must be reviewed: only the pinned Polytune repository is allowed")
    cargo = (REPO / "wasm" / "qsh-mpc" / "Cargo.toml").read_text()
    assert 'git = "https://github.com/sine-fdn/polytune"' in cargo


def test_every_ignored_advisory_has_a_reason_and_a_reopening_condition() -> None:
    ignores = tomllib.loads(POLICY.read_text())["advisories"].get("ignore", [])
    ids = []
    for entry in ignores:
        assert isinstance(entry, dict) and entry.get("reason", "").strip(), (
            f"an ignored advisory needs its reason written beside it in deny.toml: {entry}")
        ids.append(entry["id"])
    assert sorted(ids) == sorted(IGNORED), (
        f"deny.toml ignores {sorted(ids)} but this test reviews {sorted(IGNORED)}; add the reopening "
        "condition here, or drop the stale entry")


def test_the_policy_stays_out_of_the_image() -> None:
    lines = (REPO / ".dockerignore").read_text().splitlines()
    assert "wasm/qsh-mpc/deny.toml" in lines, "editing the policy would invalidate the cached Rust stage"
    assert "wasm/qsh-mpc/.cargo-home/" in lines


def test_every_npm_advisory_let_through_has_a_reason_a_path_and_an_end() -> None:
    entries = {k: v for k, v in json.loads(NPM_IGNORE.read_text()).items() if not k.startswith("_")}
    assert sorted(entries) == sorted(NPM_IGNORED), (
        f"scripts/npm_audit_ignore.json lets through {sorted(entries)} but this test reviews {sorted(NPM_IGNORED)}; add "
        "the reopening condition here, or drop the stale entry")
    for gid, e in entries.items():
        assert re.fullmatch(r"GHSA-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}", gid), gid
        assert e.get("reason", "").strip() and e.get("reopen", "").strip(), f"{gid} needs its reason and what reopens it"
        assert e.get("package") and isinstance(e.get("through"), list), f"{gid} must name its package and its path"
        date.fromisoformat(e["until"])


def test_the_script_runs_the_filter_only_after_npm_audit_fails() -> None:
    code = "\n".join(line for line in SCRIPT.read_text().splitlines() if not line.lstrip().startswith("#"))
    assert "npm audit --audit-level=info --json) && return 0" in code
    assert "node scripts/npm_audit_filter.mjs" in code
    assert re.search(r"\(cd \"\$1\" && npm audit --audit-level=info\) \|\| true\n\s*return 1", code), (
        "when the filter fails, npm's own report must be printed and the check must fail")


#: The shape npm 10 reported on 2026-11-18 for web/: the advisory on picomatch, and the three packages it reaches through.
_REPORT = {
    "auditReportVersion": 2,
    "vulnerabilities": {
        "picomatch": {"name": "picomatch", "severity": "high", "via": [{
            "source": 1247710, "name": "picomatch", "dependency": "picomatch", "severity": "high", "range": "<=4.0.7",
            "title": "picomatch vulnerable to ReDoS through single-branch repeated extglobs",
            "url": "https://github.com/advisories/GHSA-fc5q-39xc-vjpf"}], "effects": ["fdir", "tinyglobby", "vite"]},
        "fdir": {"name": "fdir", "severity": "high", "via": ["picomatch"], "effects": ["vite"]},
        "tinyglobby": {"name": "tinyglobby", "severity": "high", "via": ["fdir", "picomatch"], "effects": []},
        "vite": {"name": "vite", "severity": "high", "via": ["fdir", "picomatch", "tinyglobby"], "effects": []},
    },
}


def _filter(report: object, today: str, latest: dict) -> subprocess.CompletedProcess:
    env = {**os.environ, "AUDIT_TODAY": today, "NPM_LATEST": json.dumps(latest)}
    return subprocess.run(["node", str(NPM_FILTER), "web"], input=json.dumps(report), capture_output=True, text=True,
                          env=env, timeout=60)


def test_the_npm_filter_lets_through_only_what_the_file_names_within_its_terms() -> None:
    now, unfixed = "2026-11-18", {"picomatch": "4.0.7"}
    ok = _filter(_REPORT, now, unfixed)
    assert ok.returncode == 0, ok.stderr
    assert "GHSA-fc5q-39xc-vjpf" in ok.stdout and "let through until 2026-12-18" in ok.stdout
    assert _filter({"auditReportVersion": 2, "vulnerabilities": {}}, now, unfixed).returncode == 0

    failures = {
        "after its date": (_REPORT, "2026-12-19", unfixed),
        "a fixed release is published": (_REPORT, now, {"picomatch": "4.0.8"}),
        "a version it cannot read": (_REPORT, now, {"picomatch": "4.1.0-beta.1"}),
        "not a report": ({"error": {"code": "ENOTFOUND"}}, now, unfixed),
    }
    other = copy.deepcopy(_REPORT)
    other["vulnerabilities"]["rollup"] = {"name": "rollup", "severity": "high", "via": [{
        "name": "rollup", "range": "<4.22.4", "url": "https://github.com/advisories/GHSA-gcx4-mw62-g8wm"}], "effects": []}
    failures["another advisory"] = (other, now, unfixed)
    beyond = copy.deepcopy(_REPORT)
    beyond["vulnerabilities"]["vite-plugin-glob"] = {"name": "vite-plugin-glob", "severity": "high", "via": ["picomatch"],
                                                     "effects": []}
    failures["reported through a package it does not name"] = (beyond, now, unfixed)
    reshaped = copy.deepcopy(_REPORT)
    reshaped["vulnerabilities"]["picomatch"]["via"][0]["range"] = ">=2.0.0 <=4.0.7"
    failures["a range it cannot read"] = (reshaped, now, unfixed)
    for why, (report, today, latest) in failures.items():
        res = _filter(report, today, latest)
        assert res.returncode == 1, f"{why}: the filter let it through ({res.stdout})"
        assert "not let through" in res.stderr, why


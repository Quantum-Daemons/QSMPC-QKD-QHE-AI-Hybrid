"""Every tracked lockfile is scanned for known vulnerabilities, and every exception says why and until when (2026-11-23).

WHY THIS EXISTS

On the owner's choice of 2026-11-23, CI scans the lockfiles with OSV-Scanner against freshly downloaded OSV databases
(job `osv`), and `scripts/ci_local.sh` runs the same script locally. The first offline scan named five lockfiles; the
repository tracked nine, and the four it missed held a real advisory (zod, in a prototype) and a second copy of one
it had found (rustix). A list typed by hand falls behind the tree, so this holds it to `git ls-files`.

WHAT IS ASSERTED

- Every tracked lockfile (Python, npm, Cargo, Go) is an `-L` entry of `scripts/scan_lockfiles.sh`, and every entry is
  tracked.
- Every exception in `osv-scanner.toml` has an id, a reason of some length and an `ignoreUntil` date, so the scanner
  stops ignoring it on that day.
- CI downloads OSV-Scanner at a pinned version, checks it against a SHA-256, and runs the same script; the local gate
  runs it too, and says SKIP, not PASS, when it cannot.
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCKFILE = re.compile(r"(^|/)(package-lock\.json|Cargo\.lock|go\.mod|requirements[^/]*\.lock|poetry\.lock|uv\.lock|"
                      r"pnpm-lock\.yaml|yarn\.lock)$")


def _entries() -> set[str]:
    script = (REPO / "scripts" / "scan_lockfiles.sh").read_text(encoding="utf-8")
    return {m.group(1).split(":", 1)[-1] for m in re.finditer(r"-L (\S+)", script)}


def test_every_tracked_lockfile_is_scanned() -> None:
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, check=True, capture_output=True, text=True).stdout.split()
    lockfiles = {f for f in tracked if LOCKFILE.search(f)}
    assert lockfiles, "no lockfile found; the pattern is wrong"
    entries = _entries()
    assert lockfiles - entries == set(), f"tracked but not scanned: {sorted(lockfiles - entries)}"
    assert entries - lockfiles == set(), f"scanned but not tracked: {sorted(entries - lockfiles)}"


def test_every_exception_says_why_and_until_when() -> None:
    config = tomllib.loads((REPO / "osv-scanner.toml").read_text(encoding="utf-8"))
    ignored = config.get("IgnoredVulns", [])
    assert ignored, "osv-scanner.toml lists no exception; delete it or keep the reasons"
    for entry in ignored:
        assert re.fullmatch(r"(GHSA|RUSTSEC|GO|PYSEC|CVE)-[\w-]+", entry["id"]), entry
        assert len(entry.get("reason", "")) >= 60, entry["id"]
        assert isinstance(entry.get("ignoreUntil"), dt.date), f"{entry['id']} has no ignoreUntil date"


def test_ci_and_the_local_gate_run_the_same_scan() -> None:
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    job = ci[ci.index("\n  osv:\n"):ci.index("\n  web:\n")]
    assert re.search(r"osv-scanner/releases/download/v\d+\.\d+\.\d+/osv-scanner_linux_amd64", job)
    assert re.search(r'echo "[0-9a-f]{64}  osv-scanner" \| sha256sum -c -', job)
    assert "bash scripts/scan_lockfiles.sh ./osv-scanner --download-offline-databases" in job
    local = (REPO / "scripts" / "ci_local.sh").read_text(encoding="utf-8")
    assert 'bash scripts/scan_lockfiles.sh "$QSH_OSV_SCANNER"' in local
    assert "SKIP  OSV-Scanner" in local and "QSH_REQUIRE_OSV" in local

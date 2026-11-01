"""Round 84's node checks run in the suite, so CI runs them.

WHY THIS EXISTS. Round 84 added six checks under `web/scripts/` with npm scripts, and only one of
them (`check_qsms_field.mjs`, through `tests/test_r84_c1_secure_sum_field.py`) was run by anything:
the other five passed when run by hand and were run by nothing, which is how a check stops checking.
This file runs each from the Python suite, the job where CI sets `QSH_REQUIRE_BROWSER_GATES=1`, so a
missing toolchain fails there instead of skipping, and `scripts/ci_local.sh` gate 1 runs them too.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"
CHECKS = ["check_control_plane", "check_relay_pass_pairs", "check_model_cards",
          "check_validations_download", "check_link_and_label_contrast"]


def _require_node_or_skip() -> None:
    missing = [n for n, ok in (("web/node_modules/.bin/esbuild", (WEB / "node_modules" / ".bin" / "esbuild").exists()),
                               ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = f"{' and '.join(missing)} absent; run `npm ci` in web/. This is a SKIP, not a pass."
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


@pytest.mark.parametrize("check", CHECKS)
def test_the_check_runs_and_passes(check: str) -> None:
    _require_node_or_skip()
    script = WEB / "scripts" / f"{check}.mjs"
    assert script.exists(), f"{script} is gone; remove it from CHECKS only together with its npm script"
    r = subprocess.run(["node", str(script)], cwd=WEB, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"{check} failed:\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}"
    assert f"{check}: OK" in r.stdout, r.stdout[-2000:]


def test_every_round_84_check_with_an_npm_script_is_run_by_the_suite() -> None:
    """A new `check:*` script has to be run by something: by this file or by another test."""
    import json

    scripts = json.loads((WEB / "package.json").read_text())["scripts"]
    tests_text = "\n".join(p.read_text() for p in Path(__file__).parent.glob("test_*.py"))
    ci = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml").read_text()
    unrun = []
    for name, cmd in scripts.items():
        if not name.startswith("check:"):
            continue
        mjs = cmd.split("scripts/")[-1].split(".mjs")[0] if "scripts/" in cmd else None
        if mjs and mjs not in tests_text and f"npm run {name}" not in ci and f"scripts/{mjs}.mjs" not in ci:
            unrun.append(name)
    assert not unrun, f"npm check scripts that nothing runs: {unrun}"

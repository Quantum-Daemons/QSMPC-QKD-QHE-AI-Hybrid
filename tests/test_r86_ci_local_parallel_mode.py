"""`scripts/ci_local.sh`'s opt-in parallel mode keeps its checks and their inputs (the owner's choice of 2026-11-26).

With `QSH_GATE_PARALLEL=1` the test suite runs in the background beside step 8's two text-layout scans, which then run at
once, the README diagrams and step 8b's CSP check: the last gate spent 34 minutes in the suite, then 31 in the two scans
one after the other and 6.5 in the CSP check. That is only sound while three things hold, so they are asserted here:

  - the suite starts after every step that writes a file it reads (the generated config and prose, the wasm, the web
    build and its notices, the scan's own build), so it never reads one half-written;
  - its verdict is printed, and the script waits for it, before step 9, so no later step overlaps it;
  - the two scans and every other Chrome the script starts use different debugging ports.

Unset, the suite runs first, in CI's order, as before.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" / "ci_local.sh").read_text()


def test_the_suite_starts_after_every_step_that_writes_what_it_reads() -> None:
    start = SCRIPT.index("  start_suite\n")
    for writer in ('"$PY" scripts/export_config.py', '"$PY" scripts/check_generated_prose.py', "bash scripts/build_wasm.sh",
                   "npm run build >", "check:js-notices", 'npx vite build --outDir "$SCAN_OUT"',
                   'grep -q "$BUILT"'):
        assert SCRIPT.index(writer) < start, f"{writer!r} runs after the suite starts, which may read its output"


def test_the_suite_starts_however_the_scan_block_ends() -> None:
    """The scan block's other branches (a busy port, a failed build) must not leave the suite unstarted."""
    fallback = SCRIPT.index("start_suite                                         # the branches above")
    assert SCRIPT.index("  start_suite\n") < fallback < SCRIPT.index("check:readme-mermaid")
    assert re.search(r'start_suite\(\) \{ if \[ "\$\{QSH_GATE_PARALLEL:-0\}" = 1 \] && \[ -z "\$PYTEST_PID" \]', SCRIPT)


def test_its_verdict_is_awaited_before_step_9() -> None:
    wait = SCRIPT.index('if wait "$PYTEST_PID"; then ok "tests"')
    assert SCRIPT.index('step "8b/18') < wait < SCRIPT.index('step "9/18')


def test_every_chrome_has_its_own_debugging_port() -> None:
    scans = re.findall(r"CDP_PORT=(\d+) (?:FONTCONFIG_FILE=\S+ )?timeout 1500 node scripts/check_text_overlap\.mjs", SCRIPT)
    assert len(scans) == 2 and len(set(scans)) == 2, scans
    others = set(re.findall(r"CDP_PORT=(\d+) CSP_PAGES", SCRIPT)) | {"9356", "9366"}   # check_csp's default, the diagrams'
    assert not set(scans) & others, (scans, others)


def test_unset_the_suite_runs_first_in_cis_order() -> None:
    step1, step2 = SCRIPT.index('step "1/18'), SCRIPT.index('step "2/18')
    assert 'run_pytest && ok "tests"' in SCRIPT[step1:step2]
    assert SCRIPT.count("run_pytest") == 3, "one definition, one call in order, one start in the background"

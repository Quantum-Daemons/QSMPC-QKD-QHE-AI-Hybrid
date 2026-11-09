"""The local gate cannot test a stale build, or hang on a headless browser (Round 86).

WHY THIS EXISTS. An interrupted run of `scripts/ci_local.sh` left its `vite preview` listening on the scan's port. The next run's `--strictPort` server exited without a word, `curl` answered from the old server, and the text-layout gate reported on a build from before the fix it was meant to judge, in both fonts.
Nothing was hung: every process was doing what it had been told. So the script now refuses a busy port, checks that the page it scans is the build it just made, removes the server on any exit, and gives the scan a time limit.
"""

from __future__ import annotations

from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" / "ci_local.sh").read_text()


def test_a_busy_port_is_refused_before_the_server_starts() -> None:
    guard, start = SCRIPT.index('if curl -sf -o /dev/null http://localhost:4182/; then'), SCRIPT.index("vite preview --outDir \"$SCAN_OUT\" --port 4182 --strictPort")
    assert guard < start
    assert "already serves something" in SCRIPT


def test_the_page_scanned_is_the_build_just_made() -> None:
    assert 'BUILT=$(grep -o' in SCRIPT and 'grep -q "$BUILT"' in SCRIPT and "is not the build just made" in SCRIPT


def test_the_server_goes_away_on_any_exit() -> None:
    assert "trap \"pkill -f 'vite preview --outDir $SCAN_OUT' 2>/dev/null; rm -rf '$SCAN_OUT'\" EXIT" in SCRIPT


def test_the_scan_has_a_time_limit_in_both_fonts() -> None:
    assert SCRIPT.count("timeout 1500 node scripts/check_text_overlap.mjs") == 2

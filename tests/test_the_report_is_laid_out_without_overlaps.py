"""The report's figures carry no overlapping text, and its build refuses text wider than its box.

WHY THIS EXISTS (Round 80, the user's review of the PDF). Figure 1's right panel had a label under
its title and two labels drawn across the curve; Figure 2 had `scripts/ci_local.sh:` running to the
edge of its box. Looking further found more of the same in the published report: the models table
was 393 pt wider than the text, its Dataset column printed off the page; a file name in the licences
table ran over the next column; and the baselines figure showed numbers from August, because nothing
regenerated the figures. The build log had carried nine "Overfull \\hbox" warnings and nobody read it.

THE GATES
- `report/build_figures.py` checks every figure on the renderer before writing it: text against
  text, and every drawn line against every label, title and legend. `make report` runs it.
- `make report` fails on any Overfull \\hbox in the LaTeX log. CI has no LaTeX, so that half is
  local; what CI can see is asserted here: the check passes on the committed figures, it does catch
  the faults that were published, and the Makefile still runs both.
- The report's staleness gate counts `models/metrics.json` and the registry as inputs, so a figure
  drawn from old metrics fails it (tests/test_the_report_pdf_is_not_stale.py).
"""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "report" / "build_figures.py"


def test_the_report_figures_have_no_overlapping_text() -> None:
    r = subprocess.run([sys.executable, str(FIGURES), "--check"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "no overlapping text" in r.stdout


def test_the_check_catches_the_faults_that_were_published() -> None:
    """The permutation-null panel exactly as it was drawn before Round 80: one label under the title,
    two on the curve. A check that passes the new figure but not this one is a check of nothing."""
    argv = sys.argv
    sys.argv = [str(FIGURES), "--check"]
    try:
        g = runpy.run_path(str(FIGURES))
    finally:
        sys.argv = argv
    plt, np, check_layout = g["plt"], g["np"], g["check_layout"]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    p = np.linspace(0.001, 0.5, 400)
    ax.plot(p, p**2 + (1 - p)**2)
    for lbl, pv in (("BreastMNIST\n0.269", 0.2692), ("OPS-SAT\n0.214", 0.2136), ("AMLworld\n0.001", 1 / 981)):
        ax.plot(pv, pv**2 + (1 - pv)**2, "o", ms=4)
        ax.annotate(lbl, (pv, pv**2 + (1 - pv)**2), textcoords="offset points",
                    xytext=(4, -12 if pv > 0.2 else 4), fontsize=5.5)
    ax.set_title(r"A shuffled harvest scores $p^2+(1-p)^2$", fontsize=7.5)
    ax.set_ylim(0.45, 1.02)
    found = check_layout(fig, "published")
    plt.close(fig)
    assert any("overlaps text 'AMLworld" in f or "AMLworld" in f and "overlaps" in f for f in found), found
    assert sum("runs through" in f for f in found) >= 2, found


def test_the_report_build_runs_both_gates() -> None:
    make = (REPO / "Makefile").read_text()
    target = make[make.index("\nreport:"):make.index("\nclean:")]
    assert "report/build_figures.py" in target, "make report no longer draws (and checks) the figures"
    assert "Overfull" in target and "exit 1" in target, "make report no longer fails on an overfull box"

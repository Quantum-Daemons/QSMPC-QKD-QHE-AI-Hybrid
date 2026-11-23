"""The tab's DP ledger adds every release about the same people, the way Python composes them (Round 86, 2026-11-23).

WHY THIS EXISTS

Each Start draws fresh noise, so each run is its own release about the same people, and the cards stated one run's
epsilon only. `web/src/lib/dpLedger.ts` records every release the tab makes and composes each population's by adding
rho (Bun and Steinke, TCC 2016); `DpLedgerPanel` shows the total, the odds factor and the exact test bound. A ledger is
only worth its completeness: a release that bypasses it, or a composition that takes the largest epsilon instead of
adding, understates the privacy loss the page shows. Held here, the browser module bundled by esbuild and run in node:

1. composed epsilon is `qsh.ai.dp.composed_epsilon` of the recorded rhos (to 1e-12 relative), larger than any one
   release's, and each population is composed alone;
2. a genomics and a finance release carry the rho Python's twins compute for the same seeded draw;
3. the test bound is min(1, e^eps fpr + delta, 1 - e^-eps (1 - delta - fpr)) and the posterior bound
   e^eps p / (e^eps p + 1 - p);
4. a release with infinite rho (clipping and no noise) and a population's second delta are refused;
5. every DP release the page makes is recorded where it is made: `genomicsDp`, `dpCount` and `fedHeartDp` in
   clientCompute.ts (the FedHeart call site leaving out the row without noise), and the encrypted-aggregation
   comparison's DP row, which trains the federation once more with fresh noise in a Web Worker and is recorded when its
   result reaches the main thread (clientFedAggRun.ts); nothing else in the page calls a release or `trainFedHeartDp`.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from qsh.ai import dp

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
COUNTS = [120, 95, 140, 101]

RUNNER = r"""
const m = await import(%(bundle)s);
const out = { refused: [] };
const g = [1, 1, 2].map((e, i) => m.genomicsDp(%(counts)s, e, { seededLabel: "ledger:g" + i }));
for (const r of g) m.recordRelease({ population: "genomics", what: "count", rho: r.rho, delta: r.delta, epsilon: r.epsilonAchieved });
const f = m.dpCount(60, { seededLabel: "ledger:f" });
m.recordRelease({ population: "finance", what: "count", rho: f.rho, delta: f.delta, epsilon: f.epsilonAchieved });
for (const rho of [0.004, 0.02, 0.1]) m.recordRelease({ population: "medical-heart", what: "row", rho, delta: 1e-6, epsilon: 0 });
for (const bad of [{ rho: Infinity, delta: 1e-6 }, { rho: 0, delta: 1e-6 }, { rho: 0.01, delta: 1e-5 }]) {
  try { m.recordRelease({ population: "medical-heart", what: "bad", epsilon: 0, ...bad }); out.refused.push(false); }
  catch { out.refused.push(true); }
}
out.genomics = { rhos: g.map((r) => r.rho), V: g.map((r) => r.variancePerCohort), eps: g.map((r) => r.epsilonAchieved),
                 summary: m.ledgerSummary("genomics") };
out.finance = { rho: f.rho, V: f.variance, summary: m.ledgerSummary("finance") };
out.heart = m.ledgerSummary("medical-heart");
out.tests = [[1, 1e-5, 0.05], [2, 1e-6, 0.01], [8, 1e-8, 0.5], [0.5, 0.1, 0.9]].map(([e, d, a]) => [e, d, a, m.bestTestPower(e, d, a)]);
out.posteriors = [[1, 0.1], [3, 0.01], [0.2, 0.5]].map(([e, p]) => [e, p, m.posteriorAtMost(e, p)]);
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def browser(tmp_path_factory) -> dict:
    missing = [n for n, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()), ("node", shutil.which("node")))
               if not ok]
    if missing:
        msg = f"{' and '.join(missing)} absent, so the browser code cannot run. This is a SKIP, not a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    d = tmp_path_factory.mktemp("dpledger")
    entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
    entry.write_text(f"export * from {json.dumps(str(WEB / 'src/lib/dpLedger'))};\n"
                     f"export {{ genomicsDp, dpCount }} from {json.dumps(str(WEB / 'src/lib/clientDp'))};\n")
    subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
                    f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
    runner.write_text(RUNNER % {"bundle": json.dumps(bundle.as_uri()), "counts": json.dumps(COUNTS)})
    proc = subprocess.run(["node", str(runner)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"the browser module did not run:\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_ledger_composes_as_python_does(browser) -> None:
    g, f, h = browser["genomics"], browser["finance"], browser["heart"]
    assert g["summary"]["releases"] == 3 and f["summary"]["releases"] == 1 and h["releases"] == 3
    for summary, rhos in ((g["summary"], g["rhos"]), (f["summary"], [f["rho"]]), (h, [0.004, 0.02, 0.1])):
        want = dp.composed_epsilon(rhos, summary["delta"])
        assert summary["epsilon"] == pytest.approx(want, rel=1e-12)
        assert summary["rho"] == pytest.approx(sum(rhos), rel=1e-15)
    # Adding, not the largest: three genomics releases together exceed the largest alone.
    assert g["summary"]["epsilon"] > max(g["eps"]) + 0.1


def test_each_release_carries_the_twins_rho(browser) -> None:
    for i, (e, V, rho) in enumerate(zip((1, 1, 2), browser["genomics"]["V"], browser["genomics"]["rhos"], strict=True)):
        py = dp.genomics_dp(COUNTS, e, rb=dp.seeded_rand_below(f"ledger:g{i}"))
        assert py["variance_per_cohort"] == V
        assert py["rho"] == pytest.approx(rho, rel=1e-12)
    py = dp.dp_count(60, rb=dp.seeded_rand_below("ledger:f"))
    assert py["variance"] == browser["finance"]["V"] and py["rho"] == pytest.approx(browser["finance"]["rho"], rel=1e-12)


def test_the_bounds_are_the_formulas(browser) -> None:
    for e, d, a, got in browser["tests"]:
        want = min(1.0, math.exp(e) * a + d, 1 - math.exp(-e) * (1 - d - a))
        assert got == pytest.approx(want, rel=1e-12)
        assert a <= got <= 1
    for e, p, got in browser["posteriors"]:
        assert got == pytest.approx(math.exp(e) * p / (math.exp(e) * p + 1 - p), rel=1e-12)


def test_a_non_release_and_a_second_delta_are_refused(browser) -> None:
    assert browser["refused"] == [True, True, True]


def test_every_release_the_page_makes_is_recorded() -> None:
    src = (WEB / "src" / "lib" / "clientCompute.ts").read_text(encoding="utf-8")
    calls = [m for m in re.finditer(r"\b(genomicsDp|dpCount|fedHeartDp)\(", src)]
    assert sorted({m.group(1) for m in calls}) == ["dpCount", "fedHeartDp", "genomicsDp"]
    for m in calls:
        after = src[m.end(): m.end() + 900]
        assert "recordRelease(" in after, f"the {m.group(1)} release at offset {m.start()} is not recorded"
    fed = src[src.index("fedHeartDp("):]
    assert "Number.isFinite(x.rho)" in fed[:900], "the FedHeart rows are recorded without leaving out the row without noise"
    run = (WEB / "src" / "lib" / "clientFedAggRun.ts").read_text(encoding="utf-8")
    after = run[run.index('task: "compare"'):]
    assert "recordRelease(" in after and "d.dp.rho" in after, "the comparison's DP row is not recorded"
    releases = r"\b(genomicsDp|dpCount|fedHeartDp|trainFedHeartDp)\("
    allowed = {"clientCompute.ts": {"genomicsDp", "dpCount", "fedHeartDp"}, "clientFedHeart.ts": {"trainFedHeartDp"},
               "clientFedAgg.ts": {"trainFedHeartDp"}}
    for path in (WEB / "src").rglob("*.ts*"):
        if path.name in ("clientDp.ts", "dpLedger.ts"):
            continue
        found = {m.group(1) for m in re.finditer(releases, path.read_text(encoding="utf-8"))}
        # A definition is not a call: drop `function name(` from the count.
        found -= {n for n in found if re.search(rf"function {n}\(", path.read_text(encoding="utf-8"))
                  and len(re.findall(rf"\b{n}\(", path.read_text(encoding="utf-8"))) == 1}
        assert found <= allowed.get(path.name, set()), f"{path.name} makes a release the ledger does not see: {found}"

"""Differential privacy on the medical and genomics pages is Python's, draw for draw (Round 86).

`web/src/lib/clientDp.ts` samples Canonne, Kamath and Steinke's exact discrete Gaussian and accounts for Kairouz, Liu and
Steinke's distributed discrete Gaussian; `qsh.ai.dp` is its twin. `fedHeartDp` (clientFedHeart.ts) and `run_dp`
(qsh.ai.fed_heart) run the medical page's record-level DP gradient descent; `genomicsDp` and `genomics_dp` add the
genomics page's noise to the cohorts' counts. Held here, the browser modules bundled by esbuild and run in node, on the
seeded source both languages share - which is NOT private, and exists for these tests only:

1. the sampler's draws are the same in both languages, and its distribution is the discrete Gaussian's;
2. the accountant chooses the same per-party variances and states the same epsilons, each at or below its target,
   and larger when one party is counted out;
3. every row of the medical DP run - variance, epsilons, accuracies - is the same in both languages, and the row
   without noise draws nothing from the source;
4. the genomics contributions and variances are the same for every epsilon on the page's selector;
5. no page code calls the seeded source: outside its definition it appears only behind the `seededLabel` option.
"""

from __future__ import annotations

import collections
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from qsh.ai import dp, fed_heart

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
LABEL = "qsh-dp-twin"
COUNTS = [13, 14, 12, 15]

RUNNER = r"""
const m = await import(%(bundle)s);
const sites = JSON.parse(%(sites)s);
const rb = m.seededRandBelow(%(label)s);
const out = { samples: [], accounting: {}, fed: null, genomics: {} };
for (const V of [1, 9, 1000, 2 ** 40]) for (let i = 0; i < 200; i++) out.samples.push(m.discreteGaussian(V, rb));
for (const [name, a] of Object.entries(%(cases)s)) {
  const V = m.varianceFor(...a);
  out.accounting[name] = [V, m.epsilonOf(V, a[1], a[2], a[3], a[4], a[5]), m.epsilonOf(V, a[1], a[2], a[3] - 1, a[4], a[5])];
}
const r = await m.fedHeartDp(sites, 1, { seededLabel: %(label)s });
out.fed = r.rows.map((x) => ({ epsilon: x.epsilon === Infinity ? "inf" : x.epsilon, V: x.variancePerHospital,
  ea: x.epsilonAchieved === Infinity ? "inf" : x.epsilonAchieved, e3: x.epsilonOneDishonest === Infinity ? "inf" : x.epsilonOneDishonest,
  federated: x.federated }));
out.fedSource = r.source;
for (const e of m.GENOMICS_DP.epsilons) {
  const g = m.genomicsDp(%(counts)s, e, { seededLabel: %(label)s + ":g" + e });
  out.genomics[e] = [g.contributions, g.variancePerCohort, g.epsilonAchieved, g.epsilonOneDishonest];
}
console.log(JSON.stringify(out));
"""

D2_MED = fed_heart.DP_CLIP * 2 ** fed_heart.DP_FIXED_POINT_BITS + math.sqrt(14)
CASES = {
    "med8": (8, 50, D2_MED, 4, 14, 1e-6), "med1": (1, 50, D2_MED, 4, 14, 1e-6),
    "gen05": (0.5, 1, 32, 4, 1, 1e-5), "gen8": (8, 1, 32, 4, 1, 1e-5),
}


@pytest.fixture(scope="module")
def browser(tmp_path_factory) -> dict:
    missing = [n for n, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()), ("node", shutil.which("node")))
               if not ok]
    if missing:
        msg = f"{' and '.join(missing)} absent, so the browser code cannot run. This is a SKIP, not a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    d = tmp_path_factory.mktemp("dp")
    entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
    entry.write_text(f"export * from {json.dumps(str(WEB / 'src/lib/clientDp'))};\n"
                     f"export {{ fedHeartDp }} from {json.dumps(str(WEB / 'src/lib/clientFedHeart'))};\n")
    subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
                    f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
    runner.write_text(RUNNER % {"bundle": json.dumps(bundle.as_uri()), "sites": json.dumps(fed_heart.SAMPLE.read_text()),
                                "label": json.dumps(LABEL), "cases": json.dumps(CASES), "counts": json.dumps(COUNTS)})
    proc = subprocess.run(["node", str(runner)], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"the browser modules did not run:\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_sampler_draws_the_same_in_both_languages(browser) -> None:
    rb = dp.seeded_rand_below(LABEL)
    py = [dp.discrete_gaussian(v, rb) for v in (1, 9, 1000, 2 ** 40) for _ in range(200)]
    assert browser["samples"] == py


def test_the_sampler_is_the_discrete_gaussian() -> None:
    # 60,000 draws at variance 9 against the normalised pmf: a chi-square on the 17 cells within 8 of 0.
    rb = dp.seeded_rand_below("qsh-dp-distribution")
    n = 60_000
    c = collections.Counter(dp.discrete_gaussian(9, rb) for _ in range(n))
    z = sum(math.exp(-x * x / 18) for x in range(-80, 81))
    chi = sum((c.get(x, 0) - n * math.exp(-x * x / 18) / z) ** 2 / (n * math.exp(-x * x / 18) / z) for x in range(-8, 9))
    assert chi < 40.0, chi   # 16 degrees of freedom: 40 is beyond the 0.999 quantile (39.25)


def test_the_accountant_agrees_and_meets_each_target(browser) -> None:
    for name, a in CASES.items():
        v = dp.variance_for(*a)
        e4 = dp.epsilon_of(v, a[1], a[2], a[3], a[4], a[5])
        e3 = dp.epsilon_of(v, a[1], a[2], a[3] - 1, a[4], a[5])
        assert browser["accounting"][name] == [v, e4, e3], name
        assert e4 <= a[0] < dp.epsilon_of(v - 1, a[1], a[2], a[3], a[4], a[5]), name   # the least variance that meets it
        assert e3 > e4, name


def test_the_medical_dp_run_is_pythons_row_for_row(browser) -> None:
    rows = fed_heart.run_dp(seed=1, rb=dp.seeded_rand_below(LABEL))
    assert browser["fedSource"] == "seeded"
    assert len(browser["fed"]) == len(rows) == len(fed_heart.DP_EPSILONS)
    for js, py in zip(browser["fed"], rows, strict=True):
        assert js["V"] == py.variance_per_hospital
        assert js["federated"] == py.federated, (py.epsilon, js["federated"], py.federated)
        if math.isinf(py.epsilon):
            assert js["V"] == 0 and js["ea"] == "inf"
        else:
            assert js["ea"] == py.epsilon_achieved <= py.epsilon and js["e3"] == py.epsilon_one_dishonest


def test_the_genomics_release_is_pythons_for_every_epsilon(browser) -> None:
    for e in dp.GENOMICS_EPSILONS:
        g = dp.genomics_dp(COUNTS, e, dp.seeded_rand_below(f"{LABEL}:g{e if e != int(e) else int(e)}"))
        js = browser["genomics"][str(e if e != int(e) else int(e))]
        assert js == [g["contributions"], g["variance_per_cohort"], g["epsilon_achieved"], g["epsilon_one_dishonest"]], e
        assert g["epsilon_achieved"] <= e


def test_no_page_code_calls_the_seeded_source() -> None:
    hits = []
    for p in sorted((WEB / "src").rglob("*.ts*")):
        text = p.read_text()
        for m in re.finditer(r"seededRandBelow\(", text):
            line = text[: m.start()].count("\n") + 1
            ctx = text.splitlines()[line - 1]
            # Its definition, and the one call behind the `seededLabel` option in each DP entry point.
            if "export function seededRandBelow" in ctx or "opts?.seededLabel != null ? seededRandBelow(opts.seededLabel)" in ctx:
                continue
            hits.append(f"{p.relative_to(REPO)}:{line}")
    assert not hits, f"the seeded (non-private) source is called outside the test option: {hits}"
    assert not re.search(r"seededLabel\s*:", "\n".join(p.read_text() for p in (WEB / "src").rglob("*.ts*")
                                                       if p.name not in {"clientDp.ts", "clientFedHeart.ts"})), \
        "a page module passes a seededLabel"

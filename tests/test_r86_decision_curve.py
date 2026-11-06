"""The finance page's decision curve: one estimator in two languages, at the window's rate, over one range.

WHY THIS EXISTS

The owner chose a decision curve for finance in Round 29 ("add the decision curve and cite the source"). The citation
landed and the curve stayed in the technical report, which left the repository on 2026-11-02 without the curve ever
reaching the page. Round 86 draws it in the browser from the published evaluation sample, and three things can go
wrong without any figure looking odd:

1. THE RATE. The sample is enriched to 164 laundering rows of 8,192 by drawing each class of the test window on its
   own. Weighted by its own 2 percent, net benefit at t = 0.01 reads 95.4 per 10,000 transactions where all 1,523,504
   rows of the window give 5.44 (docs/measurements/finance-decision-curve-2026-11-06.json). The estimator puts the
   window's rate back (Pfeiffer and Gail, Biometrical Journal 62(3):764-776, 2020, eq. 11).
2. TWO COPIES. `metrics.net_benefit_at_prevalence` and `web/src/lib/decisionCurve.ts` compute the same thing, and are
   asserted equal through esbuild, as the QSDC capacity is.
3. TWO OWNERS OF ONE RANGE. The BFV scorecard is fitted over 0.002 to 0.05 and held the five thresholds as its own
   literal, while the page's curve spans the same range from the config. The trainer reads the config now, and the
   published scorecard must record the config's five.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qsh.config import get_config
from qsh.models import metrics
from qsh.models.calibration import Platt

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
MEASURED = REPO / "docs" / "measurements" / "finance-decision-curve-2026-11-06.json"


def _cfg():
    return get_config().decision_curve


def _python(y: np.ndarray, risk: np.ndarray, t: float, mu: float) -> tuple[float, float, float, float]:
    """Eq. 11 written out, not imported, so a change to `metrics.py` that is not a change to the paper has to pass here.

    Returns (net benefit, standard error, TPR, FPR).
    """
    ev, ne = risk[y == 1], risk[y == 0]
    tpr, fpr = float(np.mean(ev >= t)), float(np.mean(ne >= t))
    w = t / (1 - t)
    se = np.sqrt(mu ** 2 * tpr * (1 - tpr) / len(ev) + (1 - mu) ** 2 * w ** 2 * fpr * (1 - fpr) / len(ne))
    return mu * tpr - (1 - mu) * fpr * w, float(se), tpr, fpr


def _window(rng: np.random.Generator, rows: int = 400_000, rate: float = 0.0015):
    """A synthetic window at finance's order of prevalence, with an informative but imperfect score."""
    y = (rng.random(rows) < rate).astype(int)
    logit = np.where(y == 1, rng.normal(-3.0, 1.2, rows), rng.normal(-7.0, 1.2, rows))
    return y, 1 / (1 + np.exp(-logit))


def _enriched(rng: np.random.Generator, y: np.ndarray, s: np.ndarray, events: int, non_events: int):
    """A sample drawn from each class on its own, as `prep_model_inputs.py` draws finance's."""
    idx = np.concatenate([rng.choice(np.flatnonzero(y == 1), events, replace=False),
                          rng.choice(np.flatnonzero(y == 0), non_events, replace=False)])
    return y[idx], s[idx]


# ---- the estimator ---------------------------------------------------------------------------------------------------

def test_with_the_sample_s_own_rate_it_is_plain_net_benefit():
    """The identity that makes eq. 11 a correction and not a different quantity."""
    rng = np.random.default_rng(7)
    y = (rng.random(5000) < 0.1).astype(int)
    s = rng.random(5000)
    for t in (0.002, 0.01, 0.05, 0.3):
        nb, _ = metrics.net_benefit_at_prevalence(y, s, t, float(y.mean()))
        assert abs(nb - metrics.net_benefit(y, s, t)) < 1e-15
        assert abs(nb - _python(y, s, t, float(y.mean()))[0]) < 1e-15


def test_the_window_s_rate_corrects_the_enrichment():
    """Weighted by the enriched sample's own rate, net benefit is wrong by an order of magnitude; at the window's, it
    lands inside its interval. The second half fails if the estimator is handed the sample's rate."""
    rng = np.random.default_rng(11)
    y, s = _window(rng)
    mu = float(y.mean())
    ys, ss = _enriched(rng, y, s, events=160, non_events=7840)
    for t in (0.002, 0.01, 0.05):
        truth = metrics.net_benefit(y, s, t)
        nb, se = metrics.net_benefit_at_prevalence(ys, ss, t, mu)
        naive = metrics.net_benefit(ys, ss, t)
        assert abs(nb - truth) < 3 * se, f"t={t}: corrected {nb:.6f} against the window's {truth:.6f}, se {se:.6f}"
        assert abs(naive - truth) > 10 * se, (
            f"t={t}: weighting by the sample's own rate gave {naive:.6f}, within ten standard errors of the window's "
            f"{truth:.6f}; the enrichment this test exists for is not being exercised")


def test_the_interval_covers_at_close_to_its_rate():
    """The standard error is the two binomial proportions'; 300 draws from one window check it is neither too narrow
    nor uselessly wide at three thresholds."""
    rng = np.random.default_rng(13)
    y, s = _window(rng)
    mu = float(y.mean())
    z = _cfg().interval_z
    for t in (0.002, 0.01, 0.05):
        truth = metrics.net_benefit(y, s, t)
        hits = 0
        for _ in range(300):
            ys, ss = _enriched(rng, y, s, events=160, non_events=7840)
            nb, se = metrics.net_benefit_at_prevalence(ys, ss, t, mu)
            hits += abs(nb - truth) <= z * se
        assert 0.90 <= hits / 300 <= 0.99, f"t={t}: the {_cfg().interval_coverage:.0%} interval covered {hits}/300"


def test_z_is_the_quantile_of_the_coverage():
    from scipy.stats import norm

    c = _cfg()
    assert abs(norm.ppf(0.5 + c.interval_coverage / 2) - c.interval_z) < 1e-12


# ---- the browser's copy ----------------------------------------------------------------------------------------------

def _require_browser_or_skip() -> None:
    if ESBUILD.exists():
        return
    msg = ("web/node_modules/.bin/esbuild is absent, so the browser's decision curve cannot be executed. Run `npm ci` "
           "in web/. This is a SKIP, not a pass: the two copies are unchecked.")
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg + " QSH_REQUIRE_BROWSER_GATES=1 is set, so this is a failure.")
    pytest.skip(msg)


def _browser(script: str) -> dict:
    """Bundle `decisionCurve.ts` (and the config it reads) and run `script` against it as `m`."""
    _require_browser_or_skip()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
        entry.write_text(
            f"export * from {json.dumps(str(WEB / 'src/lib/decisionCurve'))};\n"
            f"export {{ modelsFor }} from {json.dumps(str(WEB / 'src/lib/config'))};\n")
        subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
                        f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
        runner.write_text(f"import * as m from {json.dumps(bundle.as_uri())};\n{script}\n")
        out = subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True)
        return json.loads(out.stdout)


def _fixture():
    rng = np.random.default_rng(17)
    y, s = _window(rng, rows=200_000)
    ys, ss = _enriched(rng, y, s, events=120, non_events=3880)
    return ys, ss, float(y.mean())


def test_the_browser_and_python_agree():
    ys, ss, mu = _fixture()
    c = _cfg()
    grid = metrics.decision_grid(c.curves["finance_aml"]["thresholds"], c.grid_points)
    got = _browser(
        f"const risk = {json.dumps(ss.tolist())}; const y = {json.dumps(ys.tolist())};\n"
        f"const grid = m.decisionGrid({json.dumps(c.curves['finance_aml']['thresholds'])}, {c.grid_points});\n"
        f"console.log(JSON.stringify({{ grid, pts: m.decisionCurve(risk, y.map((v) => v === 1), {mu!r}, grid, "
        f"{c.interval_z!r}) }}));")
    assert len(got["grid"]) == len(grid)
    assert max(abs(a - b) / b for a, b in zip(got["grid"], grid)) < 1e-15
    for p in got["pts"]:
        nb, se, tpr, fpr = _python(ys, ss, p["t"], mu)
        assert abs(p["model"] - nb) < 1e-15 and abs(p["tpr"] - tpr) < 1e-15 and abs(p["fpr"] - fpr) < 1e-15
        assert abs((p["hi"] - p["lo"]) / 2 - c.interval_z * se) < 1e-15
        assert abs(p["treatAll"] - (mu - (1 - mu) * p["t"] / (1 - p["t"]))) < 1e-15
        assert metrics.net_benefit_at_prevalence(ys, ss, p["t"], mu) == pytest.approx((nb, se), abs=1e-16)


def test_the_platt_map_is_the_one_the_record_publishes():
    rec = json.loads((REPO / "models" / "metrics.json").read_text())["finance_aml.gbm"]["calibration"]
    platt = Platt(a=rec["a"], b=rec["b"], n_calibration=rec["n_calibration"],
                  positives_calibration=rec["positives_calibration"], on_logit=rec["on_logit"],
                  logit_clip_eps=rec["logit_clip_eps"])
    scores = [0.0, 1e-14, 1e-6, 1e-4, 0.002, 0.05, 0.386869, 0.9, 1.0]
    got = _browser(f"const cal = {json.dumps(rec)};\n"
                   f"console.log(JSON.stringify({json.dumps(scores)}.map((s) => m.plattRisk(s, cal))));")
    want = platt.apply(np.array(scores))
    assert max(abs(a - b) for a, b in zip(got, want)) < 1e-12


def test_the_page_weights_by_the_window_s_rate_and_not_the_sample_s():
    """`publishedDecisionCurve` is what the page calls, with the spec the page reads. Fails if it computes the rate from
    the rows it is handed, which is the defect the estimator exists to avoid."""
    ys, ss, _ = _fixture()
    got = _browser(
        f"const p = {json.dumps(ss.tolist())}; const y = {json.dumps(ys.tolist())};\n"
        "const spec = m.modelsFor('finance_aml').find((s) => s.key === 'finance_aml.gbm');\n"
        "const r = m.publishedDecisionCurve('finance_aml', spec, (i) => p[i], y, y.length, 0.38686859607696533);\n"
        "const none = m.publishedDecisionCurve('medical_fl', spec, (i) => p[i], y, y.length, null);\n"
        "console.log(JSON.stringify({ r, none }));")
    assert got["none"] is None, "a use case whose config declares no curve must draw none"
    c = got["r"]["curve"]
    assert (c["windowEvents"], c["windowRows"]) == (2321, 1523504)
    assert c["prevalence"] == 2321 / 1523504
    assert (c["events"], c["nonEvents"]) == (int(ys.sum()), int(len(ys) - ys.sum()))
    assert c["atMedian"]["t"] == 0.01 and c["atMedian"]["naive"] > 5 * c["atMedian"]["corrected"]
    assert abs(c["operating"]["risk"] - 0.12883530613165148) < 1e-12


# ---- one owner of the range ------------------------------------------------------------------------------------------

def test_the_scorecard_reads_the_thresholds_the_page_draws():
    declared = _cfg().curves["finance_aml"]["thresholds"]
    tree = ast.parse((REPO / "scripts" / "train" / "finance_scorecard.py").read_text())
    assigned = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "THRESHOLDS" for t in n.targets)]
    assert len(assigned) == 1
    assert not any(isinstance(n, (ast.List, ast.Tuple)) for n in ast.walk(assigned[0])), (
        "the scorecard's THRESHOLDS is a literal again: the page's decision curve and the fit would have two owners")
    assert "decision_curve" in ast.unparse(assigned[0])
    published = json.loads((REPO / "models" / "finance_aml_scorecard" / "scorecard.json").read_text())["thresholds"]
    assert published == declared, "the published scorecard was fitted over thresholds the config no longer declares"


def test_the_recorded_measurement_is_for_this_config():
    """The record is regenerated by `scripts/measure_decision_curve.py`; a changed range or grid makes it stale."""
    rec = json.loads(MEASURED.read_text())
    c = _cfg()
    entry = c.curves["finance_aml"]
    assert rec["inputs"]["thresholds"] == entry["thresholds"] and rec["inputs"]["grid_points"] == c.grid_points
    grid = metrics.decision_grid(entry["thresholds"], c.grid_points)
    assert [r["t"] for r in rec["results"]["window"]] == pytest.approx(grid, rel=1e-15)
    inside = rec["results"]["window_inside_sample_interval"]
    assert inside["of"] == len(grid) and inside["count"] >= 0.9 * len(grid)
    med = rec["results"]["at_median_threshold"]
    assert med["lo"] <= med["window"] <= med["hi"]

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
4. THE STUDENT'S MAP (Round 86, later). The private-inference figure draws the CKKS student's curve from its decrypted
   logits, through a Platt map fitted on the student's own probability. Written by a separate script, the next record
   of a retrained student would carry the old weights' map forward onto new logits, so the trainer writes it into the
   student's record itself. And the map is fitted on the probability, as the teacher's is, because the model panel
   hands every published graph's probability to the same function.
"""

from __future__ import annotations

import ast
import json
import os
import re
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
MEASURED_STUDENT = REPO / "docs" / "measurements" / "finance-student-decision-curve-2026-11-06.json"
RECORDS = REPO / "models" / "metrics.json"


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


@pytest.mark.parametrize("key", ["finance_aml.gbm", "finance_aml.student"])
def test_the_platt_map_is_the_one_the_record_publishes(key):
    rec = json.loads(RECORDS.read_text())[key]["calibration"]
    platt = Platt(a=rec["a"], b=rec["b"], n_calibration=rec["n_calibration"],
                  positives_calibration=rec["positives_calibration"], on_logit=rec["on_logit"],
                  logit_clip_eps=rec["logit_clip_eps"])
    scores = [0.0, 1e-14, 1e-6, 1e-4, 0.002, 0.05, 0.386869, 0.5, 0.9, 1.0]
    got = _browser(f"const cal = {json.dumps(rec)};\n"
                   f"console.log(JSON.stringify({json.dumps(scores)}.map((s) => m.plattRisk(s, cal))));")
    want = platt.apply(np.array(scores))
    assert max(abs(a - b) for a, b in zip(got, want)) < 1e-12


def test_every_curve_model_s_record_carries_a_map_on_its_probability():
    """What `publishedDecisionCurve` reads, for each model the config names: a Platt map fitted on the model's own
    probability (`on_logit` true, the one convention every call site can hand over), and the event count of the window
    its sample was drawn from, which must be the window its record was scored on."""
    records = json.loads(RECORDS.read_text())
    for key in _cfg().curves["finance_aml"]["model_keys"]:
        rec = records[key]
        cal = rec.get("calibration")
        assert cal and cal["method"] == "platt", f"{key}: no Platt map, so the page withholds its curve"
        assert cal["on_logit"] is True, (
            f"{key}: a map fitted on something other than the probability, which the model panel hands every graph's "
            "probability to, would be a different function there")
        corp = cal["corp_calibrated"]
        assert (corp["positives"], corp["n"]) == (2321, 1523504)
        assert corp["n"] == rec.get("n_test", rec.get("n_eval")), f"{key}: the map was scored on another window"
        assert cal["n_calibration"] == 710969 and cal["positives_calibration"] == 801, "not the validation window"


def test_the_student_s_record_states_nothing_measured_on_the_teacher():
    """`Platt.as_dict()` publishes the teacher's Round 49 isotonic comparison as `why_not_isotonic`. The student's record
    replaces it with its own measurement, so neither model's record carries the other's numbers."""
    cal = json.loads(RECORDS.read_text())["finance_aml.student"]["calibration"]
    assert "Round 49" not in cal["why_not_isotonic"] and "0.00142218" not in cal["why_not_isotonic"]
    iso = cal["isotonic_measured"]
    for v in (iso["brier"], iso["log_loss"], cal["brier_calibrated"], cal["log_loss_calibrated"]):
        assert f"{v:.8f}" in cal["why_not_isotonic"], f"{v} is not the number the sentence states"
    word = "below" if iso["brier"] < cal["brier_calibrated"] else "above"
    assert f"Brier {iso['brier']:.8f} is {word} Platt's" in cal["why_not_isotonic"]


def test_the_trainer_records_the_student_s_map_with_its_weights():
    """The defect this prevents: `record_metrics` carries a field the new payload lacks forward from the old record, so a
    retrain writing only the fidelity block would leave the previous student's map beside new weights, and the page
    would apply it to their logits. The trainer's own call must carry the map."""
    tree = ast.parse((REPO / "scripts" / "train" / "finance_aml.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "record_metrics"
             and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "finance_aml.student"]
    assert len(calls) == 1, "the trainer no longer records the student in one call"
    payload = calls[0].args[1]
    keys = [k.value for k in getattr(payload, "keys", []) if isinstance(k, ast.Constant)]
    assert "calibration" in keys, "the student's record is written without its calibration map"
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "student_calibration")
    src = ast.unparse(fn)
    assert "fit_platt(pv, yv, on_logit=True)" in src, "the student's map is no longer fitted on its probability"


def test_the_figure_draws_the_student_s_curve_from_decrypted_scores():
    """The private-inference figure's curve is a claim about the DECRYPTED output. Fed the plaintext logits it would draw
    the same picture and stop being one."""
    src = (WEB / "src" / "lib" / "clientFHE.ts").read_text()
    call = src[src.index("publishedDecisionCurve(held.usecase"):]
    call = call[:call.index(";")]
    assert "plot.points[i].enc" in call and ".plain" not in call, call
    assert "Math.exp(-plot.points[i].enc)" in call, "the decrypted logit must reach the map as a probability"


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


def test_the_student_s_curve_reads_its_own_record():
    """The same function with the student's spec: the student's map, and the student's record's window, with its
    operating point at a probability of 0.5, where its folded threshold puts logit 0."""
    ys, ss, _ = _fixture()
    cal = json.loads(RECORDS.read_text())["finance_aml.student"]["calibration"]
    got = _browser(
        f"const p = {json.dumps(ss.tolist())}; const y = {json.dumps(ys.tolist())};\n"
        "const spec = m.modelsFor('finance_aml').find((s) => s.key === 'finance_aml.student');\n"
        "const r = m.publishedDecisionCurve('finance_aml', spec, (i) => p[i], y, y.length, 0.5);\n"
        "console.log(JSON.stringify({ r }));")
    c = got["r"]["curve"]
    assert (c["windowEvents"], c["windowRows"]) == (2321, 1523504)
    assert abs(c["operating"]["risk"] - 1 / (1 + np.exp(cal["b"]))) < 1e-12
    want = Platt(a=cal["a"], b=cal["b"], n_calibration=1, positives_calibration=1, on_logit=True).apply(ss)
    t = c["atMedian"]["t"]
    nb, _, _, _ = _python(ys, want, t, 2321 / 1523504)
    assert abs(c["atMedian"]["corrected"] - nb) < 1e-15


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


@pytest.mark.parametrize("path", [MEASURED, MEASURED_STUDENT], ids=["teacher", "student"])
def test_the_recorded_measurement_is_for_this_config(path):
    """The records are regenerated by `scripts/measure_decision_curve.py`; a changed range, grid, map or pin makes one
    stale."""
    rec = json.loads(path.read_text())
    c = _cfg()
    entry = c.curves["finance_aml"]
    key = rec["inputs"]["model_key"]
    assert key in entry["model_keys"]
    record = json.loads(RECORDS.read_text())[key]
    assert rec["inputs"]["hf_revision"] == record["hf_revision"], f"{path.name} measured another revision of {key}"
    assert {k: record["calibration"][k] for k in rec["inputs"]["calibration"]} == rec["inputs"]["calibration"]
    assert rec["inputs"]["thresholds"] == entry["thresholds"] and rec["inputs"]["grid_points"] == c.grid_points
    grid = metrics.decision_grid(entry["thresholds"], c.grid_points)
    assert [r["t"] for r in rec["results"]["window"]] == pytest.approx(grid, rel=1e-15)
    inside = rec["results"]["window_inside_sample_interval"]
    assert inside["of"] == len(grid) and inside["count"] >= 0.9 * len(grid)
    med = rec["results"]["at_median_threshold"]
    assert med["lo"] <= med["window"] <= med["hi"]


def test_a_card_prints_the_map_the_page_applies():
    """Until Round 86 no Hub card printed a calibration map, so a reader had the graph and not the function that turns its
    output into the risks the page's decision curve plots. Each curve model's card carries its record's map."""
    import importlib.util
    import sys

    sys.path.insert(0, str(REPO / "src"))
    spec = importlib.util.spec_from_file_location("publish_hf", REPO / "scripts" / "publish_hf.py")
    publish_hf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publish_hf)
    from qsh.models import registry

    for key in _cfg().curves["finance_aml"]["model_keys"]:
        rec = registry.achieved(key)
        card = publish_hf.model_card(registry.get(key), rec)
        assert "## Calibration map" in card, f"{key}: the card prints no calibration map"
        cal = rec["calibration"]
        assert f"| `a` | {cal['a']!r} |" in card and f"| `b` | {cal['b']!r} |" in card, key
        assert r"\operatorname{logit}" in card, f"{key}: the card's formula omits the logit its map applies"
        # The Hub renders `$$...$$` and `\\(...\\)` and prints a single-dollar span as raw text, backslashes and all,
        # which is what the first calibration card published did.
        section = card[card.index("## Calibration map"):]
        section = section[:section.index("\n## ")] if "\n## " in section else section
        assert "$" not in re.sub(r"\$\$.*?\$\$", "", section, flags=re.S), f"{key}: inline math in single dollars"
        assert r"\\(q\\)" in section, f"{key}: the card's inline math is not in the Hub's delimiters"

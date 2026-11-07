"""The page scores the satellite forecaster student the way Python does, flag for flag (Round 86).

Three TypeScript modules carry the student's end-to-end evaluation in the browser, and each is held to its Python
original here by bundling it with esbuild and running it under node:

- `telemanomThreshold.ts` against `qsh.models.telemanom_threshold` (itself checked against ESA-ADB's class), on
  random series with planted anomalies: identical flags, per channel and combined;
- `esaScores.ts` against `qsh.models.esa_scores`, on random labels and flags: equal counts, scores to 1e-12;
- `forecastEvents.ts` on the published stretch: the forecast, flags and event scores the sample records, with the
  flags compared through their sha256.

The ports follow pandas' moving-average update order, numpy's pairwise summation and numpy's percentile
interpolation, because the thresholding compares smoothed errors with thresholds and a last-bit difference moves a
flag.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsh.models import registry, telemanom_threshold
from qsh.models.esa_scores import esa_event_wise

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"


def _require_browser_or_skip() -> None:
    if ESBUILD.exists():
        return
    msg = ("web/node_modules/.bin/esbuild is absent, so the browser's satellite student modules cannot be executed. "
           "Run `npm ci` in web/. This is a SKIP, not a pass: the two copies are unchecked.")
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg + " QSH_REQUIRE_BROWSER_GATES=1 is set, so this is a failure.")
    pytest.skip(msg)


def _node(module: str, script: str, data: object) -> object:
    """Bundle web/src/lib/<module>.ts, hand it `data` as `input`, run `script` and parse what it prints."""
    _require_browser_or_skip()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        bundle, runner, payload = d / "bundle.mjs", d / "run.mjs", d / "input.json"
        subprocess.run([str(ESBUILD), str(WEB / "src" / "lib" / f"{module}.ts"), "--bundle", "--format=esm",
                        "--platform=node", f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True,
                       capture_output=True)
        payload.write_text(json.dumps(data))
        runner.write_text(f"import * as m from {json.dumps(bundle.as_uri())};\nimport fs from 'fs';\n"
                          f"const input = JSON.parse(fs.readFileSync({json.dumps(str(payload))}, 'utf8'));\n"
                          f"{script}\n")
        out = subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True)
        return json.loads(out.stdout)


def test_the_thresholding_port_flags_what_the_python_flags() -> None:
    cases = []
    for seed in range(24):
        r = np.random.default_rng(seed)
        n, c = int(r.integers(1500, 7000)), int(r.integers(1, 4))
        vals = np.cumsum(r.normal(size=(n, c)), axis=0) * 0.01 + np.sin(np.arange(n)[:, None] / r.uniform(30, 400))
        err = np.abs(r.normal(scale=r.uniform(0.002, 0.05), size=(n, c)))
        for _ in range(int(r.integers(0, 6))):
            a = int(r.integers(0, n - 50))
            err[a:a + int(r.integers(1, 40)), int(r.integers(0, c))] += r.uniform(0.02, 2.0)
        floor = float(r.choice([0.0, 0.007, 0.05]))
        per, anyf = telemanom_threshold.flag(err, vals, telemanom_threshold.TelemanomConfig(min_error_value=floor))
        cases.append({"err": err.T.tolist(), "vals": vals.T.tolist(), "floor": floor,
                      "per": per.T.astype(int).tolist(), "any": anyf.astype(int).tolist()})
    got = _node("telemanomThreshold", "console.log(JSON.stringify(input.map((c) => { "
                "const r = m.flag(c.err, c.vals, {...m.DEFAULT_CONFIG, minErrorValue: c.floor}); "
                "return { any: Array.from(r.any), per: r.perChannel.map((p) => Array.from(p)) }; })));", cases)
    for i, (want, g) in enumerate(zip(cases, got, strict=True)):
        assert g["any"] == want["any"] and g["per"] == want["per"], f"case {i}: the port flags other rows"


def test_the_scorer_port_scores_what_the_python_scores() -> None:
    cases = []
    t0 = pd.Timestamp("2010-01-01")
    for seed in range(40):
        r = np.random.default_rng(seed)
        n = int(r.integers(200, 3000))
        times = pd.date_range(t0, periods=n, freq="30s")
        flags = np.zeros(n, dtype=int)
        for _ in range(int(r.integers(0, 12))):
            a = int(r.integers(0, n - 5))
            flags[a:a + int(r.integers(1, 60))] = 1
        labs = []
        for _ in range(int(r.integers(1, 10))):
            start = t0 + pd.Timedelta(milliseconds=int(r.uniform(0, n * 30 - 100) * 1000))
            end = start + pd.Timedelta(milliseconds=int(float(r.choice([0, 0.4, 7.5, 95.0, 400.0])) * 1000))
            labs.append({"ID": f"id_{int(r.integers(0, 6))}", "StartTime": start, "EndTime": end,
                         "Category": ["Anomaly", "Rare Event", "Communication Gap"][int(r.integers(0, 3))]})
        sel = ["Anomaly"] if r.random() < 0.5 else ["Rare Event", "Anomaly"]
        want = esa_event_wise(times, flags, pd.DataFrame(labs), select={"Category": sel},
                              full_range=(times[0], times[-1]))
        ms = [int(x.value // 1_000_000) for x in times]
        cases.append({"times": ms, "flags": flags.tolist(), "select": sel, "full": [ms[0], ms[-1]],
                      "labels": [{"id": lb["ID"], "startMs": int(lb["StartTime"].value // 1_000_000),
                                  "endMs": int(lb["EndTime"].value // 1_000_000), "category": lb["Category"]}
                                 for lb in labs],
                      "want": {k: want[k] for k in ("tp", "fp", "fn", "redundant_detections", "f_0.50")}})
    got = _node("esaScores", "console.log(JSON.stringify(input.map((c) => "
                "m.esaEventWise(c.times, c.flags, c.labels, c.select, c.full))));", cases)
    for i, (c, g) in enumerate(zip(cases, got, strict=True)):
        w = c["want"]
        assert (g["tp"], g["fp"], g["fn"], g["redundantDetections"]) == (w["tp"], w["fp"], w["fn"],
                                                                          w["redundant_detections"]), i
        assert abs(g["f05"] - w["f_0.50"]) < 1e-12, i


def test_the_page_reproduces_the_published_stretch_flag_for_flag() -> None:
    out_dir = REPO / registry.ARTIFACT_DIRS["satellite_comms.student"]
    sample = json.loads((out_dir / "eval_sample.json").read_text())
    weights = json.loads((out_dir / "weights.json").read_text())
    assert sample["task"] == "forecast_events" and weights["kind"] == "linear-forecaster"
    got = _node("forecastEvents", """
const f = input.sample, w = input.weights;
const s = m.scoreStretch(f, w, m.plaintextForecast(w, m.featureRows(f)));
const digest = await m.flagsDigest(s.flags);
console.log(JSON.stringify({ digest, flagged: s.flagged, per: s.flaggedPerChannel,
  a: [s.anomalies.tp, s.anomalies.fp, s.anomalies.fn, s.anomalies.f05],
  e: [s.allEvents.tp, s.allEvents.fp, s.allEvents.fn, s.allEvents.f05] }));""",
                {"sample": sample, "weights": weights})
    exp = sample["expected"]
    assert got["digest"] == exp["flags_sha256"], "the page flags other rows than the generator recorded"
    assert got["flagged"] == exp["flagged"] and got["per"] == exp["flagged_per_channel"]
    assert got["flagged"] > 0, "a stretch with nothing flagged would be reproduced by a page that flags nothing"
    a, e = exp["anomalies"], exp["all_events"]
    assert got["a"][:3] == [a["tp"], a["fp"], a["fn"]] and abs(got["a"][3] - a["f_0_5"]) < 1e-12
    assert got["e"][:3] == [e["tp"], e["fp"], e["fn"]] and abs(got["e"][3] - e["f_0_5"]) < 1e-12

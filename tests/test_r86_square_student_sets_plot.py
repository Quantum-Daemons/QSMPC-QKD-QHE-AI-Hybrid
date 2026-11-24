"""The square student's parameter-set plot draws the pre-registered record (Round 86, the owner's choice of 2026-11-23).

The medical page plots the square student's largest decrypted-logit error against its time per image for the three CKKS
parameter sets the pre-registration tried, with the pre-registered 1e-3 limit and set A, which missed it, drawn as not
offered; a run in the tab adds its own point. The figures come from `web/public/data/medical/square_student_sets.json`,
which `scripts/export_square_student_sets.py` copies from `docs/measurements/medical-square-student-2026-11-19.json`.

What could go wrong, and what holds it:

- the served file drifts from the record (a re-run, a hand edit): it must equal what the script writes now, and every
  figure must be the record's own;
- the plot calls a set offered that the card cannot run, or the reverse: the offered sets are exactly those the tracked
  weights list, with the same modulus, and exactly those the page's config carries;
- the line is not the pre-registered limit, or "not offered" does not follow from it;
- the tab's point is timed differently from the record's (keys included, or evaluation alone).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SERVED = REPO / "web" / "public" / "data" / "medical" / "square_student_sets.json"
RECORD = REPO / "docs" / "measurements" / "medical-square-student-2026-11-19.json"
WEIGHTS = REPO / "models" / "medical_breastmnist_square_student" / "weights.json"
CONFIG = REPO / "web" / "src" / "lib" / "config.defaults.json"
CHART = REPO / "web" / "src" / "components" / "SquareStudentSetsChart.tsx"
CARD = REPO / "web" / "src" / "components" / "SquareStudentCkks.tsx"
PREREG = REPO / "references" / "medical-square-student-preregistration.md"
KEY = "medical_fl.square_student"


def _script():
    spec = importlib.util.spec_from_file_location("export_square_student_sets", REPO / "scripts" / "export_square_student_sets.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _served() -> dict:
    return json.loads(SERVED.read_text(encoding="utf-8"))


def _square_spec() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    found = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("key") == KEY and "achieved" in node:
                found.append(node)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(config)
    assert len(found) == 1, f"{KEY} appears {len(found)} times in the config"
    return found[0]


def test_the_served_file_is_what_the_script_writes() -> None:
    assert _served() == _script().build()


def test_every_figure_is_the_record_s() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    sets = _served()["sets"]
    assert set(sets) == {"A", "A2", "B"} == set(record["ckks"])
    for name, figures in sets.items():
        for k, v in figures.items():
            assert record["ckks"][name][k] == v, (name, k)
        assert figures["images"] == 156
    assert {n: s["criterion_3"] for n, s in sets.items()} == record["criteria"]["3_fidelity"]


def test_the_line_is_the_pre_registered_limit_and_not_offered_follows_from_it() -> None:
    served = _served()
    assert "logit is at most $`10^{-3}`$" in PREREG.read_text(encoding="utf-8")
    assert served["limit"] == 1e-3
    for name, s in served["sets"].items():
        assert s["criterion_3"] == (s["max_abs_logit_error"] <= served["limit"] and s["decisions_differ"] == 0), name
    assert [n for n, s in served["sets"].items() if not s["criterion_3"]] == ["A"]


def test_the_offered_sets_are_the_ones_the_card_runs() -> None:
    served = _served()["sets"]
    offered = {n for n, s in served.items() if s["criterion_3"]}
    weights = json.loads(WEIGHTS.read_text(encoding="utf-8"))["ckks"]
    assert offered == set(weights)
    for name in offered:
        assert weights[name]["N"] == served[name]["N"]
        assert weights[name]["coeff_mod_bits"] == served[name]["bits"]
        assert weights[name]["scale_bits"] == served[name]["scale_bits"]
        assert sum(served[name]["bits"]) == served[name]["modulus_bits"] <= served[name]["pq_bound_bits"]
    achieved = _square_spec()["achieved"]
    assert set(achieved["ckks"]) == offered
    assert set(achieved["ckks_not_offered"]) == set(served) - offered
    for name in offered:
        assert achieved["ckks"][name]["max_abs_logit_error"] == served[name]["max_abs_logit_error"]
        assert achieved["ckks"][name]["median_s"] == served[name]["median_s"]


def test_the_record_s_times_say_where_they_were_taken() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    served = _served()
    assert record["environment"]["tenseal"] in served["timed_with"]
    assert "not in a browser" in served["timed_with"]
    assert "keys excluded" in served["timing"]


def test_the_chart_reads_the_file_and_draws_the_limit_and_the_three_kinds() -> None:
    chart = CHART.read_text(encoding="utf-8")
    assert "data/medical/square_student_sets.json" in chart
    assert re.search(r"<ReferenceLine y=\{file\.limit\}", chart)
    assert 'name="offered (circle)"' in chart and 'name="not offered (cross)"' in chart and 'name="this tab (diamond)"' in chart
    assert "filter((n) => file.sets[n].criterion_3)" in chart and "filter((n) => !file.sets[n].criterion_3)" in chart
    assert 'scale="log"' in chart
    # The caption names the other library and machine, so the tab's point is not read as the set's cost.
    assert "another" in chart and "library on another machine" in chart
    assert "file.attribution.citation" in chart and "file.attribution.licence_url" in chart


def test_the_tab_s_point_is_timed_as_the_record_times_a_set() -> None:
    card = CARD.read_text(encoding="utf-8")
    assert "<SquareStudentSetsChart run={tabPoint} />" in card
    # Encrypt, evaluate and decrypt per image, keys excluded, the median over the run's images: the record's median_s.
    assert "medianMs: median(o.runs.map((r) => r.ms.encrypt + r.ms.evaluate + r.ms.decrypt))" in card
    trainer = (REPO / "scripts" / "train" / "medical_square_student.py").read_text(encoding="utf-8")
    loop = trainer[trainer.index("for x14 in images:"):trainer.index("times.append(time.perf_counter() - t0)")]
    assert "t0 = time.perf_counter()" in loop and "ckks_vector(" in loop and ".decrypt()" in loop
    assert "generate_galois_keys" not in loop


AXIS = r"""
import { createRequire } from "node:module";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
const WEB = process.argv[2], OUT = process.argv[3];
const { build } = createRequire(WEB + "/package.json")("esbuild");
await build({ entryPoints: [resolve(WEB, "src/lib/logAxis.ts")], bundle: true, format: "esm", platform: "node",
              outfile: OUT, logLevel: "error" });
const m = await import(pathToFileURL(OUT).href);
const record = [8.17e-3, 3.06e-4, 6.14e-5];
console.log(JSON.stringify({ record: m.errorDomain(record, 1e-3), deployed: m.errorDomain([...record, 1.05e-5], 1e-3),
  preview: m.errorDomain([...record, 1.77e-5], 1e-3), edge: m.EDGE }));
"""


def test_no_point_is_drawn_on_the_error_axis_s_ends(tmp_path) -> None:
    """On the deployed site (2026-11-24) a tab run of 8 images under A2 had a largest error of 1.05e-5, and the axis,
    which began at the decade at or below the smallest point, put it on its 1e-5 floor: the diamond cut in half and its
    name struck through. Every point now sits at least a factor of 1.5 inside both ends."""
    import shutil
    import subprocess

    web = REPO / "web"
    if shutil.which("node") is None or not (web / "node_modules" / "esbuild").exists():
        msg = "node or esbuild absent, so the axis cannot be checked. This is a SKIP, not a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg + " QSH_REQUIRE_BROWSER_GATES=1 is set, so this is a failure.")
        pytest.skip(msg)
    (tmp_path / "a.mjs").write_text(AXIS)
    r = subprocess.run(["node", str(tmp_path / "a.mjs"), str(web), str(tmp_path / "b.mjs")], capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["deployed"][0] == pytest.approx(1e-6) and out["preview"][0] == pytest.approx(1e-5), out
    for name in ("record", "deployed", "preview"):
        lo, hi = out[name]
        points = [8.17e-3, 3.06e-4, 6.14e-5] + ([1.05e-5] if name == "deployed" else [1.77e-5] if name == "preview" else [])
        assert all(p / lo >= out["edge"] and hi / p >= out["edge"] for p in points), (name, out[name])

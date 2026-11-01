"""The three 8,192-point scatters draw on one canvas, in recharts' own shapes and on its own domain.

WHY THIS EXISTS (Round 85, by the owner's decision)

Finance's held-out split is 8,192 rows, and three charts in `VerificationChart.tsx` (the corpus
overlay, the encrypted-against-plaintext agreement and the decision plane) drew every row as an SVG
symbol: after one run they were 49,300 of the page's 52,700 elements, and the tab's JS heap reached
0.4 to 1 GB. The owner chose to keep every point and draw them on a canvas (`CanvasMarks.tsx`); in
headless Chrome the page then held 3,576 elements after a run, with the same axis ticks, the same
legend and a readout that still reads the point under the pointer.

WHAT IS ASSERTED

- In those three charts every `<Scatter>` is legend-only (`NO_POINTS`) or the two-point domain
  carrier (`corners(...)`), and the points reach a `<Customized component={<CanvasMarks .../>}>`, so
  no chart in the file hands recharts thousands of points again.
- `drawSymbol` traces each of the five shapes exactly as d3-shape's `symbol` does at recharts'
  default area of 64 px^2, so a cross is still a cross of the same size.
- `corners` returns the extremes of every series, and nothing for no points: an automatic domain is a
  function of the extremes alone, which is why two points stand in for 8,192.

On the base commit (6bb8657) the first assertion fails and `CanvasMarks.tsx` does not exist.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
CHART = WEB / "src" / "components" / "VerificationChart.tsx"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"

CANVAS_CHARTS = ("EncryptedAgreementChart", "DecisionPlane", "RealDataScatter")

HARNESS = r"""
import * as m from %(marks)s;
import { symbol, symbolCircle, symbolCross, symbolDiamond, symbolSquare, symbolTriangle } from %(d3)s;
const record = () => {
  const cmds = [];
  const ctx = {};
  for (const k of ["moveTo", "lineTo", "arc", "closePath", "rect", "arcTo", "bezierCurveTo", "quadraticCurveTo"])
    ctx[k] = (...a) => cmds.push([k, ...a.map((v) => (typeof v === "number" ? Math.round(v * 1e9) / 1e9 : v))]);
  return { ctx, cmds };
};
const types = { circle: symbolCircle, cross: symbolCross, diamond: symbolDiamond, square: symbolSquare, triangle: symbolTriangle };
const out = {};
for (const [name, type] of Object.entries(types)) {
  const ours = record(), theirs = record();
  m.drawSymbol(ours.ctx, name, 0, 0);
  symbol(type, 64).context(theirs.ctx)();
  const moved = record();
  m.drawSymbol(moved.ctx, name, 10, -4);
  out[name] = { ours: ours.cmds, theirs: theirs.cmds, moved: moved.cmds };
}
out.corners = m.corners([{ points: [{ x: 3, y: -1 }, { x: -2, y: 7 }] }, { points: [{ x: 9, y: 0.5 }] }], "a", "b");
out.cornersEmpty = m.corners([{ points: [] }]);
console.log(JSON.stringify(out));
"""


def _chart_body(name: str) -> str:
    text = CHART.read_text()
    m = re.search(rf"^(?:export )?function {name}\(.*?^}}\n", text, re.S | re.M)
    assert m, f"{name} is gone from VerificationChart.tsx"
    return m.group(0)


def _scatter_tags(body: str) -> list[str]:
    tags, i = [], 0
    while (i := body.find("<Scatter", i)) != -1:
        if body[i + len("<Scatter")].isalnum():          # <ScatterChart
            i += 1
            continue
        j, depth = i, 0
        while j < len(body):
            depth += {"{": 1, "}": -1}.get(body[j], 0)
            if body[j] == ">" and depth == 0:
                break
            j += 1
        tags.append(body[i:j + 1])
        i = j + 1
    return tags


@pytest.mark.parametrize("name", CANVAS_CHARTS)
def test_the_chart_hands_its_points_to_the_canvas_and_not_to_recharts(name: str) -> None:
    body = _chart_body(name)
    assert "<Customized component={<CanvasMarks" in body, f"{name} draws no CanvasMarks"
    tags = _scatter_tags(body)
    assert tags, f"{name} has no <Scatter> left to key its legend"
    heavy = [t[:80] for t in tags if "data={NO_POINTS}" not in t and "data={corners(" not in t]
    assert not heavy, f"{name} still hands recharts its points: {heavy}"
    assert "<Tooltip" not in body, f"{name} keeps a recharts Tooltip with nothing left to hover"


def _require_node_or_skip() -> None:
    missing = [n for n, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()),
                               ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = f"{' and '.join(missing)} absent, so the canvas code cannot run. This is a SKIP, not a pass."
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


@pytest.fixture(scope="module")
def traced(tmp_path_factory) -> dict:
    _require_node_or_skip()
    d = tmp_path_factory.mktemp("r85canvas")
    entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
    entry.write_text("export { drawSymbol, corners } from "
                     + json.dumps(str(WEB / "src/components/CanvasMarks.tsx")) + ";\n"
                     + "export { symbol, symbolCircle, symbolCross, symbolDiamond, symbolSquare, symbolTriangle } "
                     + "from " + json.dumps(str(WEB / "node_modules/d3-shape/src/index.js")) + ";\n")
    subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node", "--jsx=automatic",
                    f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
    runner.write_text(HARNESS % {"marks": json.dumps(bundle.as_uri()), "d3": json.dumps(bundle.as_uri())})
    proc = subprocess.run(["node", str(runner)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"the canvas module did not run:\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("shape", ["circle", "cross", "diamond", "square", "triangle"])
def test_each_shape_is_traced_as_d3_shape_traces_it_at_recharts_area(traced, shape: str) -> None:
    t = traced[shape]
    assert t["ours"] == t["theirs"], (shape, t["ours"], t["theirs"])
    # And it moves with its point rather than staying at the origin.
    assert t["moved"] != t["ours"]


def test_corners_are_the_extremes_and_none_for_no_points(traced) -> None:
    assert traced["corners"] == [{"a": -2, "b": -1}, {"a": 9, "b": 7}]
    assert traced["cornersEmpty"] == []

"""Three UI defects found by pressing every button in a headless browser (Round 82 walk, 2026-10-30).

WHY THIS EXISTS

- The benchmark CSV took its header from the first row only. PapaParse's `unparse` reads the field
  names of the first object, so a field that only later rows carry (a measured row's `max_abs_error`,
  a published model's `revision`) was dropped from the file while the JSON export kept it: 4 to 7
  columns short on every use case.
- The agent panel's "Download results (JSON)" said why it was disabled only in a title tooltip, which
  touch screens and most keyboard users never see. The simulator's own exports already said it on the
  page.
- The 3D view of the comparison circuit got a new `features` array on every render. The page re-renders
  on every pipeline event, so one Start rebuilt the WebGL renderer 20 to 26 times, until the browser
  logged "Too many active WebGL contexts". The scene's cleanup also left each context to the garbage
  collector.

Each test below fails on the code before the fix (checked by running it against that tree).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"


def _code(path: Path) -> str:
    """The file without whole-line comments, so a check on behaviour is not satisfied by a comment."""
    return "\n".join(line for line in path.read_text().splitlines() if not line.lstrip().startswith(("//", "*", "/*")))


def test_the_csv_header_is_the_union_of_every_rows_fields() -> None:
    if not ESBUILD.exists() or shutil.which("node") is None:
        pytest.skip("esbuild or node absent; run `npm ci` in web/")
    src = (WEB / "src" / "lib" / "export.ts").read_text()
    assert "export function csvColumns(" in src, "export.ts no longer exports csvColumns"
    body = src[src.index("export function csvColumns("):src.index("export function downloadCSV(")]
    with tempfile.TemporaryDirectory() as tmp:
        mod = Path(tmp) / "cols.ts"
        mod.write_text(body)
        out = Path(tmp) / "cols.mjs"
        subprocess.run([str(ESBUILD), str(mod), "--format=esm", f"--outfile={out}"], check=True,
                       capture_output=True)
        runner = Path(tmp) / "run.mjs"
        papa = (WEB / "node_modules" / "papaparse" / "papaparse.js").as_uri()
        runner.write_text(
            f"import {{ csvColumns }} from {json.dumps(out.as_uri())};\n"
            f"import Papa from {json.dumps(papa)};\n"
            "const rows = [{path: 'plain', latency_ms: 1}, {path: 'ckks', latency_ms: 9, max_abs_error: 1e-9},\n"
            "              {path: 'published', revision: 'abc', latency_ms: 2}];\n"
            "const cols = csvColumns(rows);\n"
            "const csv = Papa.unparse(rows, { columns: cols });\n"
            "console.log(JSON.stringify({ cols, header: csv.split(/\\r?\\n/)[0] }));\n")
        res = json.loads(subprocess.run(["node", str(runner)], check=True, capture_output=True,
                                        text=True).stdout)
    assert res["cols"] == ["path", "latency_ms", "max_abs_error", "revision"], res
    assert res["header"] == "path,latency_ms,max_abs_error,revision", res
    code = _code(WEB / "src" / "lib" / "export.ts")
    assert "Papa.unparse(rows, { columns: csvColumns(rows) })" in code, (
        "downloadCSV no longer passes the union of the rows' fields as the CSV columns")


def test_the_agent_download_says_why_it_is_disabled_on_the_page() -> None:
    code = _code(WEB / "src" / "components" / "AgenticAIPanel.tsx")
    reason = "Run the in-browser agent first - there is no result to download yet"
    visible = re.search(r"\{!run && \(\s*<p[^>]*>\s*" + re.escape(reason), code)
    assert visible, "the disabled download's reason is not rendered as visible text when there is no run"


def test_the_3d_view_gets_a_stable_input_and_releases_its_contexts() -> None:
    page = _code(WEB / "src" / "pages" / "UseCasePage.tsx")
    assert not re.search(r"<Qcnn3D\s+features=\{\[", page), (
        "an inline array literal is passed to Qcnn3D: a new object every render rebuilds the WebGL view")
    m = re.search(r"<Qcnn3D\s+features=\{(\w+)\}", page)
    assert m and re.search(rf"^const {m.group(1)}\s*=", page, re.M), (
        "Qcnn3D's features must be a module-level constant")
    scene = _code(WEB / "src" / "components" / "Scene3D.tsx")
    cleanup = scene[scene.index("renderer.dispose();"):scene.index("renderer.dispose();") + 400]
    assert "renderer.forceContextLoss();" in cleanup, "the scene's cleanup does not release its WebGL context"


def test_none_registered_is_never_said_of_a_withheld_model() -> None:
    """Round 83 walk: the genomics metrics table printed `published_model_status unavailable`, the value
    also used when a registered model is held back because a retrain made its pin stale. The diagram
    rendered both as "none registered for this use case", which is false in the second case."""
    infer = _code(WEB / "src" / "lib" / "onnxInference.ts")
    assert "withheld: string[]" in infer, "the unavailable outcome no longer names withheld models"
    assert re.search(r"m\.hf_repo_id && !m\.hf_revision && m\.hf_revision_stale", infer), (
        "withheld models must be the registered ones whose pin is stale")
    compute = _code(WEB / "src" / "lib" / "clientCompute.ts")
    assert '"published_model_status", "unavailable"' not in compute
    assert '"published_model_status", "withheld"' in compute and '"published_model_status", "none"' in compute
    flows = _code(WEB / "src" / "lib" / "useCaseFlows.ts")
    assert 'status === "none"' in flows and 'status === "withheld"' in flows
    assert 'status === "unavailable"' not in flows

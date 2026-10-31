"""Round 84: every QCNN width the page offers changes the answer, and the shipped set is the trained set.

WHY THIS EXISTS

Checklist row QC34. Finance and medical offered 8 and 9 wires. The ninth wire carried no feature, and
pooling folded it into wire 0 without ever convolving it, so the two widths gave the same class on
every bundled row and identical metrics at five of five seeds: a choice that changed nothing, which
is the inert-control defect the checklist opens with. The user's decision was to remove the 9-wire
option rather than explain it, so the q9 artifacts are deleted and finance and medical ship one width.

Three properties, each of which the Round 82 tree failed:

  * any two widths a use case ships give a different class on at least one row (QC34);
  * finance and medical ship exactly one width and no q9 file is left behind in either directory;
  * a use case with one width renders no width SELECT - an option list of one is a control that
    cannot change anything;

and the copy step that serves the files learned the two lessons the deletion and the retrain taught:
a served file with no trained artifact behind it is reported, and keys the browser reads are shipped
even when the old copy lacked them.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
PUBLIC = WEB / "public" / "data"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
PIPELINE = WEB / "src" / "lib" / "usePipelineRun.ts"
PANEL = WEB / "src" / "components" / "ParameterPanel.tsx"
SHIP = REPO / "scripts" / "ship_qcnn_weights.py"


def _declared_widths() -> dict[str, list[int]]:
    m = re.search(r"export const QCNN_WIDTHS: Record<string, number\[\]> = \{(.*?)\};",
                  PIPELINE.read_text(), re.S)
    assert m, "QCNN_WIDTHS not found in usePipelineRun.ts"
    return {k: [int(n) for n in re.findall(r"\d+", v)]
            for k, v in re.findall(r"(\w+):\s*\[([^\]]*)\]", m.group(1))}


def _classes(paths: list[Path], rows: list[list[float]]) -> list[list[int]]:
    if not ESBUILD.exists():
        msg = "web/node_modules/.bin/esbuild is absent; run `npm ci` in web/. A SKIP, not a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg + " QSH_REQUIRE_BROWSER_GATES=1 makes it a failure.")
        pytest.skip(msg)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        bundle, runner = d / "qsim.mjs", d / "run.mjs"
        subprocess.run([str(ESBUILD), str(WEB / "src" / "lib" / "quantumSim.ts"), "--bundle",
                        "--format=esm", "--platform=node", f"--outfile={bundle}", "--log-level=error"],
                       cwd=WEB, check=True, capture_output=True)
        (d / "rows.json").write_text(json.dumps(rows))
        runner.write_text(
            f"import * as m from {json.dumps(bundle.as_uri())};\n"
            "import { readFileSync } from 'node:fs';\n"
            f"const rows = JSON.parse(readFileSync({json.dumps(str(d / 'rows.json'))}, 'utf8'));\n"
            f"const paths = {json.dumps([str(p) for p in paths])};\n"
            "console.log(JSON.stringify(paths.map((p) => { const w = JSON.parse(readFileSync(p, 'utf8'));\n"
            "  return rows.map((r) => m.qcnnInferTrained(r.slice(0, w.n_qubits), w).cls); })));\n")
        out = subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True)
        return json.loads(out.stdout)


def _rows(directory: str, n_features: int) -> list[list[float]]:
    """Rows in raw units drawn through the default artifact's own moments (mean + std * N(0, 1),
    seeded): the training distribution's scale, without depending on what the bundled sample holds."""
    w = json.loads((PUBLIC / directory / "qcnn_weights.json").read_text())
    g = np.random.default_rng(0)
    z = g.normal(0, 1, (200, n_features))
    return (np.asarray(w["feature_mean"]) + np.asarray(w["feature_std"]) * z).tolist()


@pytest.mark.parametrize("directory", ["finance", "medical", "telecom"])
def test_every_pair_of_shipped_widths_gives_a_different_class_somewhere(directory: str) -> None:
    paths = sorted((PUBLIC / directory).glob("qcnn_weights*.json"))
    assert paths, f"no QCNN artifact under {directory}"
    if len(paths) < 2:
        return                     # one width: nothing to choose between, so nothing can be inert
    nf = min(int(json.loads(p.read_text())["n_features"]) for p in paths)
    cls = _classes(paths, _rows(directory, nf))
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            differ = sum(a != b for a, b in zip(cls[i], cls[j], strict=True))
            assert differ > 0, (
                f"{paths[i].name} and {paths[j].name} give the same class on all {len(cls[i])} rows. "
                "Offering both is a control that changes nothing; ship one, or say on the page that "
                "they agree.")


def test_finance_and_medical_ship_one_width_and_no_q9_artifact_remains() -> None:
    widths = _declared_widths()
    assert widths.get("finance_aml") == [8] and widths.get("medical_fl") == [8], widths
    assert widths.get("telecom_ids") == [7, 8], widths
    left = sorted(str(p.relative_to(REPO)) for p in
                  [*REPO.glob("models/qcnn_*_q9/*"), *PUBLIC.glob("*/qcnn_weights_q9.json")])
    assert not left, f"the removed 9-wire option left files behind: {left}"


def test_a_use_case_with_one_width_renders_no_width_select() -> None:
    src = PANEL.read_text()
    assert "qcnnWidths(usecase).length > 1 && (" in src, (
        "the QCNN width SELECT is rendered for a single width; a one-option select is a control "
        "that cannot change anything")
    assert "qcnnWidths(usecase).length > 0 && (" not in src


def _ship_module(tmp: Path):
    spec = importlib.util.spec_from_file_location("ship_qcnn_weights_under_test", SHIP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.REPO, mod.PUBLIC = tmp, tmp / "web" / "public" / "data"
    return mod


def _run(mod, *argv: str) -> int:
    old = sys.argv
    sys.argv = ["ship_qcnn_weights.py", *argv]
    try:
        return mod.main()
    finally:
        sys.argv = old


def test_the_copy_step_reports_a_served_width_with_no_trained_artifact(tmp_path: Path) -> None:
    (tmp_path / "models" / "qcnn_finance").mkdir(parents=True)
    (tmp_path / "models" / "qcnn_finance" / "weights.json").write_text(json.dumps({"theta": [1]}))
    served = tmp_path / "web" / "public" / "data" / "finance"
    served.mkdir(parents=True)
    (served / "qcnn_weights.json").write_text(json.dumps({"theta": [1]}))
    orphan = served / "qcnn_weights_q9.json"
    orphan.write_text(json.dumps({"theta": [2]}))
    mod = _ship_module(tmp_path)
    assert _run(mod, "--check") == 1, "an orphaned served width passed --check"
    assert _run(mod) == 1, "an orphaned served width passed the copy step"
    assert orphan.exists(), "the copy step deleted a served file; that is a decision for git rm"


def test_the_copy_step_ships_the_keys_the_browser_reads(tmp_path: Path) -> None:
    trained = {"theta": [1], "n_val": 3, "circuit": {"name": "hur-su4"}, "encoding_scale": 0.5}
    (tmp_path / "models" / "qcnn_telecom").mkdir(parents=True)
    (tmp_path / "models" / "qcnn_telecom" / "weights.json").write_text(json.dumps(trained))
    served = tmp_path / "web" / "public" / "data" / "telecom"
    served.mkdir(parents=True)
    (served / "qcnn_weights.json").write_text(json.dumps({"theta": [0]}))     # an old, trimmed copy
    mod = _ship_module(tmp_path)
    assert _run(mod) == 0
    got = json.loads((served / "qcnn_weights.json").read_text())
    assert got == {"theta": [1], "circuit": {"name": "hur-su4"}, "encoding_scale": 0.5}, (
        f"the served copy is {got}: the trimming must survive (no n_val) and the keys "
        "qcnnInferTrained reads must arrive even though the old copy lacked them")

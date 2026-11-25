"""The browser's exact Shapley values for the finance square network equal the Python mirror's (Round 86, F7b).

`web/src/lib/quadraticShap.ts` and `qsh.ai.explainability.quadratic_shap` compute the same closed form; the Python
one is held to Shapley's definition by brute force in tests/test_explainability.py. Here the two are run on the page's
own square network, when its file is present, on rows the page ships.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / "esbuild"
NETWORK = WEB / "public" / "data" / "finance" / "square_student_weights.json"
ROWS = WEB / "public" / "data" / "finance" / "student_heldout.json"


def _network() -> dict:
    if NETWORK.exists():
        return json.loads(NETWORK.read_text())
    rng = np.random.default_rng(3)
    return {"w1": rng.normal(size=(16, 114)).tolist(), "b1": rng.normal(size=16).tolist(),
            "w2": rng.normal(size=(2, 16)).tolist(), "b2": rng.normal(size=2).tolist(),
            "centre": [0.0] * 114, "scale": [1.0] * 114, "clip": 10.0}


def test_the_browser_and_python_attributions_agree() -> None:
    if shutil.which("node") is None or not ESBUILD.exists():
        pytest.skip("node or web/node_modules is not installed")
    from qsh.ai.explainability import quadratic_shap

    w = _network()
    rows = np.asarray(json.loads(ROWS.read_text())["samples"][:8], dtype=np.float64)
    z = np.clip((rows - np.asarray(w["centre"])) / np.where(np.asarray(w["scale"]) == 0, 1, w["scale"]),
                -w["clip"], w["clip"])
    base = z.mean(0)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "in.json").write_text(json.dumps({"w": w, "z": z.tolist(), "base": base.tolist()}))
        (d / "run.mjs").write_text(f"""
import {{ createRequire }} from "node:module";
import {{ readFileSync, writeFileSync }} from "node:fs";
const {{ build }} = createRequire({json.dumps(str(WEB / "package.json"))})("esbuild");
await build({{ entryPoints: [{json.dumps(str(WEB / "src" / "lib" / "quadraticShap.ts"))}], bundle: true, format: "esm",
  platform: "node", outfile: {json.dumps(str(d / "b.mjs"))}, logLevel: "error" }});
const m = await import({json.dumps((d / "b.mjs").as_uri())});
const {{ w, z, base }} = JSON.parse(readFileSync({json.dumps(str(d / "in.json"))}, "utf8"));
const out = z.map((row) => m.quadraticShap(w.w1, w.b1, w.w2, w.b2, row, [], 114, base));
writeFileSync({json.dumps(str(d / "out.json"))}, JSON.stringify(out));
""")
        r = subprocess.run(["node", str(d / "run.mjs")], cwd=WEB, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr
        got = json.loads((d / "out.json").read_text())
    for row, js in zip(z, got, strict=True):
        py = quadratic_shap(w["w1"], w["b1"], w["w2"], w["b2"], row, baseline=base, top_k=114)
        want = {a["feature"]: a["contribution"] for a in py.top}
        have = {a["feature"].replace("feature_", ""): a["contribution"] for a in js["top_features"]}
        assert len(have) == 114 and js["efficiency_gap"] < 1e-8, js["efficiency_gap"]
        for k, c in want.items():
            assert abs(have[k.replace("feature_", "")] - round(c, 6)) <= 1e-6, (k, c)
        assert abs(js["margin"] - round(py.margin, 6)) <= 1e-6 and js["method"] == py.method

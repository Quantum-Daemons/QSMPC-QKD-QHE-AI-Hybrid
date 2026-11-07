"""The minority class's precision-recall curve and sensitivity, as the page computes them (Round 86, fact-check C7).

The medical verification's curve scored class 1, which on BreastMNIST is "normal, benign", the majority (73%), so the
curve sat on a 0.73 baseline. `web/src/lib/minorityPr.ts` adds class 0's own curve, malignant there, and the number of
class-0 rows the page's decision rule detects. This bundles it with esbuild, runs it under node on random samples, and
holds it to a Python reference. On the published ResNet-18 sample the page shows 30 of 42 malignant rows detected
(sensitivity 0.714), which the Python reference also gives from the served int8 graph.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"


def _reference(labels: np.ndarray, preds: np.ndarray, score: np.ndarray, stride: int):
    n = len(labels)
    pos0 = int((labels == 0).sum())
    if not (0 < pos0 < n / 2):
        return None
    order = np.argsort(score, kind="stable")
    pr, tp = [], 0
    for k in range(n):
        tp += int(labels[order[k]] == 0)
        if k % stride == 0 or k == n - 1:
            pr.append((tp / pos0, tp / (k + 1)))
    return {"pr": pr, "prevalence": pos0 / n, "detected": int(((labels == 0) & (preds == 0)).sum()), "total": pos0}


def test_the_page_computes_the_minority_curve_the_reference_does() -> None:
    if not ESBUILD.exists():
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail("web/node_modules/.bin/esbuild is absent and QSH_REQUIRE_BROWSER_GATES=1 is set")
        pytest.skip("web/node_modules/.bin/esbuild is absent, so the module cannot run. This is a SKIP, not a pass.")
    cases = []
    for seed in range(30):
        r = np.random.default_rng(seed)
        n = int(r.integers(20, 400))
        labels = (r.random(n) < r.uniform(0.15, 0.85)).astype(int)
        # Distinct scores, so the two sorts cannot order ties differently.
        score = r.permutation(n).astype(float) + labels * r.uniform(0, n / 2)
        preds = (score > np.median(score)).astype(int)
        stride = max(1, n // 200)
        cases.append({"labels": labels.tolist(), "preds": preds.tolist(), "score": score.tolist(), "stride": stride,
                      "want": _reference(labels, preds, score, stride)})
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        subprocess.run([str(ESBUILD), str(WEB / "src" / "lib" / "minorityPr.ts"), "--bundle", "--format=esm",
                        "--platform=node", f"--outfile={d / 'b.mjs'}", "--log-level=error"], check=True, cwd=WEB)
        (d / "in.json").write_text(json.dumps(cases))
        (d / "run.mjs").write_text(
            f"import {{ minorityPr }} from {json.dumps((d / 'b.mjs').as_uri())};\nimport fs from 'fs';\n"
            f"const cs = JSON.parse(fs.readFileSync({json.dumps(str(d / 'in.json'))}, 'utf8'));\n"
            "console.log(JSON.stringify(cs.map((c) => minorityPr(c.labels.length, c.labels, c.preds, "
            "(i) => c.score[i], c.stride))));\n")
        got = json.loads(subprocess.run(["node", str(d / "run.mjs")], check=True, capture_output=True,
                                        text=True).stdout)
    assert any(c["want"] is None for c in cases) and any(c["want"] is not None for c in cases)
    for i, (c, g) in enumerate(zip(cases, got, strict=True)):
        w = c["want"]
        if w is None:
            assert g is None, i
            continue
        assert (g["detected"], g["total"]) == (w["detected"], w["total"]) and abs(g["prevalence"] - w["prevalence"]) < 1e-12
        # Arrays, not pytest.approx: CI's pytest rejects nested sequences in approx, which the local one accepted.
        got_pr = np.asarray([(p["recall"], p["precision"]) for p in g["pr"]], dtype=float)
        want_pr = np.asarray(w["pr"], dtype=float)
        assert got_pr.shape == want_pr.shape and float(np.abs(got_pr - want_pr).max()) < 1e-12, i

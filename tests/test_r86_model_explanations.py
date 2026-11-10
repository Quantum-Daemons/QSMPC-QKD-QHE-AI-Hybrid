"""The figures that explain each page's model (Round 86, the owner's request).

WHY THIS EXISTS. The AMLworld paper shows its data as a graph because a laundering pattern spans accounts, and a reviewer of a medical image model, a polygenic score or an intrusion detector expects a saliency map, a locus plot or a confusion matrix. Each figure here is computed from data
the page already ships or from the run's own result, so what must hold is that the shipped data is the right data, carries its licence, and that the code that draws it computes what the caption says.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
GRAPHS = json.loads((WEB / "public/data/finance/transaction_graphs.json").read_text())
LOCI = json.loads((WEB / "public/data/genomics/pgs000072_loci.json").read_text())
PRS = json.loads((WEB / "public/data/genomics/pgs000072_breast_cancer.json").read_text())


def _node(body: str, tmp_path: Path) -> object:
    """Bundles the browser's own modules and runs `body`, which prints JSON."""
    script = WEB / ".tmp-explanations.mjs"
    script.write_text(f"""
import {{ build }} from "esbuild";
import {{ readFileSync }} from "node:fs";
await build({{ stdin: {{ contents: 'export * from "./src/lib/transactionGraph"; export * from "./src/lib/cnnSaliency"; export * from "./src/lib/studentConfusion";', resolveDir: {json.dumps(str(WEB))}, loader: "ts" }},
  bundle: true, format: "esm", platform: "node", outfile: {json.dumps(str(tmp_path / "b.mjs"))}, logLevel: "error" }});
const m = await import({json.dumps((tmp_path / "b.mjs").as_uri())});
{body}
""")
    try:
        out = subprocess.run(["node", script.name], cwd=WEB, check=True, capture_output=True, text=True, timeout=180)
    finally:
        script.unlink()
    return json.loads(out.stdout)


def test_the_graphs_are_the_right_rows_and_carry_their_licence() -> None:
    assert GRAPHS["licence"] == "CDLA-Sharing-1.0" and GRAPHS["licence_url"] == "https://cdla.dev/sharing-1-0/" and GRAPHS["attribution"] and GRAPHS["modifications"]
    shapes = [e["shape"] for e in GRAPHS["exemplars"]]
    assert shapes == ["fan-out", "fan-in", "gather-scatter", "bipartite", "cycle", "chain"]
    for e in GRAPHS["exemplars"]:
        assert 4 <= len(e["edges"]) <= 12 and all(x["laundering"] == 1 for x in e["edges"]), e["shape"]
        assert [x["minute"] for x in e["edges"]] == sorted(x["minute"] for x in e["edges"]), "numbered in time order"
    weights = json.loads((REPO / "models/finance_aml_mlp_large/weights.json").read_text())
    rows = GRAPHS["scored_rows"]
    assert [r["row_index_in_test_window"] for r in rows] == weights["fixtures"]["row_index_in_test_window"], "the rows the published MLP can score"
    assert [r["label"] for r in rows] == weights["fixtures"]["labels"]
    for r in rows:
        assert len(r["edges"]) <= 30 and r["edges"][r["scored_edge"]]["laundering"] == r["label"], "the scored edge carries the row's label"
        assert all(re.fullmatch(r"[A-Z]+", n) for n in r["nodes"]), "account identifiers are replaced by letters"


def test_every_layout_keeps_its_accounts_apart_and_in_the_box(tmp_path: Path) -> None:
    bad = _node(f"""
const d = JSON.parse(readFileSync({json.dumps(str(WEB / "public/data/finance/transaction_graphs.json"))}, "utf8"));
const out = [];
const check = (name, g, l, w, h) => {{ const pts = g.nodes.map((n) => l[n]);
  pts.forEach((p, i) => {{ if (!p || p.x < 11 || p.x > w - 11 || p.y < 11 || p.y > h - 11) out.push(name + " node " + g.nodes[i] + " is out of the box");
    pts.forEach((q, j) => {{ if (j > i && Math.hypot(p.x - q.x, p.y - q.y) < 22) out.push(name + " nodes " + g.nodes[i] + " and " + g.nodes[j] + " touch"); }}); }}); }};
for (const e of d.exemplars) {{ const {{ layout, h }} = m.layoutExemplar(e, 240, 180); check(e.shape, e, layout, 240, h); }}
for (const [i, r] of d.scored_rows.entries()) {{ const {{ layout, h }} = m.layoutScored(r, 720, 300); check("scored " + i, r, layout, 720, h); }}
console.log(JSON.stringify(out));
""", tmp_path)
    assert bad == []


def test_the_browsers_cnn_is_the_published_student(tmp_path: Path) -> None:
    sample = json.loads((REPO / "models/medical_breastmnist_student/eval_sample.json").read_text())
    size = int(len(sample["samples"][0]) ** 0.5)
    pred = _node(f"""
const w = JSON.parse(readFileSync({json.dumps(str(REPO / "models/medical_breastmnist_student/weights.json"))}, "utf8"));
const s = JSON.parse(readFileSync({json.dumps(str(REPO / "models/medical_breastmnist_student/eval_sample.json"))}, "utf8"));
const size = {size};
console.log(JSON.stringify(s.samples.map((f) => {{ const l = m.cnnLogits(w, Array.from({{ length: size }}, (_, y) => f.slice(y * size, (y + 1) * size))); return l.indexOf(Math.max(...l)); }})));
""", tmp_path)
    accuracy = float(np.mean(np.asarray(pred) == np.asarray(sample["labels"])))
    assert accuracy == pytest.approx(sample["expected"]["accuracy"], abs=1e-9), "the same predictions the published graph gave when the sample was built"


def test_the_confusion_matrix_reproduces_the_recorded_accuracy(tmp_path: Path) -> None:
    held = WEB / "public/data/telecom/student_heldout.json"
    weights = WEB / "public/data/telecom/student_weights.json"
    out = _node(f"""
console.log(JSON.stringify(m.confusion(JSON.parse(readFileSync({json.dumps(str(held))}, "utf8")), JSON.parse(readFileSync({json.dumps(str(weights))}, "utf8")))));
""", tmp_path)
    assert out["total"] == 600 and sum(map(sum, out["matrix"])) == 600
    assert out["correct"] / out["total"] == pytest.approx(json.loads(held.read_text())["expected"]["plaintext_accuracy"], abs=1e-6)


def test_the_loci_are_the_scored_variants() -> None:
    assert [x["rsid"] for x in LOCI["loci"]] == PRS["keys"] and [x["beta"] for x in LOCI["loci"]] == pytest.approx(PRS["weights"], abs=1e-9)
    assert all(x["pos"] > 0 and x["chr"] in {str(c) for c in range(1, 23)} for x in LOCI["loci"])
    assert LOCI["licence"].startswith("CC-BY-4.0") and LOCI["attribution"] and LOCI["modifications"]


def test_each_page_shows_its_figure() -> None:
    page = (WEB / "src/pages/UseCasePage.tsx").read_text()
    assert 'usecase === "finance_aml" && <TransactionGraphCard' in page
    assert 'usecase === "genomics" && <PrsLocusCard />' in page and 'usecase === "telecom_ids" && <ConfusionCard />' in page
    assert "report.explanation && <CnnOcclusion" in (WEB / "src/components/MpcCnnPanel.tsx").read_text()

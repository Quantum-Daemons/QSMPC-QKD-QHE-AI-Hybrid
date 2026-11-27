"""The DeFi page's QCNN on Aave v3 liquidations (Round 86, batch 8; checklist QC54), held to its pre-registration.

`references/defi-qcnn-aave-v3-preregistration.md` and its addenda fixed the task, the inputs, the arms and two criteria
before the archive was downloaded. What is held here, on the committed record and the two page files:

  - the record says both criteria hold, and the export check behind criterion 2 covered every test row;
  - the eight inputs are the account's history before the borrow, as checked on the table after the pre-registration
    (no count decreases along an account's borrows; each liquidation raises the next borrow's count);
  - the comparators the card names as above the shipped seed are exactly the rows that score above it;
  - the page's sample carries each row's eight inputs and its label, and nothing else (no account, transaction or other
    identifier), with the corpus's licence and credit;
  - the tab's re-score: the page's own tree contraction and average precision, bundled and run in node on the shipped
    rows, give the sample's recorded average precision; and that average precision is scikit-learn's, ties included;
  - the DeFi page mounts the card, and the card says the circuit is classically simulable.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
RECORD = REPO / "docs" / "measurements" / "defi-qcnn-aave-v3-2026-11-27.json"
CMP = WEB / "public" / "data" / "defi" / "qcnn_comparison.json"
CIRCUIT = WEB / "public" / "data" / "defi" / "defi_qcnn.json"
HISTORY = REPO / "docs" / "measurements" / "defi-inputs-prior-history-2026-11-27.json"

HARNESS = r"""
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
const [WEB, PLAN] = process.argv.slice(2);
const plan = JSON.parse(readFileSync(PLAN, "utf8"));
const { build } = createRequire(join(WEB, "package.json"))("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-defi-qcnn-"));
const out = join(dir, "b.mjs");
const seal = join(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { averagePrecision } from ${JSON.stringify(join(WEB, "src/lib/averagePrecision"))};
                      export { clearReadout } from ${JSON.stringify(join(WEB, "src/lib/clientQcnnTreeCkks"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/"}' },
  plugins: [{ name: "stand-ins", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" }, () => ({ contents: `export default ${JSON.stringify(seal)};`, loader: "js" }));
    b.onResolve({ filter: /\.(wasm|mjs)\?url$|qsh_mpc\.js$/ }, (a) => ({ path: a.path, namespace: "stub" }));
    b.onLoad({ filter: /.*/, namespace: "stub" }, () => ({ contents: "export default '';", loader: "js" }));
  } }],
});
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });
const res = {};
if (plan.ties) res.ties = plan.ties.map(({ s, y }) => m.averagePrecision(s, y));
if (plan.circuit) {
  const c = JSON.parse(readFileSync(plan.circuit, "utf8")), w = c.weights;
  const net = { maps: c.tree.maps, readout: c.tree.readout, scale: w.scale, bias: w.bias, feature_mean: w.feature_mean,
                feature_std: w.feature_std, encoding_scale: w.encoding_scale ?? 1, n_qubits: w.n_qubits };
  const r = m.clearReadout(net, c.rows);
  res.sampleAp = m.averagePrecision(Array.from(r, (v) => v * w.scale + w.bias), c.labels);
}
console.log(JSON.stringify(res));
process.exit(0);
"""


def _node(tmp_path: Path, plan: dict) -> dict:
    if not (WEB / "node_modules" / ".bin" / "esbuild").exists() or shutil.which("node") is None:
        pytest.skip("node or web/node_modules is absent")
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "h.mjs").write_text(HARNESS)
    p = subprocess.run(["node", str(tmp_path / "h.mjs"), str(WEB), str(tmp_path / "plan.json")], cwd=WEB,
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_the_record_meets_both_criteria_on_every_test_row() -> None:
    r = json.loads(RECORD.read_text())
    c = r["criteria"]
    assert c["1_signal"] and c["met"], c
    ex = c["2_export_check"]
    assert ex["criterion_met"] and ex["seed"] == r["arms"]["qcnn8"]["shipped_seed"]["seed"]
    assert ex["test_rows"] == r["inputs"]["counts"]["test"] == 182_351
    assert ex["page_vs_state_vector"]["max_abs_readout_difference"] <= 1e-5
    assert ex["page_vs_state_vector"]["decisions_differing"] == 0
    assert r["arms"]["qcnn8"]["shipped_seed"]["test_ap"] >= r["twice_prevalence"]


def test_the_inputs_are_the_account_history_before_the_borrow() -> None:
    """Checked on the table after the pre-registration (`scripts/data/check_defi_inputs_are_prior_history.py`)."""
    h = json.loads(HISTORY.read_text())
    assert h["features"] == json.loads(RECORD.read_text())["inputs"]["features"]
    assert all(v["decreases"] == 0 and v["increases"] > 0 for v in h["1_counts_along_each_account"].values())
    after = h["2_after_each_liquidation"]
    assert after["count_not_higher"] == 0 and after["count_one_or_more_higher"] == after["with_a_later_borrow"] > 0
    first = h["3_first_borrow_with_a_liquidation_counted"]
    assert first["accounts_liquidated_later"]["share"] < 0.01 and h["prior_history"] is True


def test_the_comparators_named_above_the_shipped_seed_are_the_ones_above_it() -> None:
    r = json.loads(RECORD.read_text())
    ship = r["arms"]["qcnn8"]["shipped_seed"]["test_ap"]
    above = sorted(k for k, v in r["arms"].items() if k not in ("qcnn8", "qcnn4", "noent") and v["test_ap"] > ship)
    assert r["criteria"]["comparators_above_the_shipped_seed"] == above
    site = json.loads(CMP.read_text())
    assert len(site["comparators_above_the_shipped_seed"]) == len(above)
    assert site["criteria_met"] is True and "No quantum advantage is claimed" in site["statement"]


def test_the_page_sample_carries_inputs_and_labels_and_nothing_else() -> None:
    c = json.loads(CIRCUIT.read_text())
    meta_keys = {"_generated_by", "preregistration", "record", "task", "features", "seed", "weights", "tree", "rows_about",
                 "rows", "labels", "sample", "figures", "export_check", "sources", "licence", "licence_url", "attribution",
                 "modifications"}
    assert set(c) == meta_keys, set(c) ^ meta_keys
    assert c["features"] == json.loads(RECORD.read_text())["inputs"]["features"] and len(c["features"]) == 8
    # The addendum's identifiers, raw time and label columns are never inputs.
    assert not set(c["features"]) & {"user", "pool", "reserve", "timestamp", "timeDiff", "status", "id"}
    assert all(len(row) == 8 and all(isinstance(v, float | int) and not isinstance(v, bool) for v in row) for row in c["rows"])
    assert len(c["rows"]) == len(c["labels"]) == c["sample"]["rows"] and set(c["labels"]) <= {0, 1}
    assert c["sample"]["positive"] == sum(c["labels"])
    # An account, a pool, a reserve and a transaction are 0x-prefixed hex on chain; none is anywhere in the file.
    assert not re.search(r"0x[0-9a-fA-F]{8,}", json.dumps(c)), "the page's file holds a hex identifier"
    assert c["licence"] == "CC-BY-4.0" and "figshare.31058368" in c["attribution"] and c["modifications"]


def test_the_tab_reproduces_the_sample_average_precision(tmp_path) -> None:
    c = json.loads(CIRCUIT.read_text())
    got = _node(tmp_path, {"circuit": str(CIRCUIT)})["sampleAp"]
    assert abs(got - c["sample"]["average_precision"]) <= 1e-9, (got, c["sample"]["average_precision"])


def test_the_tab_average_precision_is_scikit_learns_ties_included(tmp_path) -> None:
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(5)
    cases = []
    for n, levels in ((50, 4), (400, 12), (2000, 2000)):
        y = (rng.random(n) < 0.2).astype(int)
        y[0] = 1
        s = np.round(rng.random(n) * levels) / levels + 0.3 * y          # ties at every level but the last case
        cases.append({"s": s.tolist(), "y": y.tolist()})
    got = _node(tmp_path, {"ties": cases})["ties"]
    for case, g in zip(cases, got, strict=True):
        assert abs(g - average_precision_score(case["y"], case["s"])) <= 1e-12


def test_the_defi_page_mounts_the_card_and_it_says_what_the_circuit_is() -> None:
    page = (WEB / "src" / "pages" / "UseCasePage.tsx").read_text()
    assert "{lending && <DefiQcnnCard />}" in page
    card = (WEB / "src" / "components" / "DefiQcnnCard.tsx").read_text()
    assert "{d.statement}" in card and "trained and not encrypted" in card
    assert "averagePrecision" in card and "clearReadout" in card

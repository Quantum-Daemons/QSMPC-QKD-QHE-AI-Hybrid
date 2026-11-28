"""The telecom page's square network file is the pre-registered run's, and the page's module scores it as the run did.

`web/public/data/telecom_nidd/square_network.json` (scripts/export_telecom_square_network.py, Round 86, batch 12; the
owner's choice of 2026-11-28 that the network replace the linear student on the page's encrypted 5G-NIDD path) carries
the exported seed's weights, the parameter sets the criteria offered, the record's figures and the run's 2,048 test
flows, raw. Held here, without the data the run read: every figure is the record's; the flows are the record's draw
(its class counts); and the page's own CKKS module, on the file's flows scaled as the page scales them, decides as the
plaintext network does and scores the accuracy the run recorded, under the set the page runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
PAGE = WEB / "public" / "data" / "telecom_nidd" / "square_network.json"
RECORD = REPO / "docs" / "measurements" / "telecom-nidd-square-student-2026-11-28.json"


def _both() -> tuple[dict, dict]:
    return json.loads(PAGE.read_text()), json.loads(RECORD.read_text())


def test_the_page_file_is_the_records() -> None:
    page, r = _both()
    c, arms = r["criteria"], r["arms"]["square"]
    assert c["adopted"] and list(page["ckks"]) == c["sets_offered"]
    # B, not the first set offered: the owner's choice of 2026-11-28, after A2's largest logit error was found to vary
    # from run to run across the page's 1e-3 rule (references/telecom-nidd-square-student-preregistration-addendum-spread.md).
    assert page["default_set"] == "B" and "B" in c["sets_offered"] and "addendum-spread" in page["default_set_note"]
    assert page["export_seed"] == r["export_seed"] and page["classes"] == r["data"]["classes"]
    assert page["inputs"] == r["data"]["input_names"] and page["n_in"] == len(page["inputs"]) == 30
    f = page["figures"]
    test = [arms[s]["test"]["flamby_metric"] for s in sorted(arms)]
    assert f["control_test_flamby"] == r["control"]["test"]["flamby_metric"]
    assert f["test_flamby_min"] == min(test) and abs(f["test_flamby_mean"] - sum(test) / len(test)) < 5e-7
    assert f["export_seed_test"] == arms[str(r["export_seed"])]["test"]
    assert f["export_bootstrap"] == [r["export_bootstrap"]["low"], r["export_bootstrap"]["high"]]
    for name, s in r["ckks"]["sets"].items():
        assert f["ckks"][name] == {k: s[k] for k in ("max_abs_error", "decisions_differ", "encrypted_accuracy")}
    for name, s in r["browser"]["sets"].items():
        assert f["browser"][name]["slowest_total_ms"] == s["slowest_total_ms"]
    held = page["held_out"]
    assert len(held["raw"]) == len(held["labels"]) == r["ckks"]["rows"] == 2048
    assert all(len(row) == 30 for row in held["raw"])
    assert held["class_counts"] == r["ckks"]["class_counts"]
    assert page["licence"] == "CC-BY-4.0" and "5G-NIDD" in page["attribution"]


HARNESS = r"""
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
const WEB = process.argv[2], OUT = process.argv[3], PAGE = process.argv[4];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const wasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export * from ${JSON.stringify(resolve(WEB, "src/lib/clientFHESquareTabular"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  plugins: [{ name: "wasm-url", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(wasm)};`, loader: "js" }));
  } }],
});
const m = await import(`file://${OUT}`);
const w = JSON.parse(readFileSync(PAGE, "utf8"));
const raw = w.held_out.raw, k = w.classes.length;
const z = new Float64Array(raw.length * w.n_in);
raw.forEach((x, r) => z.set(m.scaleRow(w, x), r * w.n_in));
const r = await m.runSquareTabular(w, z, raw.length, w.default_set);
let right = 0, clearDiffer = 0;
for (let i = 0; i < r.rows; i++) {
  const enc = m.argmaxAt(r.logits, k * i, k), plain = m.argmaxAt(r.plain, k * i, k);
  if (enc === w.held_out.labels[i]) right++;
  const row = Array.from(r.plain.subarray(k * i, k * i + k)).sort((a, b) => b - a);
  if (enc !== plain && row[0] - row[1] > 2 * r.maxAbsError) clearDiffer++;
}
console.log(JSON.stringify({ rows: r.rows, maxAbsError: r.maxAbsError, decisionsDiffer: r.decisionsDiffer, clearDiffer,
                             accuracy: right / r.rows }));
"""


def test_the_pages_module_scores_the_shipped_flows_as_the_run_recorded() -> None:
    if not (WEB / "node_modules" / ".bin" / "esbuild").exists():
        msg = "web/node_modules is absent, so the page's CKKS module cannot run. This is a SKIP, not a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    page, r = _both()
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "h.mjs").write_text(HARNESS)
        out = subprocess.run(["node", str(Path(tmp) / "h.mjs"), str(WEB), str(Path(tmp) / "b.mjs"), str(PAGE)],
                             check=True, capture_output=True, text=True, cwd=WEB, timeout=900)
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["rows"] == 2048 and got["maxAbsError"] <= 1e-3 and got["clearDiffer"] == 0, got
    assert round(got["accuracy"], 6) == r["ckks"]["sets"][page["default_set"]]["encrypted_accuracy"], got

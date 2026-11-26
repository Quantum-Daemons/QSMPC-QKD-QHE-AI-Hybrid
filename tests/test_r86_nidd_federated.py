"""5G-NIDD's two base stations, federated, as pre-registered (Round 86, batch 6; the owner's choice of 2026-11-26).

references/telecom-nidd-federated-preregistration.md fixed the arms, the metric and three criteria before
scripts/train/telecom_nidd_federated.py ran. This holds the page's file to the run's record, the criteria to the figures
they read, and the global model to the masked sum of the two stations' last local models, bit for bit in the masks'
fixed point. Then it runs the page's own path in node against the real node-seal, as
tests/test_r86_nidd_student_under_ckks.py runs the student's: the replayed last aggregation with fresh X25519 keys, and
the global model under CKKS over the student's 2,048 held-out flows, criterion 3 included.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
SITE = WEB / "public" / "data" / "telecom_nidd" / "federated.json"
RECORD = REPO / "docs" / "measurements" / "telecom-nidd-federated-2026-11-26.json"
PREREG = "references/telecom-nidd-federated-preregistration.md"
KEYS = ("station_1", "station_2")


@pytest.fixture(scope="module")
def site() -> dict:
    return json.loads(SITE.read_text())


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text())


def test_the_record_is_the_preregistered_run(record: dict, site: dict) -> None:
    from test_markdown_math_uses_the_house_style import FROZEN

    script = (REPO / "scripts" / "train" / "telecom_nidd_federated.py").read_text()
    pinned = next(ast.literal_eval(n.value) for n in ast.parse(script).body if isinstance(n, ast.Assign)
                  and any(getattr(t, "id", None) == "PREREG_SHA256" for t in n.targets))
    assert record["preregistration"] == site["preregistration"] == {"file": PREREG, "sha256": FROZEN[PREREG]}
    assert pinned == FROZEN[PREREG]
    assert record["pooled"]["is_the_student"] and record["pooled"]["reproduces_the_students_record"]
    student = json.loads((REPO / "docs" / "measurements" / "telecom-nidd-student-2026-11-25.json").read_text())
    assert record["pooled"]["union_test"]["macro_f1"] == student["arms"]["direct"]["test_macro_f1"]
    assert record["pooled"]["l2"] == student["arms"]["direct"]["l2"]
    assert record["standardisation"]["holds"] and record["standardisation"]["exchanged_in_the_clear_per_station"] == 73
    assert record["local_step_is_the_students_objective"] is True
    assert record["federated"]["masks_cancelled_exactly_in_every_message"] is True
    assert record["federated"]["rounds"] == 40 and record["federated"]["local_iterations"] == 20


def test_the_page_carries_the_records_figures(record: dict, site: dict) -> None:
    c = site["comparison"]
    assert c["pooled"] == {"test": {k: record["pooled"]["test"][k]["macro_f1"] for k in KEYS},
                           "flamby_metric": record["pooled"]["flamby_metric"]}
    assert c["federated"] == {"test": {k: record["federated"]["test"][k]["macro_f1"] for k in KEYS},
                              "flamby_metric": record["federated"]["flamby_metric"]}
    assert c["alone"] == [{"test": {k: a["test"][k]["macro_f1"] for k in KEYS}, "flamby_metric": a["flamby_metric"]}
                          for a in record["alone"]]
    assert site["per_round"] == [{"round": p["round"], **{k: p[f"{k}_test_macro_f1"] for k in KEYS}}
                                 for p in record["federated"]["per_round"]]
    assert site["criteria"] == record["criteria"]
    assert len(site["per_round"]) == site["federated"]["rounds"] == 40


def test_the_criteria_follow_from_the_figures(record: dict) -> None:
    fed, pooled = record["federated"]["flamby_metric"], record["pooled"]["flamby_metric"]
    alone = [a["flamby_metric"] for a in record["alone"]]
    cr = record["criteria"]
    assert cr["1_federated_above_each_station_alone"] == all(fed > a for a in alone)
    assert cr["2_within_0_01_of_pooled"] == (pooled - fed <= 0.01)
    assert cr["federated_minus_pooled"] == round(fed - pooled, 6)
    assert cr["federated_minus_best_alone"] == round(fed - max(alone), 6)
    for arm in [record["pooled"], record["federated"], *record["alone"]]:
        assert arm["flamby_metric"] == round(float(np.mean([arm["test"][k]["macro_f1"] for k in KEYS])), 6)
    assert record["two_by_two_test_macro_f1"] == [[a["test"][k]["macro_f1"] for k in KEYS] for a in record["alone"]]


def test_the_global_model_is_the_masked_sum_of_the_last_local_models(site: dict) -> None:
    from qsh.ai.federated import SecureAggregator, dequantise, quantise

    f = site["federated"]
    flat = [np.concatenate([np.ravel(m["W"]), m["b"]]) for m in f["last_local"]]
    want = np.concatenate([np.ravel(f["W"]), f["b"]])
    acc = sum(quantise(f["weights"][s] * flat[s], 2).astype(np.uint64) for s in (0, 1))
    assert np.array_equal(dequantise(acc % (1 << 32)), want)
    agg = SecureAggregator(n_clients=2)                    # fresh keys: the masks cancel whatever they are
    shares = [agg.mask_update(s, f["weights"][s] * flat[s], 7) for s in (0, 1)]
    assert np.array_equal(SecureAggregator.aggregate(shares), want)
    assert abs(sum(f["weights"]) - 1) < 1e-12 and np.all(np.abs(want * (1 << 20) - np.round(want * (1 << 20))) == 0)


def test_the_held_out_flows_are_named_by_station(record: dict, site: dict) -> None:
    st = np.asarray(site["heldout_station"])
    held = json.loads((WEB / "public" / "data" / "telecom_nidd" / "student_heldout.json").read_text())
    assert len(st) == len(held["rows"]) == record["heldout"]["rows"] == 2048 and set(st.tolist()) == {0, 1}
    assert [int(np.sum(st == s)) for s in (0, 1)] == [record["heldout"]["by_station"][k]["rows"] for k in KEYS]
    assert site["shipped_rows_accuracy_by_station"] == record["heldout"]["by_station"]


def test_the_cards_text_states_its_limits() -> None:
    card = (WEB / "src" / "components" / "NiddFederatedCard.tsx").read_text()
    for phrase in ("hide the updates from the aggregator only", "in the clear", "not post-quantum",
                   "no differential privacy is applied", "cross-station reading"):
        assert phrase in card.replace("&apos;", "'"), phrase


HARNESS = r"""
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
const WEB = process.argv[2], OUT = process.argv[3];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const wasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { niddFederated } from ${JSON.stringify(resolve(WEB, "src/lib/realData"))};
export { encryptedHeldOutAgreement } from ${JSON.stringify(resolve(WEB, "src/lib/clientFHE"))};
export { secaggClientSeeds, secaggKeyPairs, secureAggregate } from ${JSON.stringify(resolve(WEB, "src/lib/clientQFL"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  plugins: [{ name: "wasm-url", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(wasm)};`, loader: "js" }));
  } }],
});
globalThis.fetch = async (url) => {
  const p = join(WEB, "public", String(url).replace(/^\/+/, ""));
  try { const t = readFileSync(p, "utf8"); return { ok: true, status: 200, json: async () => JSON.parse(t) }; }
  catch { return { ok: false, status: 404, json: async () => { throw new Error("404"); } }; }
};
const m = await import(`file://${OUT}`);
const fed = await m.niddFederated();
if (!fed) { console.log(JSON.stringify(null)); process.exit(0); }
const flat = (x) => [...x.W.flat(), ...x.b];
const kp = m.secaggKeyPairs(2);
const pks = kp.map((k) => k.publicKey);
const seeds = await Promise.all(kp.map((k, c) => m.secaggClientSeeds(c, k.privateKey, pks)));
const { aggregate } = m.secureAggregate(fed.lastLocal.map(flat), fed.weights, seeds, 0);
const want = flat(fed.global);
const bitExact = aggregate.filter((v, i) => v === want[i]).length;
const r = await m.encryptedHeldOutAgreement(fed.held);
let decidedFlips = 0;
r.plot.points.forEach((q) => { if (!q.agree && Math.abs(q.plain) > 2 * r.batch.maxAbsError) decidedFlips++; });
const per = [0, 1].map((s) => {
  const pts = r.plot.points.filter((_, i) => fed.station[i] === s);
  return { rows: pts.length, encrypted: pts.filter((q) => q.encCls === q.label).length / pts.length,
           plaintext: pts.filter((q) => q.plainCls === q.label).length / pts.length };
});
console.log(JSON.stringify({ coordinates: want.length, bitExact, rows: r.plot.rows, maxAbsError: r.batch.maxAbsError,
  decidedFlips, per, modelKey: fed.held.modelKey }));
"""


def _require_node_or_skip() -> None:
    missing = [name for name, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()),
                                     ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = f"{' and '.join(missing)} absent, so the page's path cannot run. Run `npm ci` in web/. This is a SKIP, not a pass."
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> dict:
    _require_node_or_skip()
    tmp = tmp_path_factory.mktemp("niddfed")
    script = tmp / "harness.mjs"
    script.write_text(HARNESS)
    proc = subprocess.run(["node", str(script), str(WEB), str(tmp / "bundle.mjs")], cwd=WEB,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, "the federated path did not run:\n" + proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out is not None, "niddFederated returned nothing"
    return out


def test_the_replay_reproduces_the_global_model_bit_for_bit(run: dict) -> None:
    assert run["coordinates"] == 333 and run["bitExact"] == 333, run
    assert run["modelKey"] == "telecom_ids.nidd_federated"


def test_criterion_3_holds(run: dict) -> None:
    assert run["rows"] == 2048
    assert run["maxAbsError"] < 1e-3, run["maxAbsError"]
    assert run["decidedFlips"] == 0, run


def test_the_browser_plaintext_accuracy_by_station_is_the_records(run: dict, record: dict) -> None:
    for s, key in enumerate(KEYS):
        assert run["per"][s]["rows"] == record["heldout"]["by_station"][key]["rows"]
        assert abs(run["per"][s]["plaintext"] - record["heldout"]["by_station"][key]["accuracy"]) <= 5e-7

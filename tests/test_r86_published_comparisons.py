"""Each page's published-figure table says what its verdicts say, from figures that agree with their record (Round 86).

`web/src/lib/publishedComparisons.ts` builds the table every use-case page shows: the figure as published, this
project's offline figure, this run, and one of five verdicts by a rule written there. Held here, on the module bundled
and run in node with the shipped registry:

1. every registry model with a published baseline has a recorded relation, so a new model cannot fall through to "no
   relation recorded", and no relation names a model the registry does not have;
2. FLamby's figures in the module are the ones the verification log's reproduction table records, published and
   offline, so the two cannot drift apart;
3. the verdicts come out as their rules say: seed 1's FedAvg (0.7005) is reproduced, inside two of FLamby's seed
   standard deviations; ResNet-18 is reproduced inside one of this project's; Telemanom has no spread; the finance
   gradient boosting is a reference; genomics is not comparable; a FedAvg 0.1 away differs;
4. this run's column reports the model's own re-scoring, outcome first.
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
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"

RUNNER = r"""
const m = await import(%(bundle)s);
const cfg = JSON.parse(%(config)s);
const fake = (mean) => ({ seed: 1, centres: ["Cleveland", "Hungary", "Switzerland", "Long Beach"], rounds: 15,
  mean: { alone: [0.73, 0.64, 0.6, 0.64], federated: mean, pooled: 0.72, initial: 0.5 } });
const out = { relation: Object.keys(m.MODEL_RELATION), flamby: m.FLAMBY_HEART, rows: {} };
for (const [uc, models] of Object.entries(cfg.models)) {
  const verifications = models.map((s) => ({ modelKey: s.key, selected: false, outcome: "PASS", metric: s.metric?.name,
    score: 0.5, expected: 0.5, n: 10, split: "test" }));
  out.rows[uc] = m.modelRows(models, verifications).map((r) => [r.figure.split(":")[0], r.verdict, r.thisRun]);
}
out.otherMetric = m.modelRows([cfg.models.medical_fl[0]], [{ modelKey: cfg.models.medical_fl[0].key, selected: true, outcome: "PASS", metric: "accuracy", score: 0.859, expected: 0.859, n: 156, split: "test" }])[0].thisRun;
out.qcnn = {};
for (const uc of ["finance_aml", "medical_fl", "telecom_ids", "genomics"]) {
  out.qcnn[uc] = {
    before: m.qcnnRows(uc, { metric: "accuracy", value: 0.9535 }, null),
    after: m.qcnnRows(uc, { metric: "accuracy", value: 0.9535 }, { label: "accuracy", headline: 0.9535, rows: 86, matchesRecord: true }),
  };
}
out.qcnnKeys = Object.keys(m.QCNN_REFERENCES);
out.fedAt7005 = m.flambyRows(fake(0.7005)).map((r) => [r.figure, r.verdict]);
out.fedAt785 = m.flambyRows(fake(0.785)).map((r) => [r.figure, r.verdict]);
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> dict:
    missing = [n for n, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()), ("node", shutil.which("node")))
               if not ok]
    if missing:
        msg = f"{' and '.join(missing)} absent, so the module cannot run. This is a SKIP, not a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    d = tmp_path_factory.mktemp("cmp")
    entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
    entry.write_text(f"export * from {json.dumps(str(WEB / 'src/lib/publishedComparisons'))};\n")
    subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node", f"--outfile={bundle}",
                    "--log-level=error"], cwd=WEB, check=True, capture_output=True)
    config = (WEB / "src/lib/config.defaults.json").read_text()
    runner.write_text(RUNNER % {"bundle": json.dumps(bundle.as_uri()), "config": json.dumps(config)})
    proc = subprocess.run(["node", str(runner)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_every_model_with_a_published_figure_has_a_relation(run) -> None:
    cfg = json.loads((WEB / "src/lib/config.defaults.json").read_text())
    with_baseline = {m["key"] for ms in cfg["models"].values() for m in ms if m["metric"].get("baseline") is not None}
    every = {m["key"] for ms in cfg["models"].values() for m in ms}
    rel = set(run["relation"])
    assert with_baseline <= rel, f"models with a published figure and no relation: {sorted(with_baseline - rel)}"
    assert rel <= every, f"relations for models the registry does not have: {sorted(rel - every)}"


def test_flamby_figures_are_the_logs(run) -> None:
    log = (REPO / "docs/verification-log.md").read_text()
    row = next(line for line in log.splitlines() if line.startswith("| results, five seeds (sample standard deviation) |"))
    nums = [float(x) for x in re.findall(r"0\.\d{3}", row)]
    f = run["flamby"]
    published = f["alone"] + [f["fedavg"]["mean"], f["fedavg"]["sd"], f["fedyogi"]["mean"], f["fedyogi"]["sd"], f["pooled"]]
    offline = f["offline"]["alone"] + [f["offline"]["fedavg"]["mean"], f["offline"]["fedavg"]["sd"], f["offline"]["pooled"]]
    assert nums == published + offline, (nums, published + offline)


def test_the_verdicts_follow_their_rules(run) -> None:
    v = {uc: {k: verdict for k, verdict, _ in rows} for uc, rows in run["rows"].items()}
    # 0.901 lies 1.1 sd above the mean of seven from-scratch runs (docs/measurements/medical-resnet18-seeds-2026-11-22.json);
    # it was "reproduced" while the record carried six runs of the ImageNet-pretrained variant.
    assert v["medical_fl"]["medical_fl.resnet18"] == "differs"
    assert v["medical_fl"]["medical_fl.resnet50"] == "reference"
    assert v["medical_fl"]["medical_fl.student"] == "not comparable"
    assert v["satellite_comms"]["satellite_comms.telemanom"] == "no spread"
    assert v["finance_aml"]["finance_aml.gbm"] == "reference"
    # Multi-PNA+EU itself, measured here since Round 86 on this project's windows, not the paper's split: a reference,
    # as the gradient booster is, under both protocols.
    assert v["finance_aml"]["finance_aml.gnn"] == "reference"
    assert v["finance_aml"]["finance_aml.gnn_causal"] == "reference"
    assert v["genomics"]["genomics.prs"] == "not comparable"
    assert all(x == "not comparable" for x in v["telecom_ids"].values()), v["telecom_ids"]
    fed = dict(run["fedAt7005"])
    assert fed["FedAvg, 15 rounds, FLamby's metric"] == "reproduced"
    assert fed["FedYogi, FLamby's metric"] == "not comparable"
    assert dict(run["fedAt785"])["FedAvg, 15 rounds, FLamby's metric"] == "differs"   # +0.100, beyond 2 x 0.028


def test_this_run_reports_the_models_own_rescoring_and_names_its_metric(run) -> None:
    # The fake re-scoring carries each model's own metric, so no row should add "not this row's metric"; the
    # deployed medical page showed an unnamed accuracy beside an AUC row until this was named.
    for rows in run["rows"].values():
        for _, _, this_run in rows:
            assert this_run.startswith("PASS: ") and " 0.5000 on 10 rows of its sample (test)" in this_run, this_run
            assert "not this row's metric" not in this_run, this_run
    assert run["otherMetric"].endswith(", not this row's metric"), run["otherMetric"]


def test_each_trained_qcnn_has_labelled_literature_beside_it(run) -> None:
    """Checklist CS13, the owner's choice of 2026-11-24: same-split classical baselines for both techniques (the QCNN's
    is logistic regression, in its run's log line), and literature figures beside them only as labelled references.
    Every page that offers a trained QCNN carries at least one; none is a reproduction, since no published figure is a
    QCNN on the page's data and split; each names its source; and this run's column follows the QCNN run."""
    offered = set(re.search(r'QCNN_USECASES = new Set\(\[([^\]]+)\]\)',
                            (WEB / "src/lib/usePipelineRun.ts").read_text()).group(1).replace('"', "").replace(" ", "").split(","))
    assert set(run["qcnnKeys"]) == offered, (run["qcnnKeys"], offered)
    want = {"finance_aml": ["not comparable"], "medical_fl": ["reference", "reference"], "telecom_ids": ["not comparable"]}
    for uc, verdicts in want.items():
        before, after = run["qcnn"][uc]["before"], run["qcnn"][uc]["after"]
        assert [r["verdict"] for r in before] == verdicts, (uc, before)
        for r in before:
            assert r["figure"] == "QCNN (trained, not encrypted): accuracy" and r["offline"] == "0.9535", r
            assert re.search(r"doi:10\.\d{4,}/|arXiv:\d{4}\.\d{4,5}", r["source"]), r["source"]
            assert r["why"] and r["thisRun"].startswith("not run in this tab"), r
        for r in after:
            assert r["thisRun"] == "re-scored 0.9535 on 86 rows of the trainer's test split, equal to the record", r
    assert run["qcnn"]["genomics"]["before"] == [], "genomics ships no trained QCNN"
    # Two references can share a figure (medical's two), so the card keys a row by figure and source together: keyed
    # on the figure alone, the preview of 2026-11-24 kept a stale third row once the circuit's record loaded.
    pairs = [(r["figure"], r["source"]) for r in run["qcnn"]["medical_fl"]["before"]]
    assert len(pairs) == len(set(pairs)) == 2, pairs
    card = (WEB / "src/components/PublishedComparisonCard.tsx").read_text()
    assert "<tr key={`${r.figure} | ${r.source}`}>" in card, "the card must key a row by figure and source"

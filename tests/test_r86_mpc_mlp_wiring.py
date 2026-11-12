"""The published finance MLP under MPC is on the page, one model and one switch (Round 86, the owner's choices).

WHY THIS EXISTS. The browser path in `clientHybridMlp.ts` is checked against the trainer's logits by `npm run check:hybrid-mlp`, but nothing there sees the page: a removed control, a fetch that skips the digest check, or a step gone from CI would each leave that check green. The owner
removed model-size selectors in Round 20 because they invited comparison of shapes nothing had trained, so this is one control for one model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
COMPUTE = (WEB / "src/lib/clientCompute.ts").read_text()
CONFIG = json.loads((WEB / "src/lib/config.defaults.json").read_text())


def _models() -> list[dict]:
    found: list[dict] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if str(node.get("key", "")).startswith("finance_aml."):
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(CONFIG)
    return found


def test_one_published_mlp_and_one_control() -> None:
    keys = [m["key"] for m in _models()]
    assert "finance_aml.mlp_large" in keys and "finance_aml.mlp_medium" not in keys
    controls = (WEB / "src/components/PipelineControls.tsx").read_text()
    assert 'id="chk-mpcmlp"' in controls and "sel-mpcmlp" not in controls, "one switch, not a choice of sizes"
    assert re.search(r'usecase === "finance_aml" && \(\s*<label className="muted" title="Runs one held-out row through the published MLP', controls), "offered on the finance page only"
    assert not (REPO / "models/finance_aml_mlp_medium").exists() or not list((REPO / "models/finance_aml_mlp_medium").glob("*.json"))


def test_the_page_fetches_the_pinned_weights_and_checks_them_before_reading() -> None:
    body = COMPUTE[COMPUTE.index("async function runMlpUnderMpc"):]
    body = body[:body.index("\n}\n")]
    fetch, check, parse = body.index("resolve/${spec.hf_revision}/weights.json"), body.index("await checkPinned("), body.index("asJson<MlpWeights>(bytes)")
    assert fetch < check < parse, "the digest is checked before anything reads the file"
    assert '"it is not pinned to a revision"' in body and "did not run" in body and "did not finish" in body, "each way it can end is reported"
    spec = next(m for m in _models() if m["key"] == "finance_aml.mlp_large")
    assert spec["file_sha256"]["weights.json"], "the digest the page checks against is published in the config"


def test_the_run_config_reaches_the_pipeline() -> None:
    hook = (WEB / "src/lib/usePipelineRun.ts").read_text()
    assert 'if (mpcMlp && DEMO && usecase === "finance_aml") c.mpc_mlp = true;' in hook
    assert re.search(r'usecase === "finance_aml" && config\.mpc_mlp\) await runMlpUnderMpc\(stage, seed, emit, metric, artifacts\)', COMPUTE)
    flows = (WEB / "src/lib/useCaseFlows.ts").read_text()
    assert "hidden: (c) => !c.mpcMlp" in flows and "MLP under MPC · did not finish" in flows


def test_the_panel_states_what_the_run_is() -> None:
    panel = (WEB / "src/components/MpcMlpPanel.tsx").read_text()
    for needle in ("notional", "semi-honest", "statistically", "2,048", "Multi-PNA+EU"):
        assert needle in panel, needle
    assert "report.result" in panel and "xgboost_same_features" in panel
    assert "artifacts?.mpc_mlp ? <MpcMlpPanel" in (WEB / "src/pages/UseCasePage.tsx").read_text()


def test_both_ci_paths_run_the_check() -> None:
    assert '"check:hybrid-mlp"' in (WEB / "package.json").read_text()
    assert "check:hybrid-mlp" in (REPO / "scripts/ci_local.sh").read_text()
    assert "check:hybrid-mlp" in (REPO / ".github/workflows/ci.yml").read_text()


def test_the_published_model_card_and_the_record_agree() -> None:
    rec = json.loads((REPO / "models/metrics.json").read_text())["finance_aml.mlp_large"]
    assert rec["hf_revision"], "pinned"
    assert rec["n_features"] == 114 and rec["n_parameters"] == 23106
    card = (REPO / "models/finance_aml_mlp_large/README.md").read_text()
    assert "2,048" in card and "23106" in card.replace(",", "")

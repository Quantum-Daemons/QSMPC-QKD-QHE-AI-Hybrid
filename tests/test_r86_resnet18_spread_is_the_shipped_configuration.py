"""The ResNet-18 card's seed spread is measured on the configuration that ships (Round 86, 2026-11-22).

WHY THIS EXISTS

`models/metrics.json` gave `medical_fl.resnet18` a six-seed spread, mean 0.8905 and sd 0.0198, and the card and the medical
page printed it beside the published run's 0.9147. Its source, `references/medical-student-collapse-investigation.md`, took
those six runs on the ImageNet-pretrained variant, which `docs/verification-log.md` recorded on 2026-08-10 is not what
ships: the published model is trained from random initialisation. The figure came back with the card republish of
2026-11-08. On the owner's choice of 2026-11-22 the shipped configuration was trained at the published seed, which
reproduces 0.914683, and at six more, none of which reaches the published baseline of 0.901. Over all seven runs the mean is
0.8761 and the sd 0.0218, so the published 0.901 is outside one standard deviation and the page's verdict for it is
"differs", not "reproduced".

WHAT IS ASSERTED

- The record's spread is the summary of all seven runs, from random initialisation, and the published seed among them
  reproduces the AUC the record publishes.
- The registry's caveat, the card and `docs/findings.md` quote that spread, and nothing quotes the old one as current.
- Seeded: the spread as it stood is refused.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "docs" / "measurements" / "medical-resnet18-seeds-2026-11-22.json"
KEY = "medical_fl.resnet18"


def spread_problems(spread: dict, record: dict) -> list[str]:
    further = [r["test_auc"] for r in record["runs"]]
    want = {"n_seeds": len(further), "mean": round(statistics.mean(further), 4), "sd": round(statistics.stdev(further), 4),
            "lo": round(min(further), 4), "hi": round(max(further), 4)}
    problems = [f"{k}: {spread.get(k)!r} where the runs give {v!r}" for k, v in want.items() if spread.get(k) != v]
    if spread.get("source") != "docs/measurements/" + RECORD.name:
        problems.append(f"source {spread.get('source')!r} is not the measurement")
    if "random" not in str(spread.get("init", "")):
        problems.append("the spread does not say it is the random-initialisation configuration")
    return problems


def test_the_record_carries_the_measured_spread() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    metrics = json.loads((REPO / "models" / "metrics.json").read_text(encoding="utf-8"))[KEY]
    assert spread_problems(metrics["seed_spread"], record) == []
    reproduced = record["published_seed_reproduced"]
    assert reproduced["matches"] and abs(reproduced["test_auc"] - metrics["auc"]) < 1e-6
    assert record["further_seeds_summary"]["at_or_above_baseline"] == sum(
        r["test_auc"] >= 0.901 for r in record["runs"] if r["seed"] != reproduced["seed"])


def test_the_caveat_the_card_and_the_findings_quote_it() -> None:
    from qsh.models import registry

    spread = json.loads((REPO / "models" / "metrics.json").read_text(encoding="utf-8"))[KEY]["seed_spread"]
    caveat = registry.get(KEY).metric.baseline_source
    card = (REPO / "models" / "medical_breastmnist_resnet18" / "README.md").read_text(encoding="utf-8")
    findings = (REPO / "docs" / "findings.md").read_text(encoding="utf-8")
    needle = f"{spread['mean']:.4f} +/- {spread['sd']:.4f}"
    assert needle in caveat and needle in card, needle
    assert f"mean {spread['mean']:.4f}, sd {spread['sd']:.4f}" in findings
    assert "six seeds average 0.8905" not in caveat + card


def test_the_check_refuses_the_spread_as_it_stood() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    as_it_stood = {"n_seeds": 6, "mean": 0.8905, "sd": 0.0198, "lo": 0.863, "hi": 0.9216,
                   "source": "references/medical-student-collapse-investigation.md"}
    assert len(spread_problems(as_it_stood, record)) >= 4

"""The heart federation's privacy audits train the page's own federation, and the plan note quotes them (2026-11-23).

WHY THIS EXISTS

The membership audit of the heart federation (`scripts/audit_membership_inference.py --target fedheart`) and the
attribute-inference audit (`scripts/audit_attribute_inference.py`) need the trained weights, which the twin's `run`
does not return, so the script re-implements its FedAvg, pooled and DP arms. A re-implementation is where an audit
drifts from what the page trains: a stream offset, sums in another order, and the audit measures another model. These
hold the re-implementation to the twin weight for weight (and the twin is held to the page by
tests/test_r86_fed_heart_twin.py), and hold the note's claims to the records.

WHAT IS ASSERTED

- For seed 0 on the shipped sample, the script's FedAvg and pooled weights are the twin's to the last bit (the twin's
  `accuracy` is wrapped to see the weights `run` scores), and its DP arms with clipping alone and at epsilon 8 are
  `run_dp`'s on the same seeded noise stream.
- Every DP arm's measured rates are under its guarantee's bound, and the record's check is the script's on the record's
  own rates; seeded: a rate above the bound is refused.
- Attribute inference: for both models the loss attack's guess is a function of the diagnosis, and no attack beats the
  imputation from all features on either split, which is what the note says.
- The note quotes both records' figures at the precision it prints them; seeded: a changed AUC is refused.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
M = REPO / "docs" / "measurements"
NOTE = REPO / "references" / "pets-evaluation-2026-11.md"
MIA, ATTR = "mia-fedheart-2026-11-23.json", "attribute-inference-fedheart-2026-11-23.json"


def _load(name: str) -> dict:
    return json.loads((M / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def mia():
    spec = importlib.util.spec_from_file_location("audit_mia", REPO / "scripts" / "audit_membership_inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def parts():
    from qsh.ai import fed_heart

    sample = json.loads(fed_heart.SAMPLE.read_text(encoding="utf-8"))
    return [(c["train"]["X"], c["train"]["y"]) for c in sample["centres"]]


def test_the_script_trains_the_twins_weights(mia, parts, monkeypatch) -> None:
    from qsh.ai import dp, fed_heart

    seen: list[list[float]] = []
    real = fed_heart.accuracy

    def spy(theta, X, y):
        seen.append(list(theta))
        return real(theta, X, y)

    monkeypatch.setattr(fed_heart, "accuracy", spy)
    fed_heart.run(seed=0)
    n = len(parts)
    # run scores the initial weights, each hospital alone, the pooled model, then FedAvg, each on n test sets.
    assert seen[-n] == mia.fedheart_federated(parts, 0)
    assert seen[-2 * n] == mia.fedheart_pooled(parts, 0)
    seen.clear()
    fed_heart.run_dp(seed=0, rb=dp.seeded_rand_below("heart-audit-twin"))
    rb = dp.seeded_rand_below("heart-audit-twin")
    assert seen[0] == mia.fedheart_dp(parts, 0, math.inf, rb)      # run_dp's first row: clipping alone
    assert seen[n] == mia.fedheart_dp(parts, 0, 8.0, rb)           # its second: epsilon 8, the stream continued


def test_every_dp_arm_is_under_its_bound(mia) -> None:
    rec = _load(MIA)
    assert set(rec["dp_bound_check"]) == {"dp_epsilon_8", "dp_epsilon_1"}
    assert rec["dp_bound_check"] == mia.dp_bound_check(rec["results"], rec["dp_delta"])
    for name, check in rec["dp_bound_check"].items():
        assert check["rows"] and check["all_hold"], name


def test_a_rate_above_the_bound_is_refused(mia) -> None:
    rec = _load(MIA)
    results = json.loads(json.dumps(rec["results"]))
    results["dp_epsilon_1"]["lira_online"]["tpr_at_fpr_0.1"] = 0.5     # the bound there is e * 0.1 + 1e-6 = 0.2718
    assert not mia.dp_bound_check(results, rec["dp_delta"])["dp_epsilon_1"]["all_hold"]


def test_no_attribute_attack_beats_imputation() -> None:
    for model, r in _load(ATTR)["results"].items():
        assert r["loss_attack_is_the_label_rule"], model
        for side in ("members", "non_members"):
            best = r[side]["imputation_all_features"]["accuracy"]
            for attack in ("loss", "model_inversion"):
                assert r[side][attack]["accuracy"] <= best, (model, side, attack)


def _pct(x: float) -> str:
    return f"{100 * x:.2f} percent"


def _acc(cell: dict) -> str:
    return f"{100 * cell['accuracy']:.2f}"


def note_problems(note: str) -> list[str]:
    flat = re.sub(r"\s+", " ", note)
    m = _load(MIA)
    r = m["results"]
    lira = {k: v["lira_online"] for k, v in r.items()}
    e1 = next(x for x in m["dp_bound_check"]["dp_epsilon_1"]["rows"] if x["attack"] == "lira_online" and x["fpr"] == 0.1)
    a = _load(ATTR)["results"]
    fm, pm = a["federated"]["members"], a["pooled"]["members"]
    lead = 100 * (a["pooled"]["members"]["loss"]["accuracy"] - a["pooled"]["non_members"]["loss"]["accuracy"])
    want = [
        f"FedAvg, LiRA AUC {lira['federated']['auc']:.3f} and {_pct(lira['federated']['tpr_at_fpr_0.001'])} at a "
        "false-positive rate of 0.1 percent",
        f"the pooled model, LiRA AUC {lira['pooled']['auc']:.3f} and {_pct(lira['pooled']['tpr_at_fpr_0.001'])}",
        f"LiRA AUC {lira['dp_clip_only']['auc']:.3f} with clipping alone, {lira['dp_epsilon_8']['auc']:.3f} at "
        f"epsilon 8 and {lira['dp_epsilon_1']['auc']:.3f} at epsilon 1",
        f"{e1['tpr']} against {e1['bound']:.4f} at a false-positive rate of 10 percent",
        f"model inversion (Fredrikson et al.) gets {_acc(fm['model_inversion'])} percent from the FedAvg model and "
        f"{_acc(pm['model_inversion'])} from the pooled one, where answering \"male\" every time gets "
        f"{_acc(a['federated']['always_male']['members'])}",
        f"the loss attack (Yeom et al.) gets {_acc(fm['loss'])} and {_acc(pm['loss'])}; imputation from all the "
        f"features gets {_acc(fm['imputation_all_features'])}",
        f"its lead of {lead:.2f} points",
    ]
    return [f"the note does not quote {w!r}" for w in want if w not in flat]


def test_the_note_quotes_the_records() -> None:
    assert note_problems(NOTE.read_text(encoding="utf-8")) == []


def test_a_changed_figure_is_refused() -> None:
    auc = _load(MIA)["results"]["federated"]["lira_online"]["auc"]
    note = NOTE.read_text(encoding="utf-8")
    assert note_problems(note.replace(f"FedAvg, LiRA AUC {auc:.3f}", "FedAvg, LiRA AUC 0.500"))

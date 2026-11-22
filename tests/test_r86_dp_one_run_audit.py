"""The one-run DP audit computes the paper's bound, finds every DP arm under its accountant, and catches the seeded
bugs (2026-11-23).

WHY THIS EXISTS

`scripts/audit_dp_one_run.py` turns canary guesses into an empirical lower bound on epsilon with Steinke, Nasr and
Jagielski's Corollary 5.4 (NeurIPS 2023, arXiv:2305.08846). A bound computed wrongly fails silently in the comfortable
direction: an audit that can never reach its accountant's epsilon passes every arm, broken or not. So the computation
is held to the values the paper's Appendix D prints, the record to its own figures, and the audit's power to the two
broken mechanisms it must catch.

WHAT IS ASSERTED

- p_value and epsilon_lower_bound reproduce Appendix D's four printed values.
- Every row's bound recomputes from its (canaries, guesses, right) at 0.05 over the grid's size; each arm's bound is
  the largest of its rows.
- Every real arm is under the epsilon its accountant states, recomputed here from the recorded variance or noise.
- Both seeded checks are caught: a lower bound above the epsilon their accountant states.
- The plan note quotes the record, each lower bound truncated to two decimals (never rounded up).
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "docs" / "measurements" / "dp-one-run-audit-2026-11-23.json"
NOTE = REPO / "references" / "pets-evaluation-2026-11.md"


@pytest.fixture(scope="module")
def one():
    pytest.importorskip("scipy")
    spec = importlib.util.spec_from_file_location("audit_dp_one_run", REPO / "scripts" / "audit_dp_one_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _arms(record: dict):
    for group in ("fedheart", "murmur", "seeded_checks"):
        for name, arm in record[group].items():
            yield group, name, arm


def test_the_bound_is_the_papers(one) -> None:
    assert one.p_value(100, 100, 75, math.log(3), 0) == pytest.approx(0.553, abs=1e-3)
    assert one.epsilon_lower_bound(100, 100, 75, 0, 0.05) == pytest.approx(0.702, abs=1e-3)
    assert one.epsilon_lower_bound(100, 100, 75, 1e-4, 0.05) == pytest.approx(0.699, abs=1e-3)
    assert one.epsilon_lower_bound(1000, 100, 75, 1e-4, 0.05) == pytest.approx(0.673, abs=1e-3)


def test_every_row_recomputes(one, record) -> None:
    p = record["design"]["each_tested_at"]
    assert p == pytest.approx(0.05 / len(record["design"]["guesses"]))
    for group, name, arm in _arms(record):
        delta = 1e-6 if (group == "fedheart" or name.startswith("fedheart")) else 1e-5
        for row in arm["rows"]:
            lb = one.epsilon_lower_bound(arm["canaries"], row["guesses"], row["right"], delta, p)
            assert round(lb, 4) == row["epsilon_lower_bound"], (name, row)
        assert arm["epsilon_lower_bound"] == max(r["epsilon_lower_bound"] for r in arm["rows"]), name


def test_every_dp_arm_is_under_its_accountant(record) -> None:
    from qsh.ai import dp, fed_heart

    dim, unit = 14, 2 ** fed_heart.DP_FIXED_POINT_BITS
    sens = fed_heart.DP_CLIP * unit + math.sqrt(dim)
    for name, arm in record["fedheart"].items():
        eps = dp.epsilon_of(arm["variance_per_hospital"], fed_heart.DP_ROUNDS, sens, 4, dim, fed_heart.DP_DELTA)
        assert round(eps, 4) == arm["epsilon_accounted"], name
        assert arm["epsilon_lower_bound"] <= arm["epsilon_accounted"] and arm["under_the_accountant"], name
    for name, arm in record["murmur"].items():
        assert arm["epsilon_lower_bound"] <= arm["epsilon_accounted"] and arm["under_the_accountant"], name


def test_the_murmur_accountant_is_the_studys(record) -> None:
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("murmur_dp", REPO / "scripts" / "train" / "medical_murmur_dp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, arm in record["murmur"].items():
        eps = mod.dpsgd_epsilon(arm["sampling_rate"], arm["noise_multiplier"], arm["steps"], mod.DELTA)
        assert round(eps, 4) == arm["epsilon_accounted"], name


def test_the_seeded_bugs_are_caught(record) -> None:
    checks = record["seeded_checks"]
    assert set(checks) == {"fedheart_a_sixteenth_of_the_variance", "murmur_every_person_every_step"}
    for name, arm in checks.items():
        assert arm["epsilon_lower_bound"] > arm["epsilon_accounted"] and not arm["under_the_accountant"], name


def _lb(arm: dict) -> str:
    """A lower bound as the note prints it: truncated to two decimals, never rounded up."""
    return f"{math.floor(arm['epsilon_lower_bound'] * 100 + 1e-9) / 100:.2f}"


def note_problems(note: str, record: dict) -> list[str]:
    flat = re.sub(r"\s+", " ", note)
    fh, mu, sc = record["fedheart"], record["murmur"], record["seeded_checks"]
    want = [", ".join(f"{_lb(fh[f'epsilon_{e:g}'])} at epsilon {e:g}" for e in (8, 4, 2, 1)),
            f"{_lb(mu['noise_1.89138'])} against its accountant's {mu['noise_1.89138']['epsilon_accounted']:.2f}",
            f"{_lb(mu['noise_3.45294'])} against {mu['noise_3.45294']['epsilon_accounted']:.2f}",
            f"{_lb(sc['fedheart_a_sixteenth_of_the_variance'])} against the 1",
            f"{_lb(sc['murmur_every_person_every_step'])} against "
            f"{sc['murmur_every_person_every_step']['epsilon_accounted']:.2f}"]
    return [f"the note does not quote {w!r}" for w in want if w not in flat]


def test_the_note_quotes_the_record(record) -> None:
    assert note_problems(NOTE.read_text(encoding="utf-8"), record) == []

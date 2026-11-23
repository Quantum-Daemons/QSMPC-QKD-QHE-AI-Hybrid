"""The repository's DP-SGD accountant agrees with dp_accounting's and never claims less privacy loss (2026-11-23).

WHY THIS EXISTS

The murmur study accounts DP-SGD with its own Renyi-DP code (`scripts/train/medical_murmur_dp.py`). An accountant of
one's own is the place an under-count would hide: a missing order, a wrong conversion, and the page states an epsilon
the training did not earn. On the owner's choice of 2026-11-22 it was checked against Google's dp_accounting
(Apache-2.0), in its own environment (`scripts/measure_dp_accountant_crosscheck.py`), at the study's own settings.

WHAT IS ASSERTED

- At each recorded noise multiplier the repository's epsilon is within 2 percent of dp_accounting's RDP epsilon (the
  same method), and never below its PLD epsilon (the tighter one): the repository's figure is conservative.
- Seeded: an accountant that drops the composition over steps is refused.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "docs" / "measurements" / "dp-accountant-crosscheck-2026-11-23.json"


@pytest.fixture(scope="module")
def murmur():
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("murmur_dp", REPO / "scripts" / "train" / "medical_murmur_dp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def problems(epsilon_of, record: dict) -> list[str]:
    s = record["settings"]
    out = []
    for row in record["rows"]:
        eps = epsilon_of(s["sampling_rate"], row["noise_multiplier"], s["steps"], s["delta"])
        ref, tight = row["dp_accounting_rdp_epsilon"], row["dp_accounting_pld_epsilon"]
        if abs(eps - ref) / ref > 0.02:
            out.append(f"sigma {row['noise_multiplier']}: {eps:.4f} is more than 2 percent from dp_accounting's RDP {ref}")
        if eps < tight:
            out.append(f"sigma {row['noise_multiplier']}: {eps:.4f} is below dp_accounting's PLD {tight}, an under-count")
    return out


def test_the_accountant_matches_and_is_conservative(murmur) -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["settings"]["sampling_rate"] == pytest.approx(64 / 511)
    assert problems(murmur.dpsgd_epsilon, record) == []


def test_an_accountant_that_forgets_the_steps_is_refused(murmur) -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))

    def one_step(q, sigma, steps, delta):
        return murmur.dpsgd_epsilon(q, sigma, 1, delta)

    assert problems(one_step, record)

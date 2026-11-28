"""The medical page's key points quote two figures, and each is held to where it comes from (Round 86, 2026-11-28).

The review of every use case's checks (2026-11-27, gap 12) found that `UseCasePage.tsx` prints "all 3,226 parameters"
for the encrypted path's student and "epsilon 8 down to 1, delta 1e-6" for the federation's differential privacy, and
that both equalled their sources by hand only: the student's parameter count in the generated config (from the
registry's record of the published student) and `FED_HEART_DP` in `clientFedHeart.ts`, which the DP card runs. A
retrained student or a changed privacy schedule would have left the key points saying the old figures.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
PAGE = WEB / "src" / "pages" / "UseCasePage.tsx"


def test_the_encrypted_students_parameter_count_is_the_published_students() -> None:
    found = re.findall(r"all ([0-9,]+) parameters, from its pinned Hugging Face revision", PAGE.read_text())
    assert len(found) == 1, found
    config = json.loads((WEB / "src" / "lib" / "config.defaults.json").read_text())
    student = next(s for s in config["models"]["medical_fl"] if s["key"] == "medical_fl.student")
    assert int(found[0].replace(",", "")) == student["achieved"]["student_params"]


def test_the_privacy_range_is_the_one_the_dp_card_runs() -> None:
    found = re.findall(r"epsilon (\d+) down to (\d+), delta ([0-9e.+-]+);", PAGE.read_text())
    assert len(found) == 1, found
    source = (WEB / "src" / "lib" / "clientFedHeart.ts").read_text()
    block = re.search(r"export const FED_HEART_DP = \{(.*?)\};", source, re.S)
    assert block, "FED_HEART_DP is not where the page's DP card reads it"
    epsilons = [float(x.strip().replace("Infinity", "inf"))
                for x in re.search(r"epsilons: \[([^\]]*)\]", block.group(1)).group(1).split(",")]
    delta = float(re.search(r"delta: ([0-9e.+-]+)", block.group(1)).group(1))
    finite = [e for e in epsilons if math.isfinite(e)]
    assert (float(found[0][0]), float(found[0][1]), float(found[0][2])) == (max(finite), min(finite), delta)

"""Every data file the site ships is classified, carries what its licence asks, and has its own row in the register (Round 86).

WHY THIS EXISTS (the dataset-licence review of 2026-11-21, the owner's option 3)

`tests/test_r84_shipped_data_carries_its_licence.py` checks a HAND-WRITTEN list of shipped files, and files added after it
was written were not on it. Read against `git ls-files web/public/data` on 2026-11-21:

- `satellite/boundary.json` shipped OPS-SAT-AD test segments (two feature values and the published label each, in `near`)
  with no licence, link, credit or note of changes, which CC BY 4.0 section 3(a) asks of anything shared;
- `satellite/vus_pr.json` and `medical/fedheart_dp_sweep.json`, figures computed from ESA-ADB and UCI Heart Disease, carried
  no credit, where the page's other aggregate cards (`cxr_fl`, `pcg_fl`, `murmur_dp`) credit their corpora;
- eight shipped files had no row of their own in `DATA_LICENSES.md`, the register of what may be published.

So the subject here is the set git ships, not a list: a file added tomorrow is in scope the day it lands, and it must be put
in one of three classes, each with its own rule.

- ROWS (the default): rows or values from a corpus, or generated data. `licence`, a `licence_url` that matches it,
  `modifications`, and an `attribution` unless the licence asks for none (Apache-2.0, CC0).
- AGGREGATES: figures computed from a corpus, credited by an `attribution` object, or one object per corpus or model: each
  with where it is downloaded, its citation, and its licence or, for NIH's ChestX-ray14, which grants none, NIH's terms.
- WEIGHTS: fitted numbers, which raise no such question; the QCNN weights' credit is checked by the Round 84 test.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DATA = "web/public/data/"
LINKS = {
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC-BY-3.0-IGO": "https://creativecommons.org/licenses/by/3.0/igo/",
    "CC-BY-3.0": "https://creativecommons.org/licenses/by/3.0/",
    "CDLA-Sharing-1.0": "https://cdla.dev/sharing-1-0/",
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
    "NIH ChestX-ray14 terms": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
}
NO_CREDIT = ("Apache-2.0", "CC0-1.0")
#: Figures computed from a corpus, and the corpus each credits.
AGGREGATES = {
    "medical/cxr_fl.json": "NIH ChestX-ray14",
    "medical/pcg_fl.json": "PhysioNet/CinC Challenge 2016",
    "medical/murmur_dp.json": "CirCor and the PhysioNet/CinC Challenge 2016",
    "medical/fedheart_dp_sweep.json": "UCI Heart Disease",
    "medical/square_student_sets.json": "MedMNIST v2 BreastMNIST",
    "satellite/vus_pr.json": "ESA-ADB",
    # The pre-registered threshold comparison (Round 86, batch 6), shown on the page on the owner's choice of 2026-11-26.
    "satellite/thresholds.json": "ESA-ADB",
    # The federation of 5G-NIDD's two base stations (Round 86, batch 6; the owner's choice of 2026-11-26): figures computed
    # from 5G-NIDD beside the fitted numbers, and each shipped held-out flow's station.
    "telecom_nidd/federated.json": "5G-NIDD",
    # Round 86's batch 8, the owner's choices of 2026-11-27: four pre-registered results, figures only, each crediting its corpus.
    "satellite/jinan1_pass_v2.json": "Jinan-1 source data (Li et al.)",
    "telecom_nidd/dirichlet.json": "5G-NIDD",
    "medical/nodule3d_reporting.json": "LIDC-IDRI",
    "defi/qcnn_comparison.json": "From Risk to Rescue dataset (Aave v3, Polygon)",
    # The QCNNs' tensor-network checks (Round 86, batch 6; the owner's choice of 2026-11-26): figures computed from the
    # medical, telecom and satellite QCNNs' corpora, one credit each, from the registry's dataset records.
    "qcnn_tn_checks.json": "WDBC, the synthetic QKD-attack rows, OPS-SAT-AD and ESA-ADB",
}
#: Fitted numbers.
WEIGHTS = {
    "finance/qcnn_weights.json", "finance/student_weights.json", "medical/qcnn_weights.json", "telecom/qcnn_weights.json",
    "telecom/qcnn_weights_q7.json", "telecom/student_weights.json", "satellite/student_weights.json",
    # F7b's square network (Round 86), the finance page's fully encrypted model since 2026-11-25.
    "finance/square_student_weights.json",
    # The 5G-NIDD detector's pre-registered CKKS student (Round 86, batch 6); its rows' credit is in student_heldout.json.
    "telecom_nidd/student_weights.json",
}


def _shipped() -> list[str]:
    out = subprocess.run(["git", "ls-files", DATA], cwd=REPO, check=True, capture_output=True, text=True).stdout
    return [line[len(DATA):] for line in out.splitlines() if line]


def _head(licence: str) -> str:
    return licence.split(";")[0].split(" asserted by ")[0].split(" (")[0].strip()


def row_problems(d: dict) -> list[str]:
    head = _head(d.get("licence") or "") if isinstance(d.get("licence"), str) else ""
    problems = []
    if head not in LINKS:
        problems.append(f"licence {d.get('licence')!r} is not one this test knows")
    elif d.get("licence_url") != LINKS[head]:
        problems.append(f"licence_url {d.get('licence_url')!r} does not match {head}")
    if not str(d.get("modifications") or "").strip():
        problems.append("no note of the changes")
    if head not in NO_CREDIT and not str(d.get("attribution") or "").strip():
        problems.append("no credit")
    return problems


def aggregate_problems(d: dict) -> list[str]:
    a = d.get("attribution")
    if not isinstance(a, dict):
        return ["no attribution object"]
    credits = [a] if "citation" in a else [v for v in a.values() if isinstance(v, dict)]
    if not credits:
        return ["no attribution object"]
    problems = []
    for c in credits:
        name = c.get("dataset") or c.get("model") or "?"
        problems += [f"the credit {name!r} has no {k}" for k in ("download", "citation") if not str(c.get(k) or "").strip()]
        if not str(c.get("licence") or c.get("terms") or "").strip():
            problems.append(f"the credit {name!r} has no licence or terms")
    return problems


def test_every_shipped_file_is_in_one_class() -> None:
    files = _shipped()
    assert len(files) >= 34, files
    assert set(AGGREGATES) <= set(files) and WEIGHTS <= set(files), "a class names a file the site no longer ships"
    assert not set(AGGREGATES) & WEIGHTS


@pytest.mark.parametrize("rel", _shipped())
def test_each_shipped_file_carries_what_its_class_asks(rel: str) -> None:
    if rel in WEIGHTS:
        return
    d = json.loads((REPO / DATA / rel).read_text(encoding="utf-8"))
    assert isinstance(d, dict), f"{rel}: not a JSON object"
    problems = aggregate_problems(d) if rel in AGGREGATES else row_problems(d)
    assert problems == [], f"{rel}: {problems}"


@pytest.mark.parametrize("rel", _shipped())
def test_each_shipped_file_has_its_own_row_in_the_register(rel: str) -> None:
    named = set(re.findall(r"`([^`]+)`", (REPO / "DATA_LICENSES.md").read_text(encoding="utf-8")))
    assert DATA + rel in named or rel in named, f"DATA_LICENSES.md has no row naming {DATA + rel}"


def _script_constant(script: str, name: str):
    for node in ast.parse((REPO / script).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def test_the_three_cards_carry_what_their_scripts_write() -> None:
    """The fields were added to the shipped files without re-running the measurements; the scripts write the same ones."""
    from qsh.models import registry

    b = json.loads((REPO / DATA / "satellite/boundary.json").read_text())
    for k, v in _script_constant("scripts/measure_satellite_boundary.py", "CARD_LICENCE").items():
        assert b[k] == v, k
    assert b["attribution"] == registry.get("satellite_comms.opssat").dataset.attribution
    v = json.loads((REPO / DATA / "satellite/vus_pr.json").read_text())
    assert v["attribution"] == {**_script_constant("scripts/measure_satellite_vus_pr.py", "CARD_ATTRIBUTION"),
                                "citation": registry.get("satellite_comms.telemanom").dataset.attribution}
    f = json.loads((REPO / DATA / "medical/fedheart_dp_sweep.json").read_text())
    assert f["attribution"] == _script_constant("scripts/measure_fedheart_dp_sweep.py", "CARD_ATTRIBUTION")
    sites = json.loads((REPO / DATA / "medical/heart_disease_sites.json").read_text())
    assert f["attribution"]["citation"] == sites["attribution"], "the DP band credits its corpus as the shipped corpus file does"


def test_the_checks_refuse_what_they_exist_for() -> None:
    """Seeded: the boundary as it shipped until 2026-11-21, and an aggregate credit missing its licence."""
    b = json.loads((REPO / DATA / "satellite/boundary.json").read_text())
    for k in ("licence", "licence_url", "modifications", "attribution"):
        b.pop(k)
    assert row_problems(b)
    credit = dict(json.loads((REPO / DATA / "medical/fedheart_dp_sweep.json").read_text())["attribution"])
    credit.pop("licence")
    assert aggregate_problems({"attribution": credit}) == ["the credit '" + credit["dataset"] + "' has no licence or terms"]

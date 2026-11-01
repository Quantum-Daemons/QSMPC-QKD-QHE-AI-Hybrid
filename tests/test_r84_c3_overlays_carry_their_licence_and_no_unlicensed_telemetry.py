"""Every 2-D overlay carries its licence link and change note, and no unlicensed telemetry is served.

WHY THIS EXISTS (Round 84, checklist RD23, PI7 and L8)

The five `real_2d.json` overlays printed a licence NAME and nothing else, while CC BY 4.0 (s.3(a)(1))
and CDLA-Sharing-1.0 (s.3.1(b), s.3.3) ask for a link to the licence and a note of the changes; the
points ARE changed (two columns selected, rounded, medical's class recoded). The medical overlay said
BreastMNIST "cannot be redistributed" while its test split ships in public Hub repos. The telecom
overlay drew all 3,200 rows of a QCNN's training draw under a "held-out split" caption, and its
`licence` field named no grant while the registry says none is granted. And the satellite page served
`telemetry.json`, NASA SMAP/MSL windows for which no provider grants a data licence (JPL is Caltech).

WHAT THIS ASSERTS

  - each overlay: `licence_url` (where anything is licensed) and `modifications`, rendered as a link
    and a change note under the figure; the models it belongs to and its split are named;
  - the medical wording agrees with the shipped BreastMNIST test split; telecom's licence is the
    registry's;
  - `telemetry.json` is gone from web/public, nothing in web/src fetches it, and nothing regenerates
    it; its replacement is OPS-SAT-AD (CC BY 4.0) segments the demo regression was never fitted on.

All of it fails on the baseline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PUBLIC = REPO / "web" / "public" / "data"
CHART = REPO / "web" / "src" / "components" / "VerificationChart.tsx"
USECASES = ["finance", "genomics", "medical", "satellite", "telecom"]
#: Licence families whose text asks for a link and a change note when the data is shared.
NEEDS_LINK = {"CC-BY-4.0": "creativecommons.org/licenses/by/4.0",
              "CDLA-Sharing-1.0": "cdla.dev/sharing-1-0",
              "CC0-1.0": "creativecommons.org/publicdomain/zero/1.0",
              "Apache-2.0": "apache.org/licenses/LICENSE-2.0"}


def _overlay(uc: str) -> dict:
    return json.loads((PUBLIC / uc / "real_2d.json").read_text())


@pytest.mark.parametrize("uc", USECASES)
def test_each_overlay_carries_licence_link_and_change_note(uc: str) -> None:
    d = _overlay(uc)
    assert d.get("modifications"), f"{uc}/real_2d.json has no change note (`modifications`)"
    if d["licence"] in NEEDS_LINK:
        assert NEEDS_LINK[d["licence"]] in (d.get("licence_url") or ""), (
            f"{uc}/real_2d.json names {d['licence']} with no link to it: {d.get('licence_url')!r}")
    else:
        # Every overlay names a linked licence since Round 84, when the telecom rows the repository
        # generates were released under its Apache-2.0 (until then: "no licence granted", no link).
        raise AssertionError(f"{uc}: a licence string with no link recorded here: {d['licence']!r}")


@pytest.mark.parametrize("uc", USECASES)
def test_each_overlay_names_its_split_and_its_models(uc: str) -> None:
    d = _overlay(uc)
    assert d.get("split_note"), f"{uc}: the overlay does not say which rows these are"
    assert isinstance(d.get("models"), list)
    if d["is_training_corpus"]:
        assert d["models"], f"{uc}: a training-corpus overlay that does not name its models ('this model')"
    if d.get("separability_auc_pr") is not None:
        assert d.get("separability_method"), f"{uc}: separability figure without its estimator"


def test_the_medical_overlay_agrees_with_the_shipped_breastmnist_split() -> None:
    d = _overlay("medical")
    text = d["provenance"] + d["note"] + d["split_note"]
    assert "cannot be redistributed" not in text, (
        "the medical overlay still says BreastMNIST cannot be redistributed, while its 156-image test "
        "split ships with the published models' evaluation samples")
    assert "156-image test split" in d["provenance"]
    # The bundled regression shares this plane, so the old 'no shared feature space' line is gone.
    assert "no shared feature space exists" not in text


def test_the_telecom_overlay_licence_is_the_registrys() -> None:
    from qsh.models.registry import DS_QKD_ATTACK_SYNTH
    d = _overlay("telecom")
    assert d["licence"] == DS_QKD_ATTACK_SYNTH.licence, (d["licence"], DS_QKD_ATTACK_SYNTH.licence)
    assert "all 3,200 rows" not in d["provenance"]


def test_the_overlay_renders_the_link_and_the_change_note() -> None:
    src = CHART.read_text()
    body = src[src.index("export function RealDataScatter"):]
    body = body[: re.search(r"\n(?:export )?function \w+\(", body).start()]
    assert "data.licence_url" in body and "data.modifications" in body and "<LicenceLine" in body
    line = src[src.index("function LicenceLine"):]
    line = line[: line.index("\n}\n")]
    assert "<a href={licence.url}" in line and "Changes:" in line
    for claim in ("genuine train-versus-inference comparison", "deliberately not on the same axes"):
        assert claim not in src, f"the overlay caption still claims {claim!r}"


# --------------------------------------------------------------------------------------------- #
#  satellite telemetry
# --------------------------------------------------------------------------------------------- #

def test_the_smap_msl_telemetry_is_not_served_or_regenerated() -> None:
    assert not (PUBLIC / "satellite" / "telemetry.json").exists(), (
        "web/public/data/satellite/telemetry.json (NASA SMAP/MSL, no data licence from any provider) "
        "is still served")
    hits = [str(p.relative_to(REPO)) for p in (REPO / "web" / "src").rglob("*.ts*")
            if "telemetry.json" in p.read_text()]
    assert not hits, f"the browser still references telemetry.json: {hits}"
    prep = (REPO / "scripts" / "prep_real_kaggle_data.py").read_text()
    assert '"satellite" / "telemetry.json"' not in prep, "the Kaggle prep script still writes it"


def test_the_replacement_is_opssat_segments_the_demo_model_never_saw() -> None:
    from sklearn.linear_model import LogisticRegression

    d = json.loads((PUBLIC / "satellite" / "opssat_segments.json").read_text())
    assert d["licence"] == "CC-BY-4.0" and NEEDS_LINK["CC-BY-4.0"] in d["licence_url"]
    assert "OPS-SAT" in d["provenance"] and "SMAP" not in d["provenance"]
    assert "zenodo.12588359" in d["attribution"] and d["modifications"]
    src = json.loads((REPO / "models" / "satellite_opssat" / "eval_sample.json").read_text())
    X, y = np.array(src["samples"], dtype=float), np.array(src["labels"])
    held = np.array(d["row_index"])
    fit = np.array(sorted(set(range(len(y))) - set(held.tolist())))
    assert len(held) == len(d["segments"]) == 159 and len(fit) == 370
    mu, sd = X[fit].mean(axis=0), X[fit].std(axis=0)
    assert np.abs((X[held] - mu) / sd - np.array(d["segments"])).max() < 1e-4, (
        "the shipped segments are not scaled by constants from the fitting segments alone")
    assert d["labels"] == [int(v) for v in y[held]]
    lr = LogisticRegression(max_iter=5000, class_weight="balanced").fit((X[fit] - mu) / sd, y[fit])
    assert np.abs(lr.coef_[0] - np.array(d["model"]["W"][1])).max() < 1e-3, (
        "the demo regression does not refit on the fitting segments, so it was fitted on others")
    assert "not satellite_comms.opssat" in d["real_model"].replace("NOT ", "not ")

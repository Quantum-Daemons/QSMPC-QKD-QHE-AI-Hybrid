"""The Validations page's privacy-audit card shows the recorded audits, and only what their records say (Round 86).

On the owner's choice of 2026-11-24, the third layer of the PETs evaluation (references/pets-evaluation-2026-11.md) is
shown on the Validations page as recorded results. `web/src/lib/privacyAudits.json` is what
`scripts/export_privacy_audits.py` copies from docs/measurements; the card draws that file and states no figure of its
own. These tests hold the file to the records, every record to the file, and the card to the file.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MEASUREMENTS = REPO / "docs" / "measurements"
EXPORT = REPO / "web" / "src" / "lib" / "privacyAudits.json"
CARD = REPO / "web" / "src" / "components" / "PrivacyAudits.tsx"
PAGE = REPO / "web" / "src" / "pages" / "Validations.tsx"


def _export() -> dict:
    return json.loads(EXPORT.read_text(encoding="utf-8"))


def test_the_file_is_what_the_exporter_writes_from_the_records() -> None:
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "export_privacy_audits.py"), "--check"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr


def test_every_source_is_named_with_its_digest() -> None:
    for name, digest in _export()["sources"].items():
        assert hashlib.sha256((MEASUREMENTS / name).read_bytes()).hexdigest() == digest, name


def test_no_audit_record_is_left_out() -> None:
    """A new membership, reconstruction or DP-audit record must reach the card, or the exporter must refuse it."""
    sources = set(_export()["sources"])
    for pattern in ("mia-*.json", "reconstruction-*.json", "dp-one-run-audit-*.json", "dp-kcandidate-audit-*.json",
                    "attribute-inference-*.json", "gradient-inversion-*.json"):
        found = {p.name for p in MEASUREMENTS.glob(pattern)}
        assert found <= sources, f"not on the card: {sorted(found - sources)}"


def test_each_membership_figure_is_the_records() -> None:
    for study in _export()["membership"]:
        record = json.loads((MEASUREMENTS / study["file"]).read_text(encoding="utf-8"))
        assert study["shadow_models"] == record["design"]["shadow_models"]
        for m in study["models"]:
            r = record["results"][m["key"]]
            for attack, key in (("lira", "lira_online"), ("rmia", "rmia_offline")):
                assert m[attack]["auc"] == r[key]["auc"]
                assert m[attack]["tpr"] == [r[key].get(f"tpr_at_fpr_{f:g}") for f in (0.001, 0.01, 0.1)]


def test_the_dp_arms_are_the_records_and_the_seeded_bugs_are_marked() -> None:
    dp = _export()["dp_one_run"]
    record = json.loads((MEASUREMENTS / dp["file"]).read_text(encoding="utf-8"))
    for arm in dp["arms"]:
        assert arm["lower_bound"] <= arm["accounted"] or arm["seeded_bug"], arm
    seeded = [a for a in dp["arms"] if a["seeded_bug"]]
    assert len(seeded) == len(record["seeded_checks"]) == 2
    assert sorted(a["lower_bound"] for a in seeded) == sorted(c["epsilon_lower_bound"] for c in record["seeded_checks"].values())
    assert all(a["lower_bound"] > a["accounted"] for a in seeded)
    honest = [a for a in dp["arms"] if not a["seeded_bug"]]
    assert len(honest) == len(record["fedheart"]) + len(record["murmur"])


def test_the_card_states_no_figure_of_its_own() -> None:
    """Every audit figure on the card comes from the file: none of them is typed into the component."""
    text = CARD.read_text(encoding="utf-8")
    figures: set[str] = set()

    def walk(x: object) -> None:
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, float) and x not in (0.001, 0.01, 0.1):
            figures.update({f"{x}", f"{100 * x:.2f}", f"{x:.3f}"})

    walk({k: v for k, v in _export().items() if k != "fprs"})
    # Three significant digits or more: an axis's 0.01 or a margin's 8 is layout, not a figure from a record.
    figures = {f for f in figures if len(f.replace(".", "").replace("-", "").lstrip("0")) >= 3}
    typed = {f for f in figures if re.search(rf"(?<![\w.]){re.escape(f)}(?![\w])", text)}
    assert not typed, f"figures typed into the card: {sorted(typed)}"


def test_the_card_says_what_an_attack_shows_and_publishes_no_record() -> None:
    text = re.sub(r"\s+", " ", CARD.read_text(encoding="utf-8"))
    assert "a lower bound on what leaks, never proof that nothing does" in text
    assert "no reconstructed record or image" in text
    assert "proves nothing more" in text


def test_the_card_is_on_the_validations_page() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert 'import PrivacyAudits from "../components/PrivacyAudits";' in page
    assert page.index("<CbomPanel />") < page.index("<PrivacyAudits />")


def test_the_guessing_game_rows_are_the_records_and_the_bugs_are_caught() -> None:
    kc = _export()["dp_kcandidate"]
    record = json.loads((MEASUREMENTS / kc["file"]).read_text(encoding="utf-8"))
    honest = [r for r in kc["rows"] if not r["seeded_bug"]]
    seeded = [r for r in kc["rows"] if r["seeded_bug"]]
    assert len(seeded) == len(record["seeded_checks"]) == 2
    assert all(r["under_the_accountant"] and r["within_the_bound"] for r in honest)
    assert all(not r["under_the_accountant"] and not r["within_the_bound"] for r in seeded)
    assert len(honest) == sum(len([g for g in ("k2_add_remove", "k7_swap") if g in arm])
                              for study in ("fedheart", "murmur") for arm in record[study].values())


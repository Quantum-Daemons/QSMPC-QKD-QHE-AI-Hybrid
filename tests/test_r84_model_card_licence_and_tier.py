"""Hub cards name their licence the way the Hub reads it, and a plaintext card claims no encryption.

WHY THIS EXISTS (Round 84). Every card said `license: other`, with no `license_link`, although two of
the licences are on the Hub's own list (the Hub indexed all eleven repos as "other"); and the template
gave every plaintext model "the encrypted path runs a distilled student, not this model", which is
false for the satellite teachers (no encrypted stage on that page), the finance teacher (the page
encrypts a bundled PaySim model), and the ResNet-50 and 5G-NIDD models (nothing is distilled from
either).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.models import registry  # noqa: E402

spec_ = importlib.util.spec_from_file_location("publish_hf", REPO / "scripts" / "publish_hf.py")
publish_hf = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(publish_hf)


def _cards():
    return [(s, publish_hf.model_card(s, registry.achieved(s.key))) for s in registry.specs() if s.hf_repo]


def test_every_card_names_its_licence_as_the_hub_reads_it() -> None:
    for spec, card in _cards():
        front = card.split("---")[1]
        if spec.dataset.licence == "CC-BY-4.0":
            assert "license: cc-by-4.0" in front, spec.key
        elif spec.dataset.licence == "CDLA-Sharing-1.0":
            assert "license: cdla-sharing-1.0" in front, spec.key
        elif spec.dataset.licence == "Apache-2.0":
            # The repository's own licence for the data it generates (Round 84, the owner's choice).
            assert "license: apache-2.0" in front, spec.key
        else:
            assert "license: other" in front and "license_link: https://" in front, (
                f"{spec.key}: `license: other` without a license_link")


def test_no_plaintext_card_claims_a_distilled_student_runs_encrypted() -> None:
    for spec, card in _cards():
        assert "the encrypted\npath runs a distilled student" not in card, spec.key
        assert "path runs a distilled student, not this model" not in card, spec.key


def test_the_card_links_the_repository_where_it_lives() -> None:
    for spec, card in _cards():
        assert "github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid" in card, spec.key

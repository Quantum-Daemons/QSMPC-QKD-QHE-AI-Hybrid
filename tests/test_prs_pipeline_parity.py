"""The E2E pipeline and the browser score the SAME polygenic risk score.

WHY THIS EXISTS. Round 31 migrated the genomics use case from PGS000001 (77 variants, keyed by rsID) to
PGS000004 (313 variants, keyed by ``chr:pos:effect_allele``). It changed four surfaces - the browser
payload, the training script, the model registry and the payload generator - and missed the fifth. The
E2E pipeline read::

    prs_name = config.get("prs", "breast_cancer")

and **nothing anywhere set that key**, so the literal default stood and
``python -m qsh.orchestration.pipeline --usecase genomics`` went on scoring 77 rsID-keyed variants while
every other surface scored 313 position-keyed ones. Two surfaces reported different scores for the same
use case, and `qsh.data.loaders` carried a sentence arguing that this exact split "would be worse than
either alone" - which by then described the shipped state rather than warning against it.

Nothing failed. ``config.get(key, default)`` cannot fail: an absent key silently yields the default, the
default was a valid score, and the pipeline produced a complete, plausible, correctly-shaped result over
the wrong model. The same shape has now bitten this project three times - ``config.usecase`` read but
never emitted, the free-space channel reading the fibre page's distance keys, and this - and the common
factor every time is a default that no mechanism compares against anything.

So the fix is a config FIELD (``DataConfig.genomics_prs_score``) and this test, which compares the two
surfaces rather than trusting them to agree. It reads the browser's payload through the registry rather
than by hardcoding a filename, so pointing the demo at a different score moves both sides together or
fails here.
"""
from __future__ import annotations

import json
from pathlib import Path

from qsh.config import get_config
from qsh.data.loaders import _PRS_FILES, ACTIVE_PRS, load_prs
from qsh.models.registry import DEMO_SAMPLE_FILES

REPO = Path(__file__).resolve().parents[1]


def _run_genomics() -> dict:
    """One genomics run. Imported lazily so a collection error in the use case is attributable."""
    from qsh.usecases import genomics

    return genomics.run_demo({"seed": 1})


def _browser_payload() -> dict:
    """The genomics payload the browser actually fetches, located via the registry."""
    path = REPO / DEMO_SAMPLE_FILES["genomics"]
    assert path.exists(), f"{path} is what the registry points the browser at, and it is missing"
    return json.loads(path.read_text())


def test_the_configured_score_is_a_real_bundled_score() -> None:
    """A typo in the config field must fail here, not at the first pipeline run."""
    name = get_config().data.genomics_prs_score
    assert name in _PRS_FILES, (
        f"DataConfig.genomics_prs_score is {name!r}, which is not a key of _PRS_FILES "
        f"({sorted(_PRS_FILES)}). The pipeline would raise KeyError on the first genomics run."
    )


def test_the_pipeline_RUN_matches_the_browser_payload() -> None:
    """Compare what the pipeline actually PRODUCES against what the browser fetches.

    Deliberately behavioural rather than config-versus-payload. An earlier draft of this file read the
    config field and compared that to the browser, which passes even if the use case ignores the field
    entirely - and ignoring the field is precisely the defect. Running the use case is the only version
    that cannot be satisfied by a literal somewhere in the middle.
    """
    result = _run_genomics()
    payload = _browser_payload()
    browser_keys = payload.get("keys") or payload.get("rsids")
    assert browser_keys is not None, "the browser payload carries neither `keys` nor `rsids`"

    assert result["prs"]["n_snps"] == len(browser_keys), (
        f"the pipeline run scored {result['prs']['n_snps']} variants; the browser payload "
        f"{DEMO_SAMPLE_FILES['genomics']} carries {len(browser_keys)}. This is the Round 31 defect: "
        "the migration reached one surface and not the other, and neither side raises."
    )
    assert result["prs"]["name"] == get_config().data.genomics_prs_score, (
        f"the run reports score {result['prs']['name']!r} while the config field says "
        f"{get_config().data.genomics_prs_score!r}, so the use case is not reading the field."
    )


def test_the_configured_score_and_the_browser_agree_on_the_key_space() -> None:
    """The two surfaces must agree on the KEY SPACE, not only the count.

    Count alone is not enough. PGS000001 and PGS000004 differ in both count and keying, so a partial
    migration could in principle land on the right count with rsID keys and still be scoring a
    different model. Both are compared.
    """
    configured = get_config().data.genomics_prs_score
    keys, _eff, weights = load_prs(configured)
    payload = _browser_payload()
    browser_keys = payload.get("keys") or payload.get("rsids")
    assert browser_keys is not None, "the browser payload carries neither `keys` nor `rsids`"

    assert len(keys) == len(browser_keys), (
        f"the pipeline scores {len(keys)} variants from {configured!r} and the browser payload "
        f"{DEMO_SAMPLE_FILES['genomics']} carries {len(browser_keys)}. This is the Round 31 defect: "
        "the migration reached one surface and not the other, and neither side raises."
    )
    assert list(keys) == list(browser_keys), (
        "the pipeline and the browser carry the same NUMBER of variants but not the same ones. "
        "Same count, different key space or different order means the two surfaces score different "
        "models while every count-based check passes."
    )
    assert len(weights) == len(keys), "the weight vector does not match the key list"


def test_the_pipeline_reads_the_config_field_rather_than_a_literal() -> None:
    """The specific regression, pinned: a literal default here is invisible to every other check.

    Asserted by reading the source rather than by behaviour, because the behaviour of a literal default
    and of a field default are identical whenever they happen to hold the same value - which is exactly
    why the original went unnoticed for a round.
    """
    src = (REPO / "src" / "qsh" / "usecases" / "genomics" / "__init__.py").read_text()
    assert 'config.get("prs", "breast_cancer")' not in src, (
        "the genomics use case has gone back to a hardcoded score default. That literal is how the "
        "pipeline came to score PGS000001 while the browser scored PGS000004: it is not in the config "
        "dump, not overridable per run, and not comparable against anything."
    )
    assert "genomics_prs_score" in src, (
        "the genomics use case no longer reads DataConfig.genomics_prs_score. If the score moved to a "
        "different config field, point this test at it; do not delete the check."
    )


def test_the_loader_default_and_the_config_field_name_the_same_score() -> None:
    """Two places name the active score, so assert they agree rather than hoping.

    `loaders.ACTIVE_PRS` exists so the loaders stay pure functions with no config dependency, and
    `DataConfig.genomics_prs_score` exists so the score is overridable per run. That is a deliberate
    duplication, and the cost of a deliberate duplication is exactly this test - without it the two
    drift, and the drift is invisible because every caller passes `name` explicitly today, so the
    loader default is never exercised. A dead default that disagrees with the live one is how the
    retired score survived Round 31.
    """
    assert ACTIVE_PRS == get_config().data.genomics_prs_score, (
        f"loaders.ACTIVE_PRS is {ACTIVE_PRS!r} and DataConfig.genomics_prs_score is "
        f"{get_config().data.genomics_prs_score!r}. Change both or neither."
    )

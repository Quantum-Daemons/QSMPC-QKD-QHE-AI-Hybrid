"""A directory that ships a tracked model artifact must ship a card describing it.

WHY THIS EXISTS

Three directories under `models/` shipped tracked artifacts with no `README.md` at all, and every
gate that could have said so was enumerating the wrong set:

  * `tests/test_a_model_card_states_what_its_weights_state.py` globs `models/*/README.md`. A
    directory with no card contributes no card, so it is not checked - it is not even counted.
  * `tests/test_qcnn_usecases.py` globs `models/qcnn_*`, which none of the three matched.

So the two gates whose subject is model cards were both structurally unable to notice a MISSING one.
That is the same defect their own comments already name: `test_a_model_card_states_what_its_weights_state`
opens by recording that `test_every_shipped_qcnn_card_places_its_seed` "enumerated cards rather than
artifacts and so passed an artifact with no card at all". The lesson was written down, and then the
replacement gate was built by globbing cards again.

The three, and they are not one problem:

  * `models/genomics_prs` - in `registry.ARTIFACT_DIRS` as `genomics.prs` and VISITOR-FACING, reached
    from `web/src/components/ModelPanel.tsx`. A shipped artifact a visitor's page reads, with nothing
    documenting it.
  * `models/finance_aml_scorecard` - written by `scripts/train/finance_scorecard.py`.
  * `models/telecom_ids_student` - written by `scripts/train/telecom_ids.py`.

WHAT IS ASSERTED

Every directory under `models/` that ships at least one TRACKED file other than its own card must
also ship a tracked `README.md`. Enumerated from `git ls-files`, so the subject is the artifacts and
a directory added tomorrow is in scope the day it lands - the property a card-globbing gate cannot
have, because the thing it must detect is precisely the absence of the file it enumerates.

WHAT IS DELIBERATELY NOT ASSERTED

Not what the card SAYS. `test_a_model_card_states_what_its_weights_state.py` already checks every
number a card presents against the artifact it documents, and duplicating that here would give two
places to update when the contract moves. This gate asserts only that there IS one, which is exactly
the hole the other gate cannot cover.

Not that an untracked directory needs a card. Training writes intermediates that are gitignored on
purpose; the obligation attaches to what SHIPS.
"""

from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: The three that were uncarded when this gate was written, kept as the anti-tautology's subject.
#: Not an exception list - the gate does not consult it, and the test below asserts they are carded.
UNCARDED_IN_ROUND_49 = ("finance_aml_scorecard", "genomics_prs", "telecom_ids_student")


def _shipped() -> dict[str, set[str]]:
    """directory name -> tracked file names under `models/<dir>/`."""
    out = subprocess.run(["git", "ls-files", "--", "models"], cwd=REPO,
                         check=True, capture_output=True, text=True).stdout.split("\n")
    dirs: dict[str, set[str]] = defaultdict(set)
    for line in out:
        parts = line.strip().split("/")
        if len(parts) == 3 and parts[0] == "models":       # models/<dir>/<file>, not nested deeper
            dirs[parts[1]].add(parts[2])
    return dirs


def test_the_scan_sees_the_directories_it_is_about() -> None:
    """Non-vacuity. A scan that found nothing would make the assertion below pass trivially."""
    dirs = _shipped()
    assert len(dirs) >= 15, (
        f"found only {len(dirs)} shipped model directories; `git ls-files models` is not matching "
        "the tree this gate was written against")
    carded = [d for d, files in dirs.items() if "README.md" in files]
    assert len(carded) >= 12, (
        f"only {len(carded)} directories have a card, which is too few for the scan to be reading "
        "the repository correctly")


def test_every_directory_that_ships_an_artifact_ships_a_card() -> None:
    """The property. Red on the tree that added this gate, at three directories."""
    missing = []
    for name, files in sorted(_shipped().items()):
        artifacts = files - {"README.md"}
        if artifacts and "README.md" not in files:
            missing.append(f"models/{name}: ships {sorted(artifacts)} and no README.md")
    assert not missing, (
        "these directories ship a tracked artifact with no card describing it:\n  "
        + "\n  ".join(missing)
        + "\n\nA card is how a reader finds out what an artifact is, what corpus it was fitted on "
        "and what its licence permits. Both existing card gates enumerate CARDS, so a directory "
        "with none is invisible to them - which is why this one enumerates artifacts instead."
    )


def test_that_check_could_have_failed() -> None:
    """Anti-tautology, against the three real directories rather than an invented one.

    This gate is green once the cards are written, so on its own it demonstrates nothing. What makes
    it meaningful is that the condition it tests is the one that was FALSE for three directories on
    the tree that added it, and the predicate is re-applied here to a directory listing with the
    card removed - not to a hypothetical.
    """
    dirs = _shipped()
    for name in UNCARDED_IN_ROUND_49:
        assert name in dirs, (
            f"models/{name} no longer ships tracked files. If it was deleted, delete it from "
            "UNCARDED_IN_ROUND_49 too rather than leaving a name that silently matches nothing.")
        assert "README.md" in dirs[name], (
            f"models/{name} is still uncarded - this is the state the gate was written for")
        # The predicate, applied to this directory with its card taken away, must flag it. If it
        # does not, the check above is passing for some reason other than the card existing.
        without_card = dirs[name] - {"README.md"}
        assert without_card, (
            f"models/{name} ships ONLY a README.md, so it is not an artifact directory and cannot "
            "demonstrate anything about this gate")

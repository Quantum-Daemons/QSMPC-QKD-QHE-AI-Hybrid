"""A corpus under a licence that REQUIRES attribution must carry the credit line, whatever the spelling.

WHY THIS EXISTS

Round 49 added `DatasetSpec.attribution` and backfilled it for "all four CC BY corpora in the same
commit", explicitly to avoid this project's most-repeated defect - a fix reaching one of several
twins. The backfill matched on the string `CC-BY-4.0`.

`DS_ESA_ADB` is **`CC-BY-3.0-IGO`**. So the one corpus whose licence is spelled differently is the one
the backfill missed, and it stayed missed while `satellite_comms.telemanom` and
`satellite_comms.student` were both published to the Hub against it. CC BY 3.0 IGO requires
attribution wherever the work is distributed; an empty string is an unmet licence condition, not a
tidiness matter - and it is invisible because every OTHER CC BY corpus was fixed.

That is the eighth instance of the same class in this repository, and the mechanism this time is that
the earlier fix keyed on a LITERAL rather than on the property. This gate keys on the property.

WHAT IS ASSERTED

Every `DatasetSpec` whose licence is an attribution-requiring family carries a non-empty
`attribution`. The families are matched case-insensitively on the licence identifier, so a corpus
added tomorrow under `CC-BY-SA-4.0`, `CC-BY-NC`, `CC BY 4.0` or any IGO port is in scope on the day it
lands rather than on the day someone remembers.

WHAT IS DELIBERATELY NOT ASSERTED

Not that the attribution is CORRECT. `references/REFERENCES.md` records that thirteen places here once
credited OPS-SAT-AD to the wrong first author, and that the DOI and Zenodo id are the reliable
discriminators rather than the surname - a string check cannot catch that and should not pretend to.

Not that CC0 or public-domain corpora carry one. CC0 waives the requirement, and demanding a credit
line there would be inventing an obligation the licence does not impose.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.models.registry import _SPECS  # noqa: E402

#: Licence families that require attribution. CC BY in every version and every port, plus the
#: share-alike and non-commercial variants, which add conditions rather than removing this one.
#: CDLA-Sharing is included: its section 3.1 requires the attribution notices to travel with the data.
_REQUIRES_CREDIT = ("CC-BY", "CC BY", "CDLA-SHARING")

#: Licences that explicitly waive attribution. Listed so the gate's silence about them is a decision.
_WAIVES = ("CC0", "PUBLIC DOMAIN", "US GOVERNMENT WORK")


def _datasets() -> dict[str, object]:
    out = {}
    for spec in _SPECS:
        out.setdefault(spec.dataset.name, spec.dataset)
    return out


def test_the_scan_sees_the_datasets() -> None:
    """Non-vacuity: an empty set, or one with no attribution-requiring licence, proves nothing."""
    ds = _datasets()
    assert len(ds) >= 6, f"only {len(ds)} datasets resolved from the registry"
    requiring = [d for d in ds.values()
                 if any(k in (d.licence or "").upper() for k in _REQUIRES_CREDIT)]
    assert len(requiring) >= 4, (
        f"only {len(requiring)} datasets carry an attribution-requiring licence; this gate was "
        "written when five did, and a drop means the matcher has stopped seeing them")


def test_every_attribution_requiring_corpus_carries_its_credit() -> None:
    """The property. Red before Round 51 at `DS_ESA_ADB`, which is CC-BY-3.0-IGO."""
    missing = []
    for name, d in sorted(_datasets().items()):
        lic = (d.licence or "").upper()
        if not any(k in lic for k in _REQUIRES_CREDIT):
            continue
        if any(w in lic for w in _WAIVES):
            continue
        if not (getattr(d, "attribution", "") or "").strip():
            missing.append(f"{name}\n      licence: {d.licence}\n      publishable: {d.publishable}")
    assert not missing, (
        "these corpora are under a licence that REQUIRES attribution and carry none:\n  "
        + "\n  ".join(missing)
        + "\n\nAn empty credit line is an unmet licence condition, not untidiness, and it is worst "
        "where `publishable` is True because the obligation attaches wherever the data or a model "
        "trained on it is distributed. Round 49's backfill matched the literal `CC-BY-4.0` and so "
        "missed the one corpus spelled `CC-BY-3.0-IGO`; this gate matches the family instead."
    )

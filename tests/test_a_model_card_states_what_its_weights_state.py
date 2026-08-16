"""Every number a model card presents as a metric must be a number its own artifact records.

WHY THIS EXISTS. Nothing in this repository compared a figure on a card against the figure in the
file the card documents. `test_qcnn_usecases.py` checks that every artifact HAS a card and that every
card places its seed; `test_metric_baseline_consistency.py` checks that a baseline is not rendered
beside a differently-named metric; `test_qcnn_reachable.py` checks that `headline_metric` names a
field the artifact populates. All three are about the artifact or about the registry. None of them
opens the card and reads the number.

So the cards drifted, and they drifted in every direction a number can drift:

* `models/qcnn_telecom/README.md` published macro F1 0.614583 while `weights.json` said 0.458324,
  and printed accuracy as the SAME figure - a rewrite had written accuracy into the F1 field.
* `models/qcnn_finance_q9/README.md` had a banner saying its minority-class F1 was 0.976667 and a
  table two screens down saying 0.813559; 0.976667 was that table's accuracy.
* `models/qcnn_medical_q9/README.md` did the same in the other direction, printing its accuracy as
  the minority-class F1.
* `models/qcnn_telecom/README.md` carried a per-class table with SEVEN rows for EIGHT classes, one
  of them given as "~0.80" beside six-decimal siblings, all from a retired circuit.
* `models/qcnn_finance/README.md` and `models/qcnn_medical/README.md` still printed figures from the
  ansatz that Round 46 replaced, because Round 46 updated their 9-wire siblings and not them.

Every one of those is invisible to a check that reads only the artifact, and every one of them is
what a reader actually sees. A retrain updates `weights.json`; nothing updates the prose.

THE SCOPE IS GLOBBED, NOT LISTED. `models/*/README.md`, with the directory deciding which artifact it
is measured against. A gate whose scope names files is this project's most-repeated defect: the QCNN
option list went stale in both directions, `test_every_shipped_qcnn_card_places_its_seed` enumerated
cards rather than artifacts and so passed an artifact with no card at all, and the browser's width
control offered widths nobody had trained. A hand-listed subject is a second place to forget a file.

WHICH ARTIFACT. The registry already knows: `registry.ARTIFACT_DIRS` maps a model key to the
directory its files live in, and `models/metrics.json` is keyed by that same key. Inverting the map
gives directory -> record without a second copy of the mapping to rot. Those cards are GENERATED from
that record by `scripts/publish_hf.py::model_card`, so their table is the record by construction and
this gate is a regression check on the generator. The QCNN directories are in no registry entry, are
written by hand, and are checked against their own `weights.json`, which is where the drift was.

WHAT COUNTS AS "PRESENTED AS A METRIC VALUE", stated precisely because a matcher that cannot say
what it matches is a matcher nobody can trust:

  A. every value cell of every markdown table under a heading that begins with "Measured";
  B. every value cell of every markdown table whose first header cell is `metric` or `quantity`,
     wherever it sits, which is the shape `publish_hf.py` emits;
  C. every statement of the form `key` = number anywhere in the card, which is the headline-banner
     shape three of the six QCNN cards were wrong in.

The first cell of a row is its LABEL and is not read as a value: `| 8 wires | 0.4583 |` is a row
about 8 wires, not a claim that 8 is a metric.

WHAT IT DELIBERATELY DOES NOT COVER, and why that is not a hole to be closed by widening the regex.
Prose figures, and tables under their own headings, routinely and correctly hold numbers that are NOT
in this artifact: a published baseline carried from the literature, a seed sweep over ten runs, a
superseded circuit's spread kept as the evidence that a conclusion moved, the four configurations
Round 47 tried on telecom and none of which recovered its loss. Asserting those against this
directory's artifact would be asserting something false. What this gate refuses is the narrow thing
that was actually going wrong: a card's own measured table, and its own headline, disagreeing with
its own artifact. Figures from another run belong under their own heading, labelled as such, which is
where the rewritten cards put them.

ROUNDING IS ALLOWED, EXACTLY. A card may print 0.4583 for a stored 0.458324, because six digits in a
sentence are unreadable; the card's own precision decides the tolerance, so 0.4583 matches and 0.4584
does not. Nothing else is allowed: a truncation, a stale digit or a differently-rounded figure fails.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from qsh.models import registry

REPO = Path(__file__).resolve().parents[1]

#: A heading that opens a section of this artifact's own measurements. "## Measured performance",
#: "## Measured, seed 0" and "## Measured" are all in use. It must ANCHOR at the start of the
#: heading text: "## Published baselines this is measured against" contains the word and is a
#: section of figures carried from the literature, which are not this artifact's to record.
_MEASURED_HEADING = re.compile(r"^#{2,6}\s+Measured\b", re.I)
_HEADING = re.compile(r"^#{1,6}\s")

#: Not `\d+` alone: that would find the 8 in "arXiv:2412.00241" and the 2 in "14x14". The lookarounds
#: refuse a number glued to a word character or sitting inside a longer dotted run.
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])")

#: Rule C: "Current headline `macro_f1` = **0.458324**". Three of six QCNN cards had a banner in this
#: shape that disagreed with the table on the same card, so the banner is read as a claim.
_KEY_EQUALS = re.compile(r"`([A-Za-z0-9_]+)`\s*(?:=|:)\s*\*{0,2}(-?\d+(?:\.\d+)?)")

#: Header cells that declare a table to be a metric table wherever it sits. `publish_hf.py` emits
#: "| metric | value |"; the hand-written QCNN cards use "| quantity | value |".
_METRIC_TABLE_HEADERS = {"metric", "quantity"}

#: A cell carrying one of these is a citation, not a measurement, and its digits are an identifier.
_NOT_A_MEASUREMENT = ("arxiv", "http", "doi")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    cs = [c for c in _cells(line) if c]
    return bool(cs) and all(re.fullmatch(r":?-{2,}:?", c) for c in cs)


def _scalars(obj: object, prefix: str = "") -> list[tuple[str, float]]:
    """Numeric leaves reachable through dicts, with their dotted key path.

    LISTS ARE SKIPPED, and that is the point rather than an omission. `theta` is 77 arbitrary
    trained angles and `feature_mean` is eight corpus moments; admitting them would give a card 90
    free numbers to match against and this gate would pass on almost any figure. Every metric this
    project publishes is a scalar or sits in a small dict (`per_class_f1`, `macro_f1_floors`), so
    walking dicts only keeps the pool to things a card could honestly be quoting.
    """
    out: list[tuple[str, float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out += _scalars(v, path)
            elif isinstance(v, bool) or isinstance(v, (list, tuple)) or v is None:
                continue
            elif isinstance(v, (int, float)):
                out.append((path, float(v)))
    return out


def _artifact_for(card: Path) -> tuple[str, dict]:
    """(a name to print in a failure, the JSON) the card in this directory is measured against.

    Registry directories are checked against their `models/metrics.json` record, because that record
    is what `publish_hf.py` renders the card FROM. Everything else - the six QCNN directories - is
    checked against its own `weights.json`. The mapping is inverted from `registry.ARTIFACT_DIRS`
    rather than restated here: two copies of it would drift on the first new model, which is the
    reason the registry holds it in the first place.
    """
    directory = card.parent
    key_for_dir = {Path(v).name: k for k, v in registry.ARTIFACT_DIRS.items()}
    key = key_for_dir.get(directory.name)
    if key:
        records = json.loads((REPO / "models" / "metrics.json").read_text())
        if key in records:
            return f"models/metrics.json[{key}]", records[key]
    weights = directory / "weights.json"
    if weights.exists():
        return str(weights.relative_to(REPO)), json.loads(weights.read_text())
    raise AssertionError(
        f"{card.relative_to(REPO)} has no artifact to be checked against: its directory carries no "
        f"weights.json and its name maps to no key in registry.ARTIFACT_DIRS. A card documenting "
        "nothing is the failure mode this file exists for, so this is an error rather than a skip."
    )


def _claims(text: str) -> list[tuple[str, str]]:
    """(label, number-as-written) for every number the card presents as a metric value."""
    lines = text.splitlines()
    found: list[tuple[str, str]] = []
    in_measured = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if _HEADING.match(line):
            in_measured = bool(_MEASURED_HEADING.match(line))
            i += 1
            continue
        if not line.lstrip().startswith("|"):
            i += 1
            continue
        block: list[str] = []
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            block.append(lines[i])
            i += 1
        if len(block) < 2:
            continue
        header = _cells(block[0])
        metric_table = in_measured or header[0].strip("`* ").lower() in _METRIC_TABLE_HEADERS
        if not metric_table:
            continue
        for row in block[1:]:
            if _is_separator(row):
                continue
            cs = _cells(row)
            label = cs[0].strip("*` ") or "(unlabelled row)"
            for cell in cs[1:]:
                if any(w in cell.lower() for w in _NOT_A_MEASUREMENT):
                    continue
                found += [(label, m.group(0)) for m in _NUMBER.finditer(cell)]
    found += [(f"`{m.group(1)}` = ...", m.group(2)) for m in _KEY_EQUALS.finditer(text)]
    return found


def _matches(written: str, pool: list[tuple[str, float]]) -> bool:
    """Exact, or a correct rounding of an artifact value to the card's own stated precision."""
    x = float(written)
    decimals = len(written.split(".")[1]) if "." in written else 0
    return any(v == x or round(v, decimals) == x for _, v in pool)


def _nearest(written: str, pool: list[tuple[str, float]], n: int = 5) -> str:
    x = float(written)
    close = sorted(pool, key=lambda kv: abs(kv[1] - x))[:n]
    return ", ".join(f"{k} = {v}" for k, v in close) or "(the artifact records no numbers at all)"


#: Globbed, so a new model directory is in scope the moment it has a card.
CARDS = sorted((REPO / "models").glob("*/README.md"))


@pytest.mark.parametrize("card", CARDS, ids=lambda p: p.parent.name)
def test_every_metric_a_card_prints_is_a_metric_its_artifact_records(card: Path) -> None:
    name, artifact = _artifact_for(card)
    pool = _scalars(artifact)
    claims = _claims(card.read_text(encoding="utf-8"))

    assert claims, (
        f"{card.relative_to(REPO)} presents no numbers this gate can read as metric values. Every "
        "card in this repository has a measured section; if this one has stopped having one, the "
        "card has lost its figures or the section was renamed out of scope - either way a green "
        "result here would be green because nothing was checked."
    )

    wrong = [(label, written) for label, written in claims if not _matches(written, pool)]
    assert not wrong, (
        f"{card.relative_to(REPO)} prints {len(wrong)} figure(s) that {name} does not record:\n"
        + "\n".join(
            f"  row {label!r} says {written} - nearest values in the artifact: "
            f"{_nearest(written, pool)}"
            for label, written in wrong
        )
        + f"\n\nThe artifact is the measurement and the card is the description of it. A retrain "
        f"rewrites {name} and rewrites nothing here, so a card that disagrees with it is a card "
        "quoting a circuit that no longer exists. Fix the card to the artifact - never the other "
        "way round, and never by deleting the number, which makes the card look right by making it "
        "say less."
    )


@pytest.mark.parametrize(
    "card", [c for c in CARDS if (c.parent / "weights.json").exists()], ids=lambda p: p.parent.name
)
def test_a_card_prints_the_headline_its_artifact_declares(card: Path) -> None:
    """The negative check above passes on a card that prints nothing at all. This one does not.

    `headline_metric` is the field the browser and the API both resolve to decide which number to
    show a visitor. If the artifact says the headline is macro F1 and the card never prints the macro
    F1, then the most consequential figure on the page appears in no document - which is how the
    telecom card went a round showing a retired circuit's 0.614583 while the demo served 0.458324.

    The name-to-field mapping is the one `tests/test_qcnn_reachable.py` already pins.
    """
    field = {"macro_f1": "test_macro_f1",
             "minority_class_f1": "test_minority_f1",
             "accuracy": "test_accuracy"}
    artifact = json.loads((card.parent / "weights.json").read_text())
    headline = artifact.get("headline_metric")
    if headline not in field:
        pytest.skip(f"{card.parent.name} declares no headline metric this mapping covers")
    value = artifact[field[headline]]
    printed = [w for _, w in _claims(card.read_text(encoding="utf-8"))]
    assert any(float(w) == value or round(value, len(w.split(".")[1]) if "." in w else 0) == float(w)
               for w in printed), (
        f"{card.relative_to(REPO)} never prints its own headline. The artifact declares "
        f"headline_metric = {headline!r} and records {field[headline]} = {value}, and no metric cell "
        f"or banner on the card carries that number. Printed instead: {sorted(set(printed))}"
    )


def test_the_scan_found_a_plausible_number_of_cards_and_of_figures() -> None:
    """Non-vacuity. An empty glob and a regex that matches nothing both produce a green suite.

    Three ways this check can rot into a tautology, and one assertion against each: the glob stops
    finding cards; the extractor stops finding numbers in them; the artifact resolution quietly
    sends every card to the same place, so a whole class of card is unchecked. The floors are set
    below what is on disk at Round 47 - 16 cards, 6 of them QCNN, 175 figures - so a real new model
    does not trip them and a silent collapse does.
    """
    assert len(CARDS) >= 12, (
        f"only {len(CARDS)} model cards found under models/*/README.md; 16 were on disk when this "
        "was written. The glob has stopped finding them, and an empty scope passes every assertion "
        "in this file."
    )

    per_card = {c.parent.name: len(_claims(c.read_text(encoding="utf-8"))) for c in CARDS}
    silent = sorted(k for k, n in per_card.items() if n < 3)
    assert not silent, (
        f"these cards yielded fewer than 3 readable metric figures: {silent}. Every card in this "
        "repository has a measured table; a card the extractor cannot read is a card this gate "
        "cannot check, which is worse than a card it fails."
    )
    total = sum(per_card.values())
    assert total >= 100, (
        f"the scan read {total} metric figures across {len(CARDS)} cards; 175 were readable when "
        "this was written. A large drop means the extractor stopped matching a table shape rather "
        "than that the cards got shorter."
    )

    kinds = {_artifact_for(c)[0].startswith("models/metrics.json") for c in CARDS}
    assert kinds == {True, False}, (
        "every card resolved to the same kind of artifact. Both paths must stay exercised: the "
        "registry-generated cards against models/metrics.json, and the hand-written QCNN cards "
        "against their own weights.json, which is the pair that was drifting."
    )


def test_the_extractor_fails_a_card_that_states_a_number_its_artifact_does_not() -> None:
    """Negative control, because a matcher is only worth what it rejects.

    Every prose-matching check this project has written was tried against a deliberately broken
    subject before being trusted - the `**Seed position:**` marker exists because the version that
    searched for phrasings passed a card that had its disclosure DELETED, by matching a sentence
    about a sibling. So: take a real card, move one digit, and require the failure.
    """
    card = REPO / "models" / "qcnn_telecom" / "README.md"
    _, artifact = _artifact_for(card)
    pool = _scalars(artifact)
    text = card.read_text(encoding="utf-8")

    assert not [w for _, w in _claims(text) if not _matches(w, pool)], "precondition: card is green"

    tampered = text.replace("| `macro_f1` (this circuit) | 0.510963 |",
                            "| `macro_f1` (this circuit) | 0.614583 |")
    assert tampered != text, (
        "the row this control tampers with is no longer in the card verbatim. Point it at whatever "
        "row now carries the headline; a negative control that edits nothing proves nothing."
    )
    caught = [w for _, w in _claims(tampered) if not _matches(w, pool)]
    assert caught == ["0.614583"], (
        f"restoring the retired headline 0.614583 to the telecom card was not caught: {caught}. "
        "The extractor is not reading the measured table, so its green result on the real card "
        "means nothing."
    )

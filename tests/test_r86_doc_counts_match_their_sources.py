"""Counts the docs state in prose, held to the files they count (Round 86, batch 10).

Two had drifted unnoticed: `docs/threat-model.md` still said "eighteen published models" when the pin file held
twenty, and `docs/crypto-agility.md` said the CBOM "lists all 24" when it listed 25. No test read either sentence, so
each new model or algorithm left them behind. Each count below is read from the sentence and compared with the file
it counts: the published models with `models/hub_file_hashes.json` (one entry per published model), the CBOM's
algorithms with `web/public/cbom/qsh.cdx.json`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = {"twenty": 20, "thirty": 30, "forty": 40}


def _number(word: str) -> int:
    """An English number word up to forty-nine, as these sentences write it ("twenty", "twenty-one")."""
    if word in ONES:
        return ONES.index(word)
    head, _, tail = word.partition("-")
    return TENS[head] + (ONES.index(tail) if tail else 0)


def _published_models() -> int:
    pins = json.loads((REPO / "models" / "hub_file_hashes.json").read_text())
    return len([k for k in pins if not k.startswith("_")])


def test_the_number_words_read_as_numbers() -> None:
    assert [_number(w) for w in ("eighteen", "twenty", "twenty-one", "nineteen")] == [18, 20, 21, 19]


def test_the_threat_model_counts_the_published_models() -> None:
    text = (REPO / "docs" / "threat-model.md").read_text()
    m = re.search(r"this project's ([a-z-]+) published models", text)
    assert m, "the threat model no longer states the published models' count in this form"
    assert _number(m.group(1)) == _published_models(), (m.group(1), _published_models())


def test_the_readme_counts_the_published_models() -> None:
    text = (REPO / "README.md").read_text()
    m = re.search(r"^\| Models \| ([a-z-]+) published models", text, flags=re.M)
    assert m, "README's Models row no longer starts with the count"
    assert _number(m.group(1)) == _published_models(), (m.group(1), _published_models())


def test_crypto_agility_counts_the_cbom() -> None:
    cbom = json.loads((REPO / "web" / "public" / "cbom" / "qsh.cdx.json").read_text())
    m = re.search(r"the CBOM \(`web/public/cbom/qsh\.cdx\.json`\) lists all (\d+)\.", (REPO / "docs" / "crypto-agility.md").read_text())
    assert m, "crypto-agility.md no longer states the CBOM's count in this form"
    assert int(m.group(1)) == len(cbom["components"]), (m.group(1), len(cbom["components"]))

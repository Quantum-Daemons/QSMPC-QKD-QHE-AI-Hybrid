"""Every quotation of arXiv:2512.05069's withdrawal note is the note's own words (Round 86, the live audit of 2026-11-21).

The telecom baseline note quoted the withdrawal as 'insufficient to fully support the conclusions', from Round 17
(2026-08-03) on. The note on arXiv's v2 (2026-04-02), read through the arXiv API on 2026-11-21, says "the paper's
conclusions", and the four other places in the repository that quote it already said so; the registry's copy, which the
use-case page and both Hugging Face cards print, was the one that did not. The note is held here as read, so each
quotation of it is checked against it rather than against the others.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: The `arxiv:comment` of arXiv:2512.05069v2, read through the arXiv API on 2026-11-21.
WITHDRAWAL_NOTE = (
    "The authors have identified limitations in the experimental evaluation, which are insufficient to fully support "
    "the paper's conclusions. The manuscript is withdrawn pending additional experiments and analysis"
)
#: A single-quoted span may hold an apostrophe ("paper's"); it ends at a quote not followed by a letter.
QUOTED = re.compile(r"'((?:[^']|'(?=[a-z])){12,}?)'(?![a-z])|\"([^\"]{12,}?)\"")


def _flat(text: str) -> str:
    return " ".join(text.split())


def _quotes_near_the_citation(text: str) -> list[str]:
    """Quoted spans within 400 characters after each mention of the paper that use the note's wording."""
    out: dict[int, str] = {}
    for m in re.finditer(r"2512\.05069", text):
        for q in QUOTED.finditer(text, m.end(), m.end() + 400):
            span = _flat(q.group(1) or q.group(2))
            if "insufficient" in span or "limitations" in span:
                out[q.start()] = span
    return list(out.values())


def _sources() -> dict[str, str]:
    from qsh.models import registry

    texts = {
        "registry: telecom_ids.qml": registry.get("telecom_ids.qml").metric.baseline_source,
        "registry: telecom_ids.student": registry.get("telecom_ids.student").metric.baseline_source,
    }
    for rel in ("docs/findings.md", "references/REFERENCES.md", "scripts/train/telecom_ids.py",
                "web/src/lib/config.defaults.json"):
        texts[rel] = (REPO / rel).read_text(encoding="utf-8")
    return {k: _flat(v) for k, v in texts.items()}


def test_every_quotation_of_the_note_is_in_the_note() -> None:
    seen = 0
    for where, text in _sources().items():
        for span in _quotes_near_the_citation(text):
            seen += 1
            assert span in WITHDRAWAL_NOTE, f"{where} quotes {span!r}, which the withdrawal note does not say"
    # The registry twice (one note, two models), findings, REFERENCES, the training script, and the exported config
    # twice (the same two models).
    assert seen >= 7, seen


def test_the_shortened_quotation_does_not_come_back() -> None:
    for where, text in _sources().items():
        assert "insufficient to fully support the conclusions" not in text, where


def test_the_check_refuses_a_misquotation() -> None:
    text = "arXiv:2512.05069 was withdrawn for evaluation that was 'insufficient to fully support the conclusions'."
    assert _quotes_near_the_citation(text) == ["insufficient to fully support the conclusions"]
    assert "insufficient to fully support the conclusions" not in WITHDRAWAL_NOTE

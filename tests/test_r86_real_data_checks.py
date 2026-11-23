"""The real-data checks are recorded as they came out, and the notes quote the records (Round 86, 2026-11-22).

WHY THIS EXISTS

On the owner's choice of 2026-11-22 the finance pipeline was run on real transaction data with related labels
(references/aml-real-data-2026-11.md). BitcoinHeist's result is weak by design of the split (most test-window
ransomware families never appear in the fit window), and a weak result is the kind a later edit rounds up. The note's
figures are held to the record, the record to the file it read, and the encrypted path to its own agreement.

WHAT IS ASSERTED

- The record names its data, licence and credit, says what it is not, keeps 2018 apart, and was read from the file whose
  SHA-256 the script checks.
- The student's linear score under CKKS agrees with the clear score on every row it scored.
- The AML note quotes the record's figures at the precision it prints them.
- Seeded: a record whose encrypted decisions disagree, and a note that rounds the booster up, are refused.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "docs" / "measurements" / "bitcoinheist-check-2026-11-22.json"
NOTE = REPO / "references" / "aml-real-data-2026-11.md"


def record_problems(r: dict) -> list[str]:
    problems = []
    d = r["data"]
    if d["licence"] != "CC BY 4.0" or d["doi"] != "10.24432/C5BG8V" or "Akcora" not in d["citation"]:
        problems.append("the record does not carry the licence, the DOI and the credit")
    if "laundering" not in r["not"] or "detector" not in r["not"]:
        problems.append("the record does not say what it is not")
    if r["split"]["reported_apart"]["years"] != "2018":
        problems.append("2018 is not kept apart")
    ck = r["student_under_ckks"]
    if ck["decisions_agree"] != ck["rows"] or ck["max_abs_error"] > 1e-5:
        problems.append(f"under CKKS {ck['decisions_agree']} of {ck['rows']} decisions agree, error {ck['max_abs_error']}")
    return problems


def note_problems(note: str, r: dict) -> list[str]:
    flat = re.sub(r"\s+", " ", note)
    bt, st = r["booster_test"], r["student"]["test"]
    want = [f"minority-class F1 {bt['minority_class_f1']:.3f}", f"AUPRC {bt['auprc']:.3f}", f"{bt['auprc_baseline']:.3f}",
            f"AUROC {bt['auroc']:.2f}", f"AUROC {st['auroc']:.2f}"]
    return [f"the note does not quote {w!r}" for w in want if w not in flat]


def test_the_record_is_what_it_says() -> None:
    r = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record_problems(r) == []
    script = (REPO / "scripts" / "measure_bitcoinheist_check.py").read_text(encoding="utf-8")
    assert f'SHA256 = "{r["data"]["file_sha256"]}"' in script


def test_the_note_quotes_the_record() -> None:
    r = json.loads(RECORD.read_text(encoding="utf-8"))
    assert note_problems(NOTE.read_text(encoding="utf-8"), r) == []


def test_the_checks_refuse_a_disagreeing_record_and_a_rounded_note() -> None:
    r = json.loads(RECORD.read_text(encoding="utf-8"))
    bad = json.loads(json.dumps(r))
    bad["student_under_ckks"]["decisions_agree"] = bad["student_under_ckks"]["rows"] - 1
    assert record_problems(bad)
    note = NOTE.read_text(encoding="utf-8")
    rounded = note.replace(f"minority-class F1 {r['booster_test']['minority_class_f1']:.3f}", "minority-class F1 0.1")
    assert note_problems(rounded, r)

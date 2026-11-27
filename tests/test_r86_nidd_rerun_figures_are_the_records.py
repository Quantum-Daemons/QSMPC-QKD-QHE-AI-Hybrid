"""The 5G-NIDD re-run's figures, wherever they were copied by hand, are the record's (Round 86, batch 11).

Batch 10 wrote about fifteen of the re-run's figures into the registry notes of `telecom_ids.nidd` and
`telecom_ids.nidd_federated` (which the Models panel and the two Hub cards print) and batch 9 wrote its table into
`docs/models-and-benchmarks.md`; nothing read any of them back. Each is recomputed here from
`docs/measurements/telecom-nidd-clean-protocols-2026-11-27.json` or the page's file the exporter writes from it, and has
to appear, as written, where it was copied.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "docs" / "measurements" / "telecom-nidd-clean-protocols-2026-11-27.json"
PAGE = REPO / "web" / "public" / "data" / "telecom_nidd" / "clean_protocols.json"
SPLITS = ("P0", "P1_train_1_test_2", "P1_train_2_test_1", "P2", "P3")


def _metric(t: dict) -> float:
    return t["flamby_metric"] if len(t.get("station_macro_f1", {})) > 1 else t["macro_f1"]


def _r() -> dict:
    return json.loads(RECORD.read_text())


def _teacher(r: dict) -> dict:
    out: dict = {}
    for k, v in r["teacher"].items():
        s, p, _ = k.split("/")
        out.setdefault((s, p), []).append(_metric(v["test"]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def _sci(v: float) -> str:
    """2.23e-7 as the notes write it: one decimal, no padded exponent."""
    e = math.floor(math.log10(abs(v)))
    return f"{v / 10 ** e:.1f}e{e}"


def _notes(key: str) -> str:
    sys.path.insert(0, str(REPO / "src"))
    from qsh.models import registry

    return " ".join(registry.get(key).notes.split())


def test_the_teacher_note_is_the_record() -> None:
    r = _r()
    t = _teacher(r)
    note = _notes("telecom_ids.nidd")
    for phrase in (f"{t['T46', 'P0']:.4f} on the random split", f"{t['T46', 'P2']:.4f} on a time split",
                   f"{t['T46', 'P3']:.4f} on their own Seq rule", f"macro F1 {t['T46', 'P1_train_1_test_2']:.4f}",
                   f"tested on station 1, {t['T46', 'P1_train_2_test_1']:.3f}"):
        assert phrase in note, phrase
    six = max(abs(t["T46", p] - t["T40", p]) for p in SPLITS)
    assert six <= 0.0003 and "within 0.0003 on every split" in note, six
    ttl = sorted(t["T40", p] - t["T38", p] for p in SPLITS)[1:]          # four of the five splits
    assert f"TTL is worth {ttl[0]:.2f} to {ttl[-1]:.2f} on four of the five splits" in note, ttl
    ct = r["crosstestbed"]
    macro = float(np.mean([ct[f"teacher/seed{k}"]["macro_f1"] for k in range(5)]))
    binary = float(np.mean([ct[f"teacher/seed{k}"]["binary_attack_f1"] for k in range(5)]))
    assert f"macro F1 of {macro:.2f} on the classes they share ({binary:.2f} as Benign against any attack)" in note


def test_the_federation_note_is_the_record() -> None:
    r = _r()
    note = _notes("telecom_ids.nidd_federated")
    fed = {s: _metric(v["test"]) for s, v in r["federation"].items()}
    st = {k: _metric(v["test"]) for k, v in r["student"].items()}
    assert f"scores {fed['F36']:.4f} by the mean of the stations' macro F1" in note
    gap = max(abs(fed[s] - st[f"{s}/P2"]) for s in fed)
    assert gap <= 0.007 and f"within 0.007 of one linear model trained on both ({st['F36/P2']:.4f})" in note, gap
    assert f"without the six identifier-like columns {fed['F30']:.4f}, and without TTL as well {fed['F28']:.4f}" in note
    c = r["ckks_recheck"]
    assert f"on {c['rows']:,} of its test flows" in note and c["decisions_differ"] == 0
    assert f"no decision changes; largest logit error {_sci(c['max_abs_logit_error'])}" in note
    assert f"the linear model scores a macro F1 of {st['F36/P1_train_2_test_1']:.3f}" in note
    assert f"on the classes both corpora share, {r['crosstestbed']['student']['macro_f1']:.2f}" in note


def test_the_doc_table_is_the_page_file() -> None:
    page = json.loads(PAGE.read_text())
    rows = {("teacher", "T46"): "Teacher, all 46", ("teacher", "T40"): "Teacher, without the six",
            ("teacher", "T38"): "Teacher, without the six and TTL", ("student", "F36"): "Student, all 36",
            ("student", "F30"): "Student, without the six", ("student", "F28"): "Student, without the six and TTL",
            ("mlp", "F30"): "Square MLP (new), without the six", ("federation", "F36"): "Two-station federation, all 36",
            ("federation", "F30"): "Two-station federation, without the six",
            ("federation", "F28"): "Two-station federation, without the six and TTL"}
    show = lambda c: "" if c is None else (f"{c['mean']:.4f}" if c["sd"] is None else f"{c['mean']:.4f} ({c['sd']:.4f})")  # noqa: E731
    doc = (REPO / "docs" / "models-and-benchmarks.md").read_text()
    for (arm, inputs), label in rows.items():
        grid = page["arms"][arm][inputs]
        cells = [show(grid.get(p)) for p in SPLITS]
        row = "| " + label + " | " + " | ".join(cells) + " |"
        assert row.replace("  ", " ") in doc.replace("  ", " "), row

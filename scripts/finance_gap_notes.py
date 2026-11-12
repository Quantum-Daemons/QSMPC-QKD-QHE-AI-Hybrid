"""Rewrite the finance teacher's and MLP's "Gap to the published baseline" notes from their records, without retraining (Round 86).

Each note is written by its trainer (`teacher_gap_note` in `scripts/train/finance_aml.py`, `mlp_gap_note` in
`scripts/train/finance_aml_mlp.py`) from the run's own figures. When a note's wording changes, this writes the same function's
text into the record from the figures the record already holds, and changes nothing else in it. It refuses a record whose
figures the note cannot be built from.

    .venv/bin/python scripts/finance_gap_notes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts" / "train")]


def main() -> int:
    from finance_aml import teacher_gap_note  # type: ignore
    from finance_aml_mlp import mlp_gap_note  # type: ignore

    from qsh.models import registry, train

    teacher = dict(registry.achieved("finance_aml.gbm"))
    mlp = dict(registry.achieved("finance_aml.mlp_large"))
    notes = {
        "finance_aml.gbm": teacher_gap_note(int(teacher["n_features"]), float(teacher["minority_class_f1"]),
                                            int(teacher["batch_reach_minutes"])),
        "finance_aml.mlp_large": mlp_gap_note(mlp, teacher, int(mlp["batch_reach_minutes"])),
    }
    for key, record in (("finance_aml.gbm", teacher), ("finance_aml.mlp_large", mlp)):
        if record.get("baseline_gap_note") == notes[key]:
            print(f"  {key}: note already current")
            continue
        record["baseline_gap_note"] = notes[key]
        train.record_metrics(key, record)
        print(f"  {key}: note rewritten ({len(notes[key])} characters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

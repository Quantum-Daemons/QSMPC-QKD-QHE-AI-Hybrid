#!/usr/bin/env python3
"""Copy trained QCNN weights into the directory the browser serves them from.

WHY THIS EXISTS

`tests/test_qcnn_usecases.py::test_the_shipped_copy_agrees_with_the_trained_artifact` documents the
gap this closes, in its own words: "`qcnn_variational.py` writes only
`models/qcnn_<uc>/weights.json`. **Nothing copies it to** `web/public/data/<dir>/qcnn_weights.json`
- not the Makefile, not a script, not a documented step. The shipped copies were made by hand."

That is this round's defect class - one value living in two places, kept equal by memory - and
Round 47 hit it directly: retraining six models after two circuit corrections left every browser
copy on the previous circuit's angles, with the test the only thing standing between that and a
deployed page serving weights whose published metrics belong to a different model.

A test that detects the drift is worth more than nothing, but it still requires a human to perform
the copy correctly afterwards. This performs it.

WHAT IT PRESERVES

The shipped copies are deliberately TRIMMED subsets - the browser has no use for `n_train` or the
optimiser's bookkeeping - so this writes the intersection of the trained artifact's keys with the
key set the shipped copy already had, rather than the whole file. A new key therefore does not
start being served just because it was added to training, and the trimming stays a decision rather
than an accident of whatever the trainer last wrote.

Usage:  .venv/bin/python scripts/ship_qcnn_weights.py [--check]
        --check reports what would change and exits non-zero, for CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PUBLIC = REPO / "web" / "public" / "data"

#: `clientCompute.dataDirFor`: use case -> directory under public/data. `medical` is the one that
#: does not follow the `split("_")[0]` rule, and the exception lives in one place in the TS too.
USECASE_DIR = {"telecom": "telecom", "finance": "finance", "medical": "medical"}


def _pairs() -> list[tuple[Path, Path]]:
    """(trained artifact, shipped copy) for every `models/qcnn_*` that has a browser destination.

    Derived by globbing `models/`, not from a hardcoded list: a width trained tomorrow is shipped
    by the same command, and a model with no destination is reported rather than skipped silently.
    """
    out: list[tuple[Path, Path]] = []
    unmapped: list[str] = []
    for src in sorted(REPO.glob("models/qcnn_*/weights.json")):
        name = src.parent.name[len("qcnn_"):]          # e.g. "telecom", "telecom_q7", "medical_q9"
        base, _, suffix = name.partition("_")
        directory = USECASE_DIR.get(base)
        if directory is None:
            unmapped.append(src.parent.name)
            continue
        dest = PUBLIC / directory / (f"qcnn_weights_{suffix}.json" if suffix else "qcnn_weights.json")
        out.append((src, dest))
    if unmapped:
        print(f"note: no browser destination for {', '.join(unmapped)}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report differences and exit 1 instead of writing")
    a = ap.parse_args()

    pairs = _pairs()
    if not pairs:
        print("no models/qcnn_*/weights.json found; nothing to ship", file=sys.stderr)
        return 1

    stale: list[str] = []
    for src, dest in pairs:
        trained = json.loads(src.read_text())
        if not dest.exists():
            stale.append(f"{dest.relative_to(REPO)} does not exist")
            if not a.check:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(trained, indent=1) + "\n")
            continue
        shipped = json.loads(dest.read_text())
        # The intersection, so the deliberate trimming survives a retrain.
        merged = {k: trained[k] for k in shipped if k in trained}
        dropped = [k for k in shipped if k not in trained]
        if dropped:
            print(f"note: {dest.relative_to(REPO)} has keys the trained artifact lacks, kept as-is: "
                  f"{dropped}", file=sys.stderr)
            for k in dropped:
                merged[k] = shipped[k]
        if merged != shipped:
            changed = sorted(k for k in merged if merged[k] != shipped.get(k))
            stale.append(f"{dest.relative_to(REPO)}: {len(changed)} key(s) differ -> {changed[:6]}")
            if not a.check:
                dest.write_text(json.dumps(merged, indent=1) + "\n")

    if a.check:
        if stale:
            print("shipped QCNN weights are stale against models/:\n  " + "\n  ".join(stale))
            print("\nRun: .venv/bin/python scripts/ship_qcnn_weights.py")
            return 1
        print(f"shipped QCNN weights agree with models/ ({len(pairs)} artifacts)")
        return 0

    for line in stale:
        print("updated " + line)
    print(f"shipped {len(pairs)} QCNN artifact(s) to web/public/data/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""`config.yaml` shadows the dataclass, so every key it repeats must repeat the same value.

WHY THIS EXISTS

`config.yaml`'s own comment, written in Round 46, says it: "This file SHADOWS the dataclass and the
header above promises it shows the defaults, so a value changed there and not here makes that
promise false while every gate stays green. It did: raising the dataclass to 2_000_000 moved
nothing until this line changed too."

That paragraph diagnosed the defect exactly and then left it to a human to remember. It is this
round's defect class - a value that lives in two places stays equal only while someone recalls both
- and it is the fifth instance of that class found in four rounds. A comment is not a mechanism.

The scope is derived by walking whatever `config.yaml` actually contains, so a key added to the
file tomorrow is checked the day it is added, and a key REMOVED from the dataclass is reported
rather than silently skipped.
"""

from __future__ import annotations

import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _yaml() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((REPO / "config.yaml").read_text()) or {}


def _defaults() -> dict:
    """The dataclass defaults, as a nested dict, from the same loader the app uses."""
    from qsh.config.defaults import QSHConfig

    def walk(obj):
        out = {}
        for f in fields(obj):
            v = getattr(obj, f.name)
            out[f.name] = walk(v) if is_dataclass(v) else v
        return out

    return walk(QSHConfig())


def test_the_scan_has_a_subject() -> None:
    """Non-vacuity: an empty yaml or a failed import would make the comparison below pass."""
    y, d = _yaml(), _defaults()
    assert len(y) >= 3, f"config.yaml parsed to {len(y)} top-level sections; nothing to compare"
    shared = set(y) & set(d)
    assert len(shared) >= 2, (
        f"config.yaml and the dataclass share only {sorted(shared)} at the top level, so almost "
        "nothing is being checked"
    )
    assert "qkd" in shared, "the qkd section is the one this gate exists for and it is not present"


def test_every_shadowed_value_matches_the_dataclass() -> None:
    """The property: if config.yaml names a key the dataclass defines, the values are equal."""
    y, d = _yaml(), _defaults()
    mismatches: list[str] = []
    orphans: list[str] = []

    def compare(path: str, yv, dv) -> None:
        # YAML gives lists where the dataclass holds tuples; that is a serialisation difference,
        # not a configuration difference, so sequences are compared by contents.
        if isinstance(yv, (list, tuple)) and isinstance(dv, (list, tuple)):
            if list(yv) != list(dv):
                mismatches.append(f"{path}: config.yaml {list(yv)!r} vs dataclass {list(dv)!r}")
            return
        if isinstance(yv, dict):
            for k, v in yv.items():
                if not isinstance(dv, dict) or k not in dv:
                    orphans.append(f"{path}.{k}" if path else k)
                    continue
                compare(f"{path}.{k}" if path else k, v, dv[k])
            return
        if isinstance(yv, float) or isinstance(dv, float):
            try:
                if abs(float(yv) - float(dv)) > 1e-12 * max(1.0, abs(float(dv))):
                    mismatches.append(f"{path}: config.yaml {yv!r} vs dataclass {dv!r}")
            except (TypeError, ValueError):
                if yv != dv:
                    mismatches.append(f"{path}: config.yaml {yv!r} vs dataclass {dv!r}")
        elif yv != dv:
            mismatches.append(f"{path}: config.yaml {yv!r} vs dataclass {dv!r}")

    compare("", y, d)
    assert not mismatches, (
        "config.yaml shadows the dataclass with DIFFERENT values:\n  " + "\n  ".join(mismatches)
        + "\n\nThe file's header promises it shows the defaults. Change both, or neither."
    )
    assert not orphans, (
        "config.yaml sets keys the dataclass does not define:\n  " + "\n  ".join(sorted(orphans))
        + "\n\nEither the dataclass lost a field and this file kept it - in which case the value is "
        "silently ignored at runtime - or the key is misspelled."
    )

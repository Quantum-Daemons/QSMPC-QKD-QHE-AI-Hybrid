#!/usr/bin/env python
"""Assert that every prose field in models/metrics.json still matches its GENERATOR.

WHY THIS EXISTS, and why the existing parity gate cannot do it

`models/metrics.json` is written by the training scripts. `web/src/lib/config.defaults.json` is
derived from it by `scripts/export_config.py`. CI step 2/8 compares those two, which catches drift
between the two DERIVED artifacts and is blind by construction to the case where both derived copies
agree with each other and neither agrees with the code that produced them.

That case happened. Round 32 judged the `medical_fl.student` `baseline_gap_note` an overclaim ("a
CKKS-feasibility budget, not a diagnostic one"), corrected it by hand in metrics.json and in
config.defaults.json, and left `scripts/train/medical_breastmnist.py` emitting the original. Both
derived sides agreed, so step 2/8 passed. Three things would have followed from the next retrain:
the correction reverts silently, `scripts/publish_hf.py` pushes the reverted sentence into a public
Hugging Face model card, and the ledger entry recording the correction becomes false without anyone
editing it. The project had already named this defect class - "a hand-patched generated artifact is
a defect with a timer on it" - one round before committing it again.

Retraining to check is not an option: these scripts need a GPU and hours, and the point of a gate is
that it runs in seconds on every push. So this reads the generator's string literal STATICALLY, out
of the AST, and compares it to the recorded value. No model runs.

WHAT IT CANNOT SEE, stated so the pass is not read as more than it is
    A note assembled at runtime from measured values (an f-string, a join, a concatenation with a
    variable) has no static literal to compare, so it is reported as UNCHECKABLE and does not fail
    the gate. That is a real hole and the count is printed on every run rather than hidden: a field
    that moves from a literal to an f-string silently leaves this gate's coverage, and the printed
    count is what makes that visible.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "models" / "metrics.json"
TRAINERS = sorted((ROOT / "scripts" / "train").glob("*.py"))

#: Free-text fields whose wording carries a CLAIM rather than a measurement. These are the ones worth
#: gating: a number that drifts shows up in a test, a sentence that drifts shows up nowhere. Keys are
#: matched exactly against the dict passed to `record_metrics`.
PROSE_FIELDS = frozenset({
    "baseline_gap_note",
    "metric_note",
    "sample_size_note",
    "classical_rf_attribution_note",
    "framing",
    "note",
})


def _module_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "..."`` bindings, so a note hoisted to a constant still resolves."""
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):
                    continue
                if isinstance(value, str):
                    out[target.id] = value
    return out


def _static_str(node: ast.expr, constants: dict[str, str]) -> str | None:
    """The literal string this expression denotes, or None if it is only knowable at runtime."""
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None
    return value if isinstance(value, str) else None


def _record_metrics_calls(tree: ast.Module) -> list[tuple[str, ast.Dict]]:
    """Every ``record_metrics("<key>", {...})`` in the module, as (key, dict-node) pairs."""
    found: list[tuple[str, ast.Dict]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "record_metrics":
            continue
        key = _static_str(node.args[0], {})
        if key is not None and isinstance(node.args[1], ast.Dict):
            found.append((key, node.args[1]))
    return found


def main() -> int:
    recorded = json.loads(METRICS.read_text())
    checked = mismatched = uncheckable = 0
    problems: list[str] = []
    holes: list[str] = []

    for path in TRAINERS:
        tree = ast.parse(path.read_text())
        constants = _module_constants(tree)
        rel = path.relative_to(ROOT)
        for metrics_key, dict_node in _record_metrics_calls(tree):
            for key_node, value_node in zip(dict_node.keys, dict_node.values, strict=True):
                field = _static_str(key_node, {}) if key_node is not None else None
                if field not in PROSE_FIELDS:
                    continue
                generated = _static_str(value_node, constants)
                if generated is None:
                    uncheckable += 1
                    holes.append(f"{rel}: {metrics_key}.{field} is built at runtime")
                    continue
                if metrics_key not in recorded or field not in recorded[metrics_key]:
                    # A generator emits it and the artifact has no such record. Either the trainer
                    # has never been run since the field was added, or the key was renamed.
                    mismatched += 1
                    problems.append(
                        f"{rel}: emits {metrics_key}.{field}, which metrics.json does not record"
                    )
                    continue
                checked += 1
                if recorded[metrics_key][field] != generated:
                    mismatched += 1
                    problems.append(
                        f"{rel}: {metrics_key}.{field} DIFFERS from the recorded value.\n"
                        f"      generator: {generated[:160]}...\n"
                        f"      artifact : {recorded[metrics_key][field][:160]}..."
                    )

    for hole in holes:
        print(f"  ~ uncheckable  {hole}")
    for problem in problems:
        print(f"  x {problem}")

    print(
        f"\n{checked} prose field(s) verified against their generator, "
        f"{uncheckable} not statically checkable, {mismatched} mismatched"
    )
    if not checked and not uncheckable:
        # The empty-snapshot failure this project has now hit twice: a gate that passes because it
        # found nothing to test. Matching zero generators means the parse or the field list is wrong.
        print("FAIL: no prose fields were found at all - this gate did not run")
        return 1
    if mismatched:
        print(
            "FAIL: a generated artifact was edited by hand without its generator.\n"
            "      Fix the trainer, not the artifact: the artifact is regenerated on the next run,\n"
            "      and publish_hf.py copies these fields into public model cards."
        )
        return 1
    print("OK: every statically-readable prose field matches its generator")
    return 0


if __name__ == "__main__":
    sys.exit(main())

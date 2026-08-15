"""Everything a run writes must be cleared when the use case changes.

WHY THIS EXISTS

`usePipelineRun.ts` carried this comment, and then implemented one member of the set it describes:

    Anything keyed to the use case has to be reset when the use case changes, or gated at the
    point of use; this does both.
    useEffect(() => { setPublishedModel(""); }, [usecase]);

`publishedModel` was cleared. The nine fields a run writes were not, because `reset()` is only
called from `launch()`. Measured on the deployed site in Round 43: run `finance_aml` on
`/orchestration`, switch the selector to `demo`, and the QHE/FHE node still reads
`(finance_aml.gbm on wasm)`. A fresh load with `demo` selected reads `Private Inference`, so the
caption is correct until a run happens and stale afterwards.

The caption was the visible corner. With `metrics`, `events` and `artifacts` surviving,
`Orchestration.tsx` also offered the previous run's events for export **under the new use case's
filename**, and drew the old encrypted-agreement chart under the new selection.

WHY THE SET IS DERIVED AND NOT LISTED

A hand-written list of "fields a run writes" is a second place to forget one - and forgetting one is
the defect. So this reads the source: a field is run-derived if its setter is called inside
`handleEvent`, inside either `onEnd` block, or by `launch` itself. `reset()` must clear every one.

That is what makes this non-tautological. "Assert `reset()` appears in the `[usecase]` effect" would
be the fix restated in a different file. The *membership* of the set comes from a different part of
the source than the part being fixed, so the gate can disagree with the fix.

KNOWN LIMIT, STATED

This is a source-text check. It cannot see whether React actually re-runs the effect, nor whether a
surface reads a field this file never resets. The independent check that belongs beside it is the
deployed page: switch use case after a run and read the DAG caption, the export filename, the
encrypted-agreement card and the state badge. Round 43 records that measurement with its bundle hash.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "web" / "src" / "lib" / "usePipelineRun.ts"

_COMMENT = re.compile(r"/\*[\s\S]*?\*/|//[^\n]*")


def _code_only(text: str) -> str:
    """Comments blanked, line count preserved.

    The comments here quote the setter names while explaining the defect, so a scan that reads them
    would derive its "run-derived" set partly from prose about the bug. Round 42 shipped exactly that
    in `test_a_frame_is_not_drawn_around_nothing.py`.
    """
    return _COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def _state_fields(code: str) -> dict[str, str]:
    """`setter name -> state name`, for every `const [x, setX] = useState(...)`."""
    return {
        m.group(2): m.group(1)
        for m in re.finditer(r"const \[(\w+), (set\w+)\] = useState", code)
    }


def _block(code: str, start: str) -> str:
    """From `start` to the end of its brace-balanced body."""
    i = code.index(start)
    j, depth, seen = i, 0, False
    while j < len(code):
        if code[j] == "{":
            depth += 1
            seen = True
        elif code[j] == "}":
            depth -= 1
            if seen and depth == 0:
                return code[i: j + 1]
        j += 1
    return code[i:]


def _run_derived(code: str) -> set[str]:
    """Setters the RUN calls: from `handleEvent`, from either `onEnd`, and from `launch` itself."""
    regions = [_block(code, "const handleEvent"), _block(code, "const launch")]
    called: set[str] = set()
    for r in regions:
        called |= set(re.findall(r"\b(set[A-Z]\w*)\s*\(", r))
    return called


def test_the_parse_finds_the_state_and_the_writers() -> None:
    """A parse that finds nothing would pass every assertion below for the wrong reason."""
    code = _code_only(SRC.read_text())
    fields, writers = _state_fields(code), _run_derived(code)
    assert len(fields) >= 15, f"only {len(fields)} useState declarations parsed; has the hook moved?"
    run_state = {f for f in writers if f in fields}
    assert len(run_state) >= 7, (
        f"only {len(run_state)} run-derived state fields found ({sorted(run_state)}); the derivation "
        "has stopped working, and an empty set makes the real check vacuous"
    )


def test_reset_clears_every_field_the_run_writes() -> None:
    """`reset()` must be complete, or the `[usecase]` effect cannot be."""
    code = _code_only(SRC.read_text())
    fields = _state_fields(code)
    reset = _block(code, "const reset =")
    cleared = set(re.findall(r"\b(set[A-Z]\w*)\s*\(", reset))
    missing = sorted(fields[s] for s in _run_derived(code) if s in fields and s not in cleared)
    assert not missing, (
        f"`reset()` does not clear these fields that a run writes: {missing}.\n"
        "From `launch()` that is invisible - the next statements set them - but the `[usecase]` "
        "effect has no follow-up, so the badge would read `done` over an empty log. Clearing seven "
        "of nine is the same defect one layer in."
    )


def test_a_use_case_change_resets_and_cancels() -> None:
    """The effect keyed on `[usecase]` must call `reset()` and stop the in-flight run."""
    code = _code_only(SRC.read_text())
    effects = [
        _block(code, m.group(0))
        for m in re.finditer(r"useEffect\(\(\) => \{", code)
    ]
    keyed = [e for e in effects if re.search(r"\}, \[usecase\]\)|\[usecase\]\)\s*;?\s*$", e)]
    # The effect body and its dependency array are separate braces, so pick by what follows the body.
    idx = [m.start() for m in re.finditer(r"useEffect\(", code)]
    keyed = []
    for i in idx:
        tail = code[i: i + 1400]
        if re.search(r"\}\s*,\s*\[\s*usecase\s*\]\s*\)", tail):
            keyed.append(tail[: tail.index("[usecase]")])
    assert keyed, "no useEffect keyed on [usecase] found; the reset has been removed entirely"
    body = "\n".join(keyed)
    assert "reset()" in body, (
        "the [usecase] effect does not call reset(). Clearing `publishedModel` alone is what left the "
        "previous run's model in the DAG caption, its events under the new use case's export "
        "filename, and its agreement chart under the new selection."
    )
    assert "stopRef.current" in body and "runTokenRef.current" in body, (
        "the [usecase] effect does not cancel the in-flight run. `/usecase/:id` navigation is not "
        "blocked by `inFlight`, so a stage still running can write into the use case the visitor "
        "just switched to - which reset() alone would not prevent."
    )


def test_the_gate_fires_when_a_new_run_field_is_added_and_not_reset() -> None:
    """Seed the SET, because the set is the side that was wrong.

    A synthetic state field written from `handleEvent` and absent from `reset()` must be reported.
    The negative control matters as much: `setKem` is never written by the run, so demanding it
    would make the gate require wiping the crypto selects on every use-case change - a different
    defect, introduced by a gate meant to prevent one.
    """
    code = _code_only(SRC.read_text())
    fields = _state_fields(code)
    assert "setKem" in fields, "setKem is the negative control and it has moved"
    assert "setKem" not in _run_derived(code), (
        "setKem is now written by the run; pick another control, or the assertion below proves nothing"
    )

    seeded = code.replace(
        "const [runId, setRunId] = useState<string | null>(null);",
        "const [runId, setRunId] = useState<string | null>(null);\n"
        "  const [seededField, setSeededField] = useState<number>(0);",
    ).replace(
        'if (e.type === "pipeline_resume") setState("running");',
        'if (e.type === "pipeline_resume") setState("running");\n'
        "    setSeededField(1);",
    )
    fields2 = _state_fields(seeded)
    reset2 = _block(seeded, "const reset =")
    cleared2 = set(re.findall(r"\b(set[A-Z]\w*)\s*\(", reset2))
    missing2 = {fields2[s] for s in _run_derived(seeded) if s in fields2 and s not in cleared2}
    assert "seededField" in missing2, (
        "a state field written by handleEvent and never cleared by reset() was NOT reported. The "
        "derivation has stopped reading the writers, so the gate would go green on the next field "
        "somebody forgets."
    )
    assert "kem" not in missing2, (
        "the crypto selection was demanded as run-derived; the gate would require wiping the KEM "
        "choice whenever the use case changes"
    )
